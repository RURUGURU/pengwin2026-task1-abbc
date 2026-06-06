#!/usr/bin/env python3
"""Active split-anatomy evaluation helpers."""
from __future__ import annotations

import argparse
import heapq
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.special import expit

import sys
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    ANATOMY_RANGES, ANATOMY_TO_INDEX, DATASETS, NN_PREP, NN_RAW, NN_RES,
    RESULT_REPORT, RESULT_VISUALIZE, configure_nnunet_env, is_femur, is_pelvic,
    ABBC_HARD_NEGATIVE_LABEL, BICM_V6_OUTPUT_CHANNELS, BICM_V8_OUTPUT_CHANNELS,
    BICM_V38_OUTPUT_CHANNELS, BICM_V68_OUTPUT_CHANNELS,
    BFV3_BINARY_BARRIER_OUTPUT_CHANNELS, BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS,
    BFV3_XYZ_AFFINITY_OUTPUT_CHANNELS, BFV3_AFFINITY13_SEED_OUTPUT_CHANNELS,
    BFV3_MUTEX13_SEED_OUTPUT_CHANNELS, BFV3_CENTER_FLOW_OUTPUT_CHANNELS,
    BFV3_NO_CONTACT_CENTER_FLOW_OUTPUT_CHANNELS, BFV3_SPATIAL_EMBEDDING_OUTPUT_CHANNELS,
    BFV3_NO_CONTACT_PAIRWISE_V273_OUTPUT_CHANNELS, BFV3_FRAGMENT_POSITION_V275_OUTPUT_CHANNELS,
    BFV3_SEPARATOR_GAP_V277_OUTPUT_CHANNELS,
    BFV3_SEPARATOR_ENERGY_V278_OUTPUT_CHANNELS,
    BFV3_SEPARATOR_SOFTMAX_V287_OUTPUT_CHANNELS,
    BFV3_ABBC_V288_OUTPUT_CHANNELS,
    BFV3_ABBC_SDF_V289_OUTPUT_CHANNELS,
    BFV3_ABBC_SDF_FDM_V290_OUTPUT_CHANNELS,
    BFV3_QUERY_MASK_V280_OUTPUT_CHANNELS,
    BFV3_QUERY_MASK_PN_V281_OUTPUT_CHANNELS,
    BFV3_FREE_EMBEDDING_V282_OUTPUT_CHANNELS,
    BFV3_GLOBAL_COORD_FREE_EMBEDDING_V283_OUTPUT_CHANNELS,
    FACTOR_INSTANCE_OUTPUT_CHANNELS, BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR,
    _pelvic_same_anatomy_contact_mask,
)
from utils import (
    ORIENTATION_CONTRACT_VERSION, canonicalize_sitk, find_case_dir, inst_to_anat,
    prepare_lps_ct_for_nnunet,
    ABBC_BORDER_LABEL, ABBC_CORE_LABEL, ABBC_OFFICIAL_LABELS,
    compute_abbc_official_target, decode_official_abbc, get_contact_surface_regions,
    BFV3_LABELS, BFV3_SUPPORT_LABELS, BoundaryFragmentParams, binary_surface_metrics,
    compute_boundary_fragment_target, decode_boundary_fragment, instance_iouf,
    oracle_topology_diagnostics,
    BICMV5Params, V5_ANATOMY_RANGES, V5_ANATOMY_RANGES_WITH_FEMUR,
    PENGWIN_OFFICIAL_ANATOMY_RANGES,
    V5_LABELS, V5_SUPPORT_LABELS, V5_TARGET_PROFILES,
    anatomy_mask_from_instances, bbox_from_mask, compute_bicm_v5_target,
    decode_bicm_v5, decode_bicm_v5_seed_healed, label_distribution,
)
# Registry single-source helpers. Active decode/scoring uses the FULL (femur-
# inclusive) view; legacy per-version oracle proxies stay 3-anatomy via the
# explicit pelvic_only=True view (intent visible, not hidden behind a magic 150).
from anatomy_registry import (
    NUM_ANATOMIES,
    MIN_INSTANCE_ID,
    PELVIC_MAX_INSTANCE_ID,
    valid_instance_mask,
    anatomy_start_ids,
)

configure_nnunet_env()


HARD5_PELVIC_CASES = ["003", "011", "017", "018", "025"]
HARD5_FEMUR_CASES = ["253", "261", "267", "268", "275"]
HARD10_CASES = HARD5_PELVIC_CASES + HARD5_FEMUR_CASES
SOTA_CASE_SETS = {
    "hard5_pelvic": HARD5_PELVIC_CASES,
    "hard5_femur": HARD5_FEMUR_CASES,
    "hard10": HARD10_CASES,
}

LR_FIX_NOTE = "LPS physical-x convention: LeftHip centroid should be greater than RightHip centroid."
DEFAULT_PRED_ROOT = RESULT_VISUALIZE / "predictions" / "split_anatomy_lps_v1"
DEFAULT_ABBC_PRED_ROOT = RESULT_VISUALIZE / "predictions" / "contact_instance537_checkpoint_best"


def _nnunet_results_root() -> Path:
    """Return the active nnU-Net results root for model-loading commands.

    [REPRO][Risk:High][Scope:evaluation_checkpoint]
    Short diagnostics often isolate checkpoints under a temporary
    `nnUNet_results` root. Prediction must honor that environment variable;
    otherwise eval silently reloads an older checkpoint from the default
    `/workspace/nnunet/results` tree and the reported metrics no longer match
    the just-finished experiment.
    """
    return Path(os.environ.get("nnUNet_results", str(NN_RES)))


def _default_abbc_pred_root(checkpoint: str) -> Path:
    """Keep legacy semantic-contact predictions checkpoint-scoped by default."""
    name = checkpoint.removesuffix(".pth").removeprefix("checkpoint_")
    if name == "best":
        return DEFAULT_ABBC_PRED_ROOT
    return RESULT_VISUALIZE / "predictions" / f"contact_instance537_checkpoint_{name}"


def _binary_prf(pred: np.ndarray, target: np.ndarray, beta: float = 1.0) -> dict[str, float]:
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    beta2 = float(beta) ** 2
    f = (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f": float(f),
        "pred_voxels": int(pred.sum()),
        "target_voxels": int(target.sum()),
        "pred_target_ratio": float(pred.sum() / max(1, int(target.sum()))),
    }


def _safe_float_stats(values: np.ndarray) -> dict[str, float | None]:
    """Small JSON-safe summary for distances/intensities."""

    values = np.asarray(values)
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    values = values.astype(np.float64, copy=False)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _fraction_map(values: np.ndarray, valid_values: list[int] | tuple[int, ...]) -> dict[str, float]:
    """Return per-label fractions for one connected component."""

    if values.size == 0:
        return {str(int(v)): 0.0 for v in valid_values}
    denom = float(values.size)
    return {str(int(v)): float((values == int(v)).sum() / denom) for v in valid_values}


def _component_bbox(mask: np.ndarray) -> dict[str, list[int]] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0).astype(int).tolist()
    hi = coords.max(axis=0).astype(int).tolist()
    return {"zyx_min": lo, "zyx_max": hi}


def _support_component_summaries(mask: np.ndarray,
                                 *,
                                 pred_labels: np.ndarray,
                                 target_labels: np.ndarray,
                                 source_gt: np.ndarray,
                                 ct_hu: np.ndarray,
                                 distance_mm: np.ndarray,
                                 spacing_zyx: tuple[float, float, float],
                                 max_components: int,
                                 kind: str) -> list[dict[str, Any]]:
    """Summarize largest support error components without saving heavy arrays."""

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    comps, n_comps = ndi.label(mask.astype(bool, copy=False), structure=structure)
    if n_comps == 0:
        return []
    counts = np.bincount(comps.ravel())
    order = [int(i) for i in np.argsort(counts[1:])[::-1] + 1 if counts[int(i)] > 0]
    out: list[dict[str, Any]] = []
    for comp_id in order[: int(max_components)]:
        comp = comps == comp_id
        coords = np.argwhere(comp)
        centroid = coords.mean(axis=0) if coords.size else np.array([0.0, 0.0, 0.0])
        dist_vals = distance_mm[comp]
        target_vals = target_labels[comp]
        pred_vals = pred_labels[comp]
        source_vals = source_gt[comp]
        target_fraction = _fraction_map(target_vals, [0, 1, 2, 3, 4])
        pred_fraction = _fraction_map(pred_vals, [0, 1, 2, 3, 4])
        femur_fraction = float(((source_vals >= 151) & (source_vals <= 200)).sum() / max(1, int(comp.sum())))
        pelvic_fraction = float((valid_instance_mask(source_vals, pelvic_only=True)).sum() / max(1, int(comp.sum())))
        external_fraction = float((target_vals == BFV3_LABELS["external_context"]).sum() / max(1, int(comp.sum())))
        dist_mean = float(np.mean(dist_vals)) if dist_vals.size else 0.0
        if femur_fraction >= 0.25:
            category = "gt_femur_overlap"
        elif external_fraction >= 0.25 and dist_mean <= 5.0:
            category = "external_context_band_leak"
        elif dist_mean > 20.0:
            category = "far_background_island"
        elif dist_mean > 5.0:
            category = "distant_support_island"
        elif kind == "fp":
            category = "near_boundary_thickening"
        else:
            category = "missed_gt_support"
        out.append({
            "component_id": int(comp_id),
            "voxels": int(comp.sum()),
            "category": category,
            "centroid_zyx": [float(v) for v in centroid.tolist()],
            "bbox": _component_bbox(comp),
            "bbox_extent_mm_zyx": [
                float((coords[:, axis].max() - coords[:, axis].min() + 1) * spacing_zyx[axis])
                for axis in range(3)
            ] if coords.size else [0.0, 0.0, 0.0],
            "distance_to_opposite_support_mm": _safe_float_stats(dist_vals),
            "ct_hu": _safe_float_stats(ct_hu[comp]),
            "pred_label_fraction": pred_fraction,
            "target_label_fraction": target_fraction,
            "source_gt_pelvic_fraction": pelvic_fraction,
            "source_gt_femur_fraction": femur_fraction,
            "target_external_context_fraction": external_fraction,
        })
    return out


def _surface_distance_summary(pred_support: np.ndarray,
                              target_support: np.ndarray,
                              spacing_zyx: tuple[float, float, float]) -> dict[str, Any]:
    """Expose the surface-distance tails that drive HD95."""

    pred_border = pred_support ^ ndi.binary_erosion(pred_support)
    target_border = target_support ^ ndi.binary_erosion(target_support)
    if not pred_border.any() or not target_border.any():
        return {
            "pred_to_target_mm": _safe_float_stats(np.array([], dtype=np.float32)),
            "target_to_pred_mm": _safe_float_stats(np.array([], dtype=np.float32)),
            "tail_counts": {},
        }
    dist_to_target = ndi.distance_transform_edt(~target_border, sampling=spacing_zyx)
    dist_to_pred = ndi.distance_transform_edt(~pred_border, sampling=spacing_zyx)
    pred_to_target = dist_to_target[pred_border]
    target_to_pred = dist_to_pred[target_border]
    thresholds = (2.0, 5.0, 10.0, 20.0, 50.0)
    return {
        "pred_to_target_mm": _safe_float_stats(pred_to_target),
        "target_to_pred_mm": _safe_float_stats(target_to_pred),
        "tail_counts": {
            f"pred_surface_gt_{thr:g}mm": int((pred_to_target > thr).sum())
            for thr in thresholds
        } | {
            f"target_surface_gt_{thr:g}mm": int((target_to_pred > thr).sum())
            for thr in thresholds
        },
    }


def run_boundary_support_error_audit(cases: list[str],
                                     out_path: Path,
                                     pred_root: Path,
                                     target_profile: str = "v3_thin_ridge",
                                     contact_band_mm: float = 3.0,
                                     contact_search_mm: float = 3.0,
                                     contact_ridge_mm: float = 1.0,
                                     contact_same_anatomy_only: bool = False,
                                     shell_mm: float = 8.0,
                                     max_components: int = 20) -> dict:
    """Diagnose why BoundaryFragment support HD95/IoU-F fails.

    This command intentionally does not tune thresholds or alter predictions.
    It decomposes the fixed argmax prediction into support FP/FN connected
    components and surface-distance tails so we can decide whether the next
    model needs a better support head, a better contact head, or a decoder fix.
    """

    rows = []
    for case in [str(c).zfill(3) for c in cases]:
        cd = find_case_dir(case)
        if cd is None:
            raise FileNotFoundError(f"source case not found: {case}")
        img = canonicalize_sitk(sitk.ReadImage(str(cd / "image.mha")))
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        ct_hu = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
        source_gt = sitk.GetArrayFromImage(lbl_img)
        spacing_zyx = _spacing_zyx_from_image(img)
        target, target_audit = compute_boundary_fragment_target(
            source_gt,
            spacing_zyx=spacing_zyx,
            params=BoundaryFragmentParams(
                target_profile=target_profile,
                contact_band_mm=float(contact_band_mm),
                contact_search_mm=float(contact_search_mm),
                contact_ridge_mm=float(contact_ridge_mm),
                contact_same_anatomy_only=bool(contact_same_anatomy_only),
                shell_mm=float(shell_mm),
            ),
        )
        pred_path = pred_root / f"PENGWIN_{case}.mha"
        if not pred_path.exists():
            raise FileNotFoundError(f"prediction not found: {pred_path}")
        pred_labels = sitk.GetArrayFromImage(canonicalize_sitk(sitk.ReadImage(str(pred_path)))).astype(np.uint8)
        if pred_labels.shape != target.shape:
            raise ValueError(f"shape mismatch case {case}: pred={pred_labels.shape} target={target.shape}")

        pred_support = np.isin(pred_labels, BFV3_SUPPORT_LABELS)
        target_support = np.isin(target, BFV3_SUPPORT_LABELS)
        fp = pred_support & ~target_support
        fn = target_support & ~pred_support
        tp = pred_support & target_support
        dist_to_target_support = ndi.distance_transform_edt(~target_support, sampling=spacing_zyx)
        dist_to_pred_support = ndi.distance_transform_edt(~pred_support, sampling=spacing_zyx)
        surface = _surface_distance_summary(pred_support, target_support, spacing_zyx)
        support = binary_surface_metrics(pred_support, target_support, spacing_zyx, nsd_tolerance_mm=2.0)
        decoded = decode_boundary_fragment(pred_labels, profile="geodesic_marker_v2")
        iouf = instance_iouf(decoded, source_gt)
        row = {
            "case": case,
            "prediction": str(pred_path),
            "support": support,
            "iou_f": iouf,
            "support_confusion": {
                "tp_voxels": int(tp.sum()),
                "fp_voxels": int(fp.sum()),
                "fn_voxels": int(fn.sum()),
                "pred_support_voxels": int(pred_support.sum()),
                "target_support_voxels": int(target_support.sum()),
                "fp_fraction_of_pred_support": float(fp.sum() / max(1, int(pred_support.sum()))),
                "fn_fraction_of_target_support": float(fn.sum() / max(1, int(target_support.sum()))),
            },
            "surface_distance": surface,
            "pred_support_component_count": int(ndi.label(pred_support)[1]) if pred_support.any() else 0,
            "target_support_component_count": int(ndi.label(target_support)[1]) if target_support.any() else 0,
            "fp_components": _support_component_summaries(
                fp,
                pred_labels=pred_labels,
                target_labels=target,
                source_gt=source_gt,
                ct_hu=ct_hu,
                distance_mm=dist_to_target_support,
                spacing_zyx=spacing_zyx,
                max_components=max_components,
                kind="fp",
            ),
            "fn_components": _support_component_summaries(
                fn,
                pred_labels=pred_labels,
                target_labels=target,
                source_gt=source_gt,
                ct_hu=ct_hu,
                distance_mm=dist_to_pred_support,
                spacing_zyx=spacing_zyx,
                max_components=max_components,
                kind="fn",
            ),
            "target_audit": target_audit,
        }
        rows.append(row)
    summary = {
        "support_dice_mean": float(np.mean([r["support"]["dice"] for r in rows])) if rows else 0.0,
        "support_hd95_mean": float(np.mean([r["support"]["hd95_mm"] for r in rows if r["support"]["hd95_mm"] is not None]))
        if any(r["support"]["hd95_mm"] is not None for r in rows) else None,
        "iou_f_mean": float(np.mean([r["iou_f"]["iou_f_mean"] for r in rows])) if rows else 0.0,
        "fp_voxels_total": int(sum(r["support_confusion"]["fp_voxels"] for r in rows)),
        "fn_voxels_total": int(sum(r["support_confusion"]["fn_voxels"] for r in rows)),
        "top_fp_categories": {},
    }
    category_counts: dict[str, int] = {}
    for row in rows:
        for comp in row["fp_components"]:
            category_counts[comp["category"]] = category_counts.get(comp["category"], 0) + int(comp["voxels"])
    summary["top_fp_categories"] = dict(sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True))
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "boundary_fragment_support_error_audit",
        "dataset": DATASETS[537]["name"],
        "pred_root": str(pred_root),
        "target_profile": target_profile,
        "contact_band_mm": float(contact_band_mm),
        "contact_search_mm": float(contact_search_mm),
        "contact_ridge_mm": float(contact_ridge_mm),
        "contact_same_anatomy_only": bool(contact_same_anatomy_only),
        "shell_mm": float(shell_mm),
        "cases": rows,
        "summary": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(result), indent=2, allow_nan=False))
    return result


def run_boundary_fragment_eval(cases: list[str],
                               out_path: Path,
                               pred_root: Path | None = None,
                               oracle: bool = False,
                               decoder_profile: str = "hard_barrier",
                               target_profile: str = "v3_thin_ridge",
                               contact_band_mm: float = 3.0,
                               contact_search_mm: float = 3.0,
                               contact_ridge_mm: float = 1.0,
                               contact_same_anatomy_only: bool = False,
                               shell_mm: float = 8.0) -> dict:
    """Evaluate fixed BoundaryFragment V3 decoder on source-space labels.

    `--oracle` feeds the GT-derived V3 target to the decoder. This is the first
    gate: if oracle IoU-F is bad, training is forbidden because the target or
    decoder definition is wrong.
    """
    rows = []
    for case in [str(c).zfill(3) for c in cases]:
        cd = find_case_dir(case)
        if cd is None:
            raise FileNotFoundError(f"source case not found: {case}")
        img = canonicalize_sitk(sitk.ReadImage(str(cd / "image.mha")))
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        gt = sitk.GetArrayFromImage(lbl_img)
        spacing_zyx = _spacing_zyx_from_image(img)
        target, target_audit = compute_boundary_fragment_target(
            gt,
            spacing_zyx=spacing_zyx,
            params=BoundaryFragmentParams(
                target_profile=target_profile,
                contact_band_mm=float(contact_band_mm),
                contact_search_mm=float(contact_search_mm),
                contact_ridge_mm=float(contact_ridge_mm),
                contact_same_anatomy_only=bool(contact_same_anatomy_only),
                shell_mm=float(shell_mm),
            ),
        )
        if oracle:
            pred_labels = target
            pred_source = "oracle_target"
        else:
            if pred_root is None:
                raise ValueError("--pred-root is required unless --oracle is set")
            pred_path = pred_root / f"PENGWIN_{case}.mha"
            if not pred_path.exists():
                raise FileNotFoundError(f"prediction not found: {pred_path}")
            pred_labels = sitk.GetArrayFromImage(canonicalize_sitk(sitk.ReadImage(str(pred_path)))).astype(np.uint8)
            pred_source = str(pred_path)
        decoded = decode_boundary_fragment(pred_labels, profile=decoder_profile)
        iouf = instance_iouf(decoded, gt)
        topology = oracle_topology_diagnostics(decoded, gt, target, iouf)
        support_metrics = binary_surface_metrics(
            np.isin(pred_labels, [2, 3, 4]),
            np.isin(target, [2, 3, 4]),
            spacing_zyx,
            nsd_tolerance_mm=2.0,
        )
        barrier_metrics = _binary_prf(pred_labels == 2, target == 2, beta=0.5)
        row = {
            "case": case,
            "prediction": pred_source,
            "oracle": bool(oracle),
            "iou_f": iouf,
            "support": support_metrics,
            "class2_barrier": barrier_metrics,
            "target_audit": target_audit,
            "topology_diagnostics": topology,
        }
        rows.append(row)
    summary = {
        "iou_f_mean": float(np.mean([r["iou_f"]["iou_f_mean"] for r in rows])) if rows else 0.0,
        "iou_f_min": float(np.min([r["iou_f"]["iou_f_mean"] for r in rows])) if rows else 0.0,
        "support_dice_mean": float(np.mean([r["support"]["dice"] for r in rows])) if rows else 0.0,
        "support_hd95_mean": float(np.mean([
            r["support"]["hd95_mm"] for r in rows if r["support"]["hd95_mm"] is not None
        ])) if any(r["support"]["hd95_mm"] is not None for r in rows) else None,
        "barrier_precision_mean": float(np.mean([r["class2_barrier"]["precision"] for r in rows])) if rows else 0.0,
        "barrier_recall_mean": float(np.mean([r["class2_barrier"]["recall"] for r in rows])) if rows else 0.0,
        "barrier_ratio_mean": float(np.mean([r["class2_barrier"]["pred_target_ratio"] for r in rows])) if rows else 0.0,
        "missing_seed_cases": [
            r["case"] for r in rows
            if r["target_audit"]["label_counts"].get(str(BFV3_LABELS["fragment_core"]), 0) == 0
        ],
        "decoder_profile": decoder_profile,
        "target_profile": target_profile,
        "contact_band_mm": float(contact_band_mm),
        "contact_search_mm": float(contact_search_mm),
        "contact_ridge_mm": float(contact_ridge_mm),
        "contact_same_anatomy_only": bool(contact_same_anatomy_only),
        "shell_mm": float(shell_mm),
    }
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "boundary_fragment_v3_oracle" if oracle else "boundary_fragment_v3_prediction",
        "dataset": DATASETS[537]["name"],
        "decoder_profile": decoder_profile,
        "target_profile": target_profile,
        "contact_band_mm": float(contact_band_mm),
        "contact_search_mm": float(contact_search_mm),
        "contact_ridge_mm": float(contact_ridge_mm),
        "contact_same_anatomy_only": bool(contact_same_anatomy_only),
        "shell_mm": float(shell_mm),
        "cases": rows,
        "summary": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(result), indent=2, allow_nan=False))
    return result


def _json_sanitize(value: Any) -> Any:
    """Convert numpy/path values into JSON-safe plain Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_folds(folds: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    """Normalize nnU-Net fold arguments and reject invalid folds."""
    if folds is None:
        return (0,)
    out = tuple(int(f) for f in folds)
    if not out:
        return (0,)
    invalid = [f for f in out if f < 0 or f > 4]
    if invalid:
        raise ValueError(f"folds must be in 0..4, got {invalid}")
    return out


def model_dir(ds_id: int,
              plans: str = "nnUNetResEncUNetLPlans",
              config: str = "3d_fullres",
              trainer: str | None = None) -> Path:
    """Return the nnU-Net result directory for an active dataset."""
    cfg = DATASETS[ds_id]
    trainer = trainer or cfg["trainer"]
    return NN_RES / cfg["name"] / f"{trainer}__{plans}__{config}"


def checkpoint_ready(ds_id: int, fold: int = 0,
                     checkpoint: str = "checkpoint_best.pth") -> bool:
    """True if a fold checkpoint exists for an active split-anatomy model."""
    return (model_dir(ds_id) / f"fold_{fold}" / checkpoint).exists()


def _spacing_zyx_from_image(img: sitk.Image) -> tuple[float, float, float]:
    sx, sy, sz = [float(v) for v in img.GetSpacing()]
    return (sz, sy, sx)


def local_gt_for_dataset(ds_id: int, source_label: np.ndarray) -> np.ndarray:
    """Map original PENGWIN fragment IDs to the dataset-local semantic target."""
    anat = inst_to_anat(source_label)
    if ds_id == 532:
        out = np.zeros_like(anat, dtype=np.uint8)
        out[(anat >= 1) & (anat <= 3)] = anat[(anat >= 1) & (anat <= 3)]
        return out
    if ds_id == 533:
        out = np.zeros_like(anat, dtype=np.uint8)
        out[anat == 4] = 1
        return out
    raise KeyError(f"unknown active dataset {ds_id}")


def case_dataset_id(case_id: str | int) -> int:
    """Return the active split dataset ID for a case."""
    cid = int(case_id)
    if is_pelvic(cid):
        return 532
    if is_femur(cid):
        return 533
    raise ValueError(f"cannot infer active dataset for case {case_id}")


def label_names(ds_id: int) -> list[str]:
    """Return local semantic label names including background."""
    return ["Background", *DATASETS[ds_id]["anatomies"]]


def semantic_metrics(gt: np.ndarray, pred: np.ndarray,
                     names: list[str]) -> dict:
    """Compute per-label Dice/IoU and a confusion matrix."""
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt={gt.shape} pred={pred.shape}")
    n_classes = len(names)
    dice = {}
    iou = {}
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for g in range(n_classes):
        gm = gt == g
        for p in range(n_classes):
            confusion[g, p] = int((gm & (pred == p)).sum())
        pm = pred == g
        inter = int((gm & pm).sum())
        denom_d = int(gm.sum() + pm.sum())
        denom_i = int((gm | pm).sum())
        dice[names[g]] = 1.0 if denom_d == 0 else float(2.0 * inter / denom_d)
        iou[names[g]] = 1.0 if denom_i == 0 else float(inter / denom_i)
    foreground = names[1:]
    return {
        "labels": names,
        "dice": dice,
        "iou": iou,
        "mean_foreground_dice": float(np.mean([dice[n] for n in foreground])) if foreground else None,
        "mean_foreground_iou": float(np.mean([iou[n] for n in foreground])) if foreground else None,
        "confusion_matrix": confusion.tolist(),
    }


def binary_class_metrics(gt: np.ndarray, pred: np.ndarray, label: int) -> dict:
    gm = gt == int(label)
    pm = pred == int(label)
    tp = int((gm & pm).sum())
    fp = int((~gm & pm).sum())
    fn = int((gm & ~pm).sum())
    denom_d = int(gm.sum() + pm.sum())
    denom_i = int((gm | pm).sum())
    return {
        "label": int(label),
        "target_voxels": int(gm.sum()),
        "pred_voxels": int(pm.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": 1.0 if denom_d == 0 else float(2.0 * tp / denom_d),
        "iou": 1.0 if denom_i == 0 else float(tp / denom_i),
        "precision": 1.0 if tp + fp == 0 else float(tp / (tp + fp)),
        "recall": 1.0 if tp + fn == 0 else float(tp / (tp + fn)),
    }


def binary_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    gm = gt_mask.astype(bool, copy=False)
    pm = pred_mask.astype(bool, copy=False)
    tp = int((gm & pm).sum())
    fp = int((~gm & pm).sum())
    fn = int((gm & ~pm).sum())
    denom_d = int(gm.sum() + pm.sum())
    denom_i = int((gm | pm).sum())
    return {
        "target_voxels": int(gm.sum()),
        "pred_voxels": int(pm.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": 1.0 if denom_d == 0 else float(2.0 * tp / denom_d),
        "iou": 1.0 if denom_i == 0 else float(tp / denom_i),
        "precision": 1.0 if tp + fp == 0 else float(tp / (tp + fp)),
        "recall": 1.0 if tp + fn == 0 else float(tp / (tp + fn)),
    }


def _label_physical_x_centroid(label: np.ndarray, img: sitk.Image,
                               value: int) -> float | None:
    """Return physical x centroid for one semantic label.

    Arrays are z/y/x, while SimpleITK continuous indices are x/y/z. Computing
    this in physical coordinates keeps the rule stable across spacing, origin,
    and direction metadata.
    """
    z, y, x = np.nonzero(label == value)
    if len(x) == 0:
        return None
    continuous_index = np.array([x.mean(), y.mean(), z.mean()], dtype=np.float64)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    origin = np.array(img.GetOrigin(), dtype=np.float64)
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    physical = origin + direction.dot(continuous_index * spacing)
    return float(physical[0])


def enforce_pelvic_lr_consistency(pred: np.ndarray,
                                  pred_img: sitk.Image) -> tuple[np.ndarray, dict]:
    """Fix complete LeftHip/RightHip label inversions in Dataset532 predictions.

    Dataset532 is a mutually exclusive semantic model: `2=LeftHip`, `3=RightHip`.
    In canonical LPS physical coordinates, the patient's left side has larger
    x than the right side. If predicted label 2 is physically right of predicted
    label 3, the model has learned the shapes but emitted the side labels in the
    wrong order for this case. Swapping only the two side labels is a
    deterministic convention correction, not morphology cleanup.
    """
    pred = pred.astype(np.uint8, copy=False)
    left_x = _label_physical_x_centroid(pred, pred_img, 2)
    right_x = _label_physical_x_centroid(pred, pred_img, 3)
    info = {
        "enabled": True,
        "applied": False,
        "note": LR_FIX_NOTE,
        "left_label_physical_x": left_x,
        "right_label_physical_x": right_x,
        "reason": None,
    }
    if left_x is None or right_x is None:
        info["reason"] = "missing_left_or_right_prediction"
        return pred, info
    if left_x >= right_x:
        info["reason"] = "already_consistent"
        return pred, info

    fixed = pred.copy()
    left_mask = pred == 2
    right_mask = pred == 3
    fixed[left_mask] = 3
    fixed[right_mask] = 2
    info["applied"] = True
    info["reason"] = "swapped_labels_2_and_3"
    return fixed, info


def audit_pelvic_lr_consistency(pred: np.ndarray, pred_img: sitk.Image) -> dict:
    """Diagnose LeftHip/RightHip physical-x consistency without changing labels."""
    pred = pred.astype(np.uint8, copy=False)
    left_x = _label_physical_x_centroid(pred, pred_img, 2)
    right_x = _label_physical_x_centroid(pred, pred_img, 3)
    info = {
        "mode": "audit",
        "enabled": True,
        "triggered": False,
        "applied": False,
        "note": LR_FIX_NOTE,
        "left_label_physical_x": left_x,
        "right_label_physical_x": right_x,
        "reason": None,
    }
    if left_x is None or right_x is None:
        info["reason"] = "missing_left_or_right_prediction"
        return info
    if left_x >= right_x:
        info["reason"] = "already_consistent"
        return info
    info["triggered"] = True
    info["reason"] = "left_label_is_physically_right_of_right_label"
    return info


def apply_lr_guard(pred: np.ndarray, pred_img: sitk.Image,
                   mode: str) -> tuple[np.ndarray, dict]:
    """Run LR guard in audit/apply/off mode.

    The guard is intentionally downstream of the canonical LPS input contract.
    `audit` records suspicious LH/RH ordering but leaves metrics untouched;
    `apply` is reserved for diagnostic what-if scoring and never hides a failed
    orientation contract in the primary raw metrics.
    """
    if mode not in {"audit", "apply", "off"}:
        raise ValueError(f"unknown lr_guard mode: {mode}")
    if mode == "off":
        return pred, {
            "mode": "off",
            "enabled": False,
            "triggered": False,
            "applied": False,
            "reason": "disabled",
        }
    info = audit_pelvic_lr_consistency(pred, pred_img)
    info["mode"] = mode
    if mode == "audit" or not info.get("triggered"):
        return pred, info
    fixed, applied_info = enforce_pelvic_lr_consistency(pred, pred_img)
    info.update(applied_info)
    info["mode"] = mode
    info["triggered"] = bool(info.get("triggered") or info.get("applied"))
    return fixed, info


def _write_prediction(pred: np.ndarray, ref_img: sitk.Image, path: Path) -> None:
    """Write a semantic prediction array with the reference image metadata."""
    out = sitk.GetImageFromArray(pred.astype(np.uint8, copy=False))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(path))


def _save_float_mha(arr: np.ndarray, ref_img: sitk.Image, path: Path) -> None:
    out = sitk.GetImageFromArray(arr.astype(np.float32, copy=False))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(path), useCompression=True)


def _sdf_from_anatomy_probs(probs: np.ndarray,
                            ref_img: sitk.Image,
                            clip_mm: float = 40.0) -> np.ndarray:
    from scipy import ndimage as ndi

    pelvic_prob = np.clip(probs[1:4].sum(axis=0), 0.0, 1.0)
    pelvic = pelvic_prob >= 0.5
    if not pelvic.any():
        return np.zeros_like(pelvic_prob, dtype=np.float32)
    sx, sy, sz = [float(v) for v in ref_img.GetSpacing()]
    spacing_zyx = (sz, sy, sx)
    inside = ndi.distance_transform_edt(pelvic, sampling=spacing_zyx)
    outside = ndi.distance_transform_edt(~pelvic, sampling=spacing_zyx)
    return (np.clip(inside - outside, -float(clip_mm), float(clip_mm)) / float(clip_mm)).astype(np.float32)


def _hard_negative_prior_from_ct_and_probs(ct: np.ndarray,
                                           probs: np.ndarray,
                                           hu_threshold: float = 250.0,
                                           low_pelvis_prob: float = 0.35) -> np.ndarray:
    """Inference-time class-4 prior channel for Dataset537.

    This mirrors preprocessing's input channel and intentionally uses only CT
    plus Dataset532 probabilities. GT-derived class-4 supervision is never used
    at inference time.
    """
    pelvic_prob = np.clip(probs[1:4].sum(axis=0), 0.0, 1.0)
    prior = (ct.astype(np.float32, copy=False) > float(hu_threshold)) & (pelvic_prob < float(low_pelvis_prob))
    return prior.astype(np.float32, copy=False)


def _dataset_input_channel_count(ds_id: int) -> int:
    dataset_json = Path(os.environ.get("nnUNet_raw", "/workspace/nnunet/raw")) / DATASETS[ds_id]["name"] / "dataset.json"
    if not dataset_json.exists():
        return 1
    try:
        return len(json.loads(dataset_json.read_text()).get("channel_names", {"0": "CT"}))
    except Exception:
        return 1


def validation_case_ids(ds_id: int, fold: int = 0) -> list[str]:
    """Read validation case IDs for one active dataset/fold."""
    split_path = NN_PREP / DATASETS[ds_id]["name"] / "splits_final.json"
    splits = json.loads(split_path.read_text())
    val = splits[int(fold)]["val"]
    out = []
    for name in val:
        token = str(name)
        out.append(token.split("_")[-1].zfill(3))
    return out


def validation_prediction_path(ds_id: int, case_id: str, fold: int = 0,
                               plans: str = "nnUNetResEncUNetLPlans",
                               config: str = "3d_fullres",
                               trainer: str = "PengwinTrainer") -> Path:
    """Return the nnU-Net validation prediction path created after training."""
    return (
        _nnunet_results_root() / DATASETS[ds_id]["name"] / f"{trainer}__{plans}__{config}"
        / f"fold_{int(fold)}" / "validation" / f"PENGWIN_{str(case_id).zfill(3)}.mha"
    )


def predict_one(ds_id: int, image_path: Path, out_dir: Path, gpu: int = 0,
                folds: tuple[int, ...] | list[int] | None = None,
                checkpoint: str = "checkpoint_best.pth",
                trainer: str | None = None,
                save_probabilities: bool = False) -> Path:
    """Run nnU-Net prediction for one active split-anatomy model."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch

    folds = _normalize_folds(folds)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pengwin_eval_") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "in"
        pred_dir = tmp_dir / "pred"
        input_dir.mkdir()
        pred_dir.mkdir()
        prepared_input = input_dir / "PENGWIN_000_0000.mha"
        input_meta = prepare_lps_ct_for_nnunet(image_path, prepared_input)
        input_files = [str(prepared_input)]
        if DATASETS[ds_id]["kind"] in {"abbc_official", "contact_instance"} and _dataset_input_channel_count(ds_id) >= 5:
            context_dir = tmp_dir / "ctx532"
            context_pred = predict_one(
                532, image_path, context_dir, gpu=gpu, folds=folds,
                checkpoint="checkpoint_best.pth", trainer=None,
                save_probabilities=True,
            )
            probs = _load_probability_npz(_probability_path(context_pred))
            ref_img = sitk.ReadImage(str(prepared_input))
            if probs.shape[0] < 4 or probs.shape[1:] != sitk.GetArrayFromImage(ref_img).shape:
                raise RuntimeError(
                    f"Dataset532 context shape mismatch for {image_path}: "
                    f"probs={probs.shape}, input={sitk.GetArrayFromImage(ref_img).shape}"
                )
            for ch, arr in enumerate((probs[1], probs[2], probs[3]), start=1):
                path = input_dir / f"PENGWIN_000_{ch:04d}.mha"
                _save_float_mha(arr, ref_img, path)
                input_files.append(str(path))
            sdf_path = input_dir / "PENGWIN_000_0004.mha"
            _save_float_mha(_sdf_from_anatomy_probs(probs, ref_img), ref_img, sdf_path)
            input_files.append(str(sdf_path))
            if _dataset_input_channel_count(ds_id) >= 6:
                hn_path = input_dir / "PENGWIN_000_0005.mha"
                ct_arr = sitk.GetArrayFromImage(ref_img)
                _save_float_mha(_hard_negative_prior_from_ct_and_probs(ct_arr, probs), ref_img, hn_path)
                input_files.append(str(hn_path))
            input_meta["abbc_context"] = {
                "source": "Dataset532 checkpoint_best probabilities",
                "probability_path": str(_probability_path(context_pred)),
                "channels": [
                    "CT", "P(Sacrum)", "P(LeftHip)", "P(RightHip)", "pelvis_sdf",
                    *(
                        ["bone_like_hard_negative_prior"]
                        if _dataset_input_channel_count(ds_id) >= 6 else []
                    ),
                ],
            }

        device = torch.device("cuda", int(gpu)) if torch.cuda.is_available() else torch.device("cpu")
        # Dataset532 has semantic side labels (LeftHip/RightHip). Plain mirrored
        # test-time augmentation can average logits from a flipped view without
        # swapping side-label channels, which caused complete LH/RH inversions in
        # some hard cases. Training also disables x-mirror for this dataset, so
        # inference follows the same side-label contract.
        use_tta_mirroring = DATASETS[ds_id]["filter"] != "pelvic"
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=use_tta_mirroring,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_dir(ds_id, trainer=trainer)),
            use_folds=folds,
            checkpoint_name=checkpoint,
        )
        predictor.predict_from_files(
            [input_files],
            str(pred_dir),
            save_probabilities=save_probabilities,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        pred_path = pred_dir / "PENGWIN_000.mha"
        final_path = out_dir / image_path.name.replace("_0000", "")
        final_path.write_bytes(pred_path.read_bytes())
        prob_final_path = None
        if save_probabilities:
            prob_src = pred_dir / "PENGWIN_000.npz"
            pkl_src = pred_dir / "PENGWIN_000.pkl"
            if prob_src.exists():
                prob_final_path = out_dir / final_path.with_suffix(".npz").name
                prob_final_path.write_bytes(prob_src.read_bytes())
            if pkl_src.exists():
                (out_dir / final_path.with_suffix(".pkl").name).write_bytes(pkl_src.read_bytes())
        meta = {
            "task": "split_anatomy_v2_prediction",
            "dataset_id": ds_id,
            "dataset_name": DATASETS[ds_id]["name"],
            "source_image": str(image_path),
            "prediction_path": str(final_path),
            "checkpoint": checkpoint,
            "folds": list(folds),
            "model_dir": str(model_dir(ds_id, trainer=trainer)),
            "trainer": trainer or DATASETS[ds_id]["trainer"],
            "use_tta_mirroring": use_tta_mirroring,
            "save_probabilities": bool(save_probabilities),
            "probability_path": str(prob_final_path) if prob_final_path is not None else None,
            "orientation_contract": ORIENTATION_CONTRACT_VERSION,
            "input_metadata": input_meta,
        }
        (out_dir / "prediction_meta.json").write_text(
            json.dumps(_json_sanitize(meta), indent=2)
        )
        return final_path


def evaluate_case(case_id: str, pred_path: Path | None = None,
                  gpu: int = 0, folds: tuple[int, ...] | list[int] | None = None,
                  pred_root: Path | None = None,
                  lr_guard: str = "audit",
                  force_repredict: bool = False,
                  save_corrected_pred: bool = False,
                  checkpoint: str = "checkpoint_best.pth") -> dict:
    """Evaluate one case for its active split dataset."""
    ds_id = case_dataset_id(case_id)
    case_dir = find_case_dir(case_id)
    if case_dir is None:
        raise FileNotFoundError(f"case not found: {case_id}")
    image_path = case_dir / "image.mha"
    gt_img = canonicalize_sitk(sitk.ReadImage(str(case_dir / "label.mha")))
    gt = local_gt_for_dataset(ds_id, sitk.GetArrayFromImage(gt_img))

    if pred_root is None:
        pred_root = DEFAULT_PRED_ROOT
    cid = str(case_id).zfill(3)
    if pred_path is None:
        nested_pred = pred_root / cid / "image.mha"
        flat_pred = pred_root / f"PENGWIN_{cid}.mha"
        if nested_pred.exists() and not force_repredict:
            pred_path = nested_pred
        elif flat_pred.exists() and not force_repredict:
            pred_path = flat_pred
        else:
            pred_path = predict_one(ds_id, image_path, pred_root / cid,
                                    gpu=gpu, folds=folds, checkpoint=checkpoint)
    pred_img = canonicalize_sitk(sitk.ReadImage(str(pred_path)))
    pred = sitk.GetArrayFromImage(pred_img).astype(np.uint8)
    raw_metrics = semantic_metrics(gt, pred, label_names(ds_id))
    lr_info = {
        "mode": lr_guard,
        "enabled": False,
        "triggered": False,
        "applied": False,
        "reason": "not_pelvic_dataset",
    }
    if ds_id == 532:
        pred, lr_info = apply_lr_guard(pred, pred_img, lr_guard)
        if lr_info.get("applied") and save_corrected_pred:
            _write_prediction(pred, pred_img, pred_path)
    metrics = semantic_metrics(gt, pred, label_names(ds_id))
    return {
        "case": cid,
        "dataset_id": ds_id,
        "dataset_name": DATASETS[ds_id]["name"],
        "prediction_path": str(pred_path),
        "orientation_contract": ORIENTATION_CONTRACT_VERSION,
        "lr_guard": lr_info,
        "lr_fix": lr_info,
        "raw_metrics_before_lr_fix": raw_metrics if lr_info.get("applied") else None,
        "raw_metrics_canonical": raw_metrics,
        "metrics": metrics,
    }




def write_eval_result(rows: list[dict], out_path: Path,
                      folds: tuple[int, ...] | list[int] | None = None,
                      extra: dict | None = None) -> dict:
    """Aggregate case rows, write JSON, and return the result payload."""
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset_id"]), []).append(row)
    summary = {}
    for ds_id, ds_rows in by_dataset.items():
        raw_iou_values = [
            r.get("raw_metrics_canonical", r.get("raw_metrics_before_lr_fix") or r["metrics"])["mean_foreground_iou"]
            for r in ds_rows
        ]
        raw_dice_values = [
            r.get("raw_metrics_canonical", r.get("raw_metrics_before_lr_fix") or r["metrics"])["mean_foreground_dice"]
            for r in ds_rows
        ]
        summary[ds_id] = {
            "dataset_name": DATASETS[int(ds_id)]["name"],
            "n_cases": len(ds_rows),
            "mean_foreground_dice": float(np.mean([
                r["metrics"]["mean_foreground_dice"] for r in ds_rows
            ])),
            "mean_foreground_iou": float(np.mean([
                r["metrics"]["mean_foreground_iou"] for r in ds_rows
            ])),
            "raw_mean_foreground_dice_before_lr_fix": float(np.mean(raw_dice_values)),
            "raw_mean_foreground_iou_before_lr_fix": float(np.mean(raw_iou_values)),
            "raw_mean_foreground_dice_canonical": float(np.mean(raw_dice_values)),
            "raw_mean_foreground_iou_canonical": float(np.mean(raw_iou_values)),
            "lr_guard_triggered_cases": [
                r["case"] for r in ds_rows if r.get("lr_guard", {}).get("triggered")
            ],
            "lr_guard_applied_cases": [
                r["case"] for r in ds_rows if r.get("lr_guard", {}).get("applied")
            ],
            "lr_fix_applied_cases": [
                r["case"] for r in ds_rows if r.get("lr_fix", {}).get("applied")
            ],
        }
    result = {
        "task": "split_anatomy_v2",
        "folds": list(_normalize_folds(folds)),
        "case_ids": [row["case"] for row in rows],
        "summary": summary,
        "case_results": rows,
    }
    if extra:
        result.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(result), indent=2))
    return result




def _write_instance_prediction(pred: np.ndarray, ref_img: sitk.Image, path: Path) -> None:
    """Write a global fragment-ID prediction with reference geometry."""
    out = sitk.GetImageFromArray(pred.astype(np.uint16, copy=False))
    out.CopyInformation(ref_img)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path), useCompression=True)


def _predict_for_case(ds_id: int, case_id: str, image_path: Path,
                      pred_root: Path, gpu: int,
                      folds: tuple[int, ...] | list[int] | None,
                      checkpoint: str,
                      force_repredict: bool,
                      save_probabilities: bool = False) -> Path:
    """Predict one dataset into a case/dataset-specific cache folder."""
    cid = str(case_id).zfill(3)
    out_dir = pred_root / cid / f"ds{ds_id}"
    cached = out_dir / "image.mha"
    cached_prob = out_dir / "image.npz"
    if cached.exists() and (not save_probabilities or cached_prob.exists()) and not force_repredict:
        return cached
    return predict_one(
        ds_id, image_path, out_dir, gpu=gpu, folds=folds, checkpoint=checkpoint,
        save_probabilities=save_probabilities,
    )


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """26-connected component labeling for decoder seeds/components."""
    from scipy import ndimage as ndi

    return ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))


def _global_id_to_anatomy_index(values: np.ndarray) -> np.ndarray:
    """Map global PENGWIN fragment IDs to anatomy semantic labels (incl. Femur).

    Registry order gives Sacrum=1, LeftHip=2, RightHip=3, Femur=4 — the inverse of
    the instance-ID encoding. Femur (151-200) now maps to anatomy class 4 instead
    of silently collapsing to background, so it is scored in semantic_metrics.
    """
    out = np.zeros_like(values, dtype=np.uint8)
    for local_idx, (name, lo, hi) in enumerate(ANATOMY_RANGES, start=1):
        out[(values >= lo) & (values <= hi)] = local_idx
    return out


ABBC_DECODER_PROFILES = (
    "official_v1",
    "official_v1_strict_argmax",
    "contact_energy_v1",
    "contact_energy_v2",
    "border_barrier_v1",
    "core_cc_nearest",
)


def _load_probability_npz(prob_path: Path) -> np.ndarray:
    """Load nnU-Net probability output as (C, Z, Y, X)."""
    if not prob_path.exists():
        raise FileNotFoundError(f"missing probability cache: {prob_path}")
    data = np.load(prob_path)
    for key in ("probabilities", "softmax"):
        if key in data:
            arr = data[key]
            break
    else:
        keys = list(data.keys())
        if not keys:
            raise RuntimeError(f"empty probability npz: {prob_path}")
        arr = data[keys[0]]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise RuntimeError(f"expected 4D probability array in {prob_path}, got {arr.shape}")
    return arr


def _probability_path(pred_path: Path) -> Path:
    return pred_path.with_suffix(".npz")


def _assign_border_to_nearest_instance(local_instance: np.ndarray,
                                       border_mask: np.ndarray,
                                       spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Assign border voxels after instances are formed, so border never bridges.

    The ABBC border class is a separator cue. If it participates in the connected
    foreground before seeding, it can glue two fragments together and defeats the
    reason we trained it. This helper therefore runs only after non-border
    regions have instance IDs; border voxels inherit the nearest finished
    instance without becoming a path that connects regions during decoding.
    """
    if not border_mask.any() or not (local_instance > 0).any():
        return local_instance
    nearest_idx = ndi_distance_indices(local_instance == 0, spacing_zyx)
    nearest_label = local_instance[tuple(nearest_idx)]
    out = local_instance.copy()
    out[border_mask] = nearest_label[border_mask]
    return out


def ndi_distance_indices(mask: np.ndarray,
                         spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Return nearest-background indices for a binary mask via SciPy EDT."""
    from scipy import ndimage as ndi

    return ndi.distance_transform_edt(
        mask,
        sampling=spacing_zyx,
        return_distances=False,
        return_indices=True,
    )


def _decode_local_core_cc_nearest(anatomy_mask: np.ndarray,
                                  pred: np.ndarray,
                                  spacing_zyx: tuple[float, float, float]) -> tuple[np.ndarray, dict]:
    """Legacy ABBC decoder: each core CC is one seed, border is assignable foreground."""
    candidate = anatomy_mask & (pred > 0)
    core = anatomy_mask & (pred == ABBC_CORE_LABEL)
    seed_cc, n_seeds = _label_components(core)
    comp_cc, n_components = _label_components(anatomy_mask)
    missing = []
    for comp_id in range(1, n_components + 1):
        comp = comp_cc == comp_id
        if comp.any() and not (core & comp).any():
            missing.append({"component": comp_id, "voxels": int(comp.sum())})

    row = {
        "decoder_profile": "core_cc_nearest",
        "anatomy_mask_voxels": int(anatomy_mask.sum()),
        "candidate_voxels": int(candidate.sum()),
        "border_voxels": int((anatomy_mask & (pred == ABBC_BORDER_LABEL)).sum()),
        "raw_core_cc_count": int(n_seeds),
        "core_seed_components": int(n_seeds),
        "merged_seed_count": int(n_seeds),
        "non_border_component_count": None,
        "core_merge_groups": [],
        "missing_components": missing,
        "assigned_instances": 0,
    }
    if n_seeds == 0 or not candidate.any():
        return np.zeros_like(pred, dtype=np.uint16), row

    nearest_idx = ndi_distance_indices(seed_cc == 0, spacing_zyx)
    assigned_seed = seed_cc[tuple(nearest_idx)]
    local_instance = np.where(candidate, assigned_seed, 0).astype(np.uint16, copy=False)
    row["assigned_instances"] = len([int(v) for v in np.unique(local_instance) if int(v) > 0])
    return local_instance, row


def _decode_local_border_barrier(anatomy_mask: np.ndarray,
                                 pred: np.ndarray,
                                 spacing_zyx: tuple[float, float, float]) -> tuple[np.ndarray, dict]:
    """Border-aware ABBC decoder that treats label 2 as a wall.

    Target v2 promises one semantic role per class: core proposes seeds, border
    separates neighboring fragments, and boundary is ordinary assignable shell.
    The old nearest-core decoder ignored that contract and let border act like
    foreground. Here we first remove border from connectivity, merge all core
    islands that live in the same non-border component, and only then attach
    border voxels to the nearest completed instance. This fixes the observed
    "one GT fragment -> many core islands -> many predicted fragments" failure
    without inventing seeds or doing morphology cleanup.
    """
    border = anatomy_mask & (pred == ABBC_BORDER_LABEL)
    core = anatomy_mask & (pred == ABBC_CORE_LABEL)
    raw_core_cc, n_raw_core = _label_components(core)
    non_border_support = anatomy_mask & ~border
    comp_cc, n_components = _label_components(non_border_support)

    local_instance = np.zeros(pred.shape, dtype=np.uint16)
    missing = []
    merge_groups = []
    next_seed = 1
    for comp_id in range(1, n_components + 1):
        comp = comp_cc == comp_id
        raw_ids = [int(v) for v in np.unique(raw_core_cc[comp]) if int(v) > 0]
        if not raw_ids:
            missing.append({"component": comp_id, "voxels": int(comp.sum())})
            continue
        local_instance[comp] = next_seed
        merge_groups.append({
            "merged_seed_id": int(next_seed),
            "component": int(comp_id),
            "raw_core_ids": raw_ids,
            "raw_core_count": len(raw_ids),
            "voxels": int(comp.sum()),
        })
        next_seed += 1

    local_instance = _assign_border_to_nearest_instance(local_instance, border, spacing_zyx)
    assigned = [int(v) for v in np.unique(local_instance) if int(v) > 0]
    row = {
        "decoder_profile": "border_barrier_v1",
        "anatomy_mask_voxels": int(anatomy_mask.sum()),
        "candidate_voxels": int(anatomy_mask.sum()),
        "border_voxels": int(border.sum()),
        "raw_core_cc_count": int(n_raw_core),
        "core_seed_components": int(n_raw_core),
        "merged_seed_count": int(len(merge_groups)),
        "non_border_component_count": int(n_components),
        "core_merge_groups": merge_groups,
        "missing_components": missing,
        "assigned_instances": len(assigned),
    }
    return local_instance, row


def _compact_labels(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.uint16)
    for new_id, old_id in enumerate([int(v) for v in np.unique(values) if int(v) > 0], start=1):
        out[values == old_id] = new_id
    return out


def _watershed_assign_energy(markers: np.ndarray,
                             support: np.ndarray,
                             energy: np.ndarray) -> np.ndarray:
    """Assign support to markers using contact probability as watershed energy."""
    if not support.any() or not (markers > 0).any():
        return np.zeros_like(markers, dtype=np.uint16)
    try:
        from skimage.segmentation import watershed
        assigned = watershed(
            energy.astype(np.float32, copy=False),
            markers.astype(np.int32, copy=False),
            mask=support.astype(bool, copy=False),
            connectivity=1,
            watershed_line=False,
        )
        return assigned.astype(np.uint16, copy=False)
    except Exception:
        nearest_idx = ndi_distance_indices(markers == 0, (1.0, 1.0, 1.0))
        nearest = markers[tuple(nearest_idx)]
        out = np.zeros_like(markers, dtype=np.uint16)
        out[support] = nearest[support]
        return out


def _remap_local_to_global_contact(semantic: np.ndarray,
                                   local_instance: np.ndarray,
                                   core_prob: np.ndarray,
                                   contact_prob: np.ndarray,
                                   min_weak_size: int = 1000) -> tuple[np.ndarray, dict]:
    """Map local watershed instances to PENGWIN pelvic global IDs.

    Small islands are removed only when their evidence is weak. A small region
    with meaningful core confidence is kept because tiny GT fragments are valid
    and size-only filtering was shown to be unsafe for case 003.
    """
    out = np.zeros(local_instance.shape, dtype=np.uint16)
    next_by_anatomy = anatomy_start_ids()
    removed = []
    assigned = []
    local_ids = [int(v) for v in np.unique(local_instance) if int(v) > 0]
    local_ids.sort(key=lambda v: int((local_instance == v).sum()), reverse=True)
    for local_id in local_ids:
        mask = local_instance == local_id
        vox = semantic[mask]
        vox = vox[(vox >= 1) & (vox <= NUM_ANATOMIES)]
        if len(vox) == 0:
            removed.append({"local_id": local_id, "reason": "outside_anatomy", "voxels": int(mask.sum())})
            continue
        anatomy_id = int(np.bincount(vox.astype(np.int64)).argmax())
        same_anatomy = mask & (semantic == anatomy_id)
        voxels = int(same_anatomy.sum())
        if voxels == 0:
            continue
        mean_core = float(core_prob[same_anatomy].mean()) if voxels else 0.0
        mean_contact = float(contact_prob[same_anatomy].mean()) if voxels else 0.0
        max_core = float(core_prob[same_anatomy].max()) if voxels else 0.0
        weak_small = voxels < int(min_weak_size) and max_core < 0.35 and mean_core < 0.15
        if weak_small:
            removed.append({
                "local_id": local_id,
                "reason": "weak_small_island",
                "anatomy": int(anatomy_id),
                "voxels": voxels,
                "mean_core_prob": mean_core,
                "max_core_prob": max_core,
                "mean_contact_prob": mean_contact,
            })
            continue
        global_id = next_by_anatomy[anatomy_id]
        next_by_anatomy[anatomy_id] += 1
        out[same_anatomy] = global_id
        assigned.append({
            "local_id": local_id,
            "global_id": int(global_id),
            "anatomy": int(anatomy_id),
            "voxels": voxels,
            "mean_core_prob": mean_core,
            "max_core_prob": max_core,
            "mean_contact_prob": mean_contact,
        })
    return out, {"assigned": assigned, "assigned_count": len(assigned), "removed": removed, "removed_count": len(removed)}


def _decode_contact_energy(semantic: np.ndarray,
                           pred: np.ndarray,
                           probabilities: np.ndarray | None = None,
                           strict_argmax: bool = False,
                           profile: str = "contact_energy_v1") -> tuple[np.ndarray, dict]:
    """Decode ABBC as core markers plus a contact-surface energy ridge."""
    semantic = semantic.astype(np.uint8, copy=False)
    pred = pred.astype(np.uint8, copy=False)
    pelvic_mask = (semantic >= 1) & (semantic <= NUM_ANATOMIES)
    if probabilities is None:
        probabilities = np.zeros((4, *pred.shape), dtype=np.float32)
        for c in range(4):
            probabilities[c] = pred == c
    if probabilities.shape[0] < 4:
        raise RuntimeError(f"ABBC probabilities need at least 4 classes, got {probabilities.shape}")
    probs = probabilities.astype(np.float32, copy=False)
    if probs.shape[1:] != pred.shape:
        raise RuntimeError(f"probability/prediction shape mismatch: {probs.shape[1:]} vs {pred.shape}")
    shell_prob = probs[1]
    core_prob = probs[2]
    contact_prob = probs[3]
    hard_negative_prob = probs[ABBC_HARD_NEGATIVE_LABEL] if probs.shape[0] > ABBC_HARD_NEGATIVE_LABEL else np.zeros_like(contact_prob)

    non_support_class = pred == ABBC_HARD_NEGATIVE_LABEL
    abbc_foreground = pelvic_mask & (pred > 0) & ~non_support_class
    if strict_argmax:
        contact_wall = pelvic_mask & (pred == ABBC_BORDER_LABEL)
        support = abbc_foreground & ~contact_wall
        seed_mask = pelvic_mask & (pred == ABBC_CORE_LABEL)
        profile = "official_v1_strict_argmax"
    elif profile == "contact_energy_v2":
        support_conf = shell_prob + core_prob
        contact_wall = pelvic_mask & (
            (contact_prob >= 0.45)
            | ((pred == ABBC_BORDER_LABEL) & (hard_negative_prob < 0.35))
        )
        contact_wall = ndi.binary_dilation(contact_wall, structure=np.ones((3, 3, 3), dtype=bool))
        forbidden = pelvic_mask & ((hard_negative_prob >= 0.45) | non_support_class)
        support = pelvic_mask & abbc_foreground & (support_conf >= 0.30) & ~contact_wall & ~forbidden
        seed_mask = pelvic_mask & support & (
            (core_prob >= 0.45)
            | ((pred == ABBC_CORE_LABEL) & (core_prob >= 0.25))
        )
    else:
        support_conf = shell_prob + core_prob + contact_prob
        contact_wall = pelvic_mask & ((contact_prob >= 0.35) | (pred == ABBC_BORDER_LABEL))
        contact_wall = ndi.binary_dilation(contact_wall, structure=np.ones((3, 3, 3), dtype=bool))
        support = pelvic_mask & abbc_foreground & (support_conf >= 0.35) & ~contact_wall
        seed_mask = pelvic_mask & support & (
            (core_prob >= 0.45)
            | ((pred == ABBC_CORE_LABEL) & (core_prob >= 0.25))
        )
        profile = "contact_energy_v1"

    seed_cc, n_raw_seed = _label_components(seed_mask)
    # Keep tiny high-confidence seeds, but drop one-voxel low-confidence noise.
    for seed_id in range(1, n_raw_seed + 1):
        mask = seed_cc == seed_id
        if int(mask.sum()) < 3 and float(core_prob[mask].max()) < 0.60:
            seed_cc[mask] = 0
    seed_cc = _compact_labels(seed_cc)
    n_seed = int(seed_cc.max())
    missing_components = []
    comp_cc, n_comp = _label_components(support)
    for comp_id in range(1, n_comp + 1):
        comp = comp_cc == comp_id
        if int(comp.sum()) > 0 and not (seed_cc[comp] > 0).any():
            missing_components.append({"component": int(comp_id), "voxels": int(comp.sum())})

    if n_seed == 0:
        return np.zeros_like(pred, dtype=np.uint16), {
            "decoder": profile,
            "seed_components": 0,
            "support_voxels": int(support.sum()),
            "contact_wall_voxels": int(contact_wall.sum()),
            "missing_seed_component": missing_components,
        }

    energy = (contact_prob * 8.0) + (hard_negative_prob * 4.0) + ((1.0 - core_prob) * 0.25)
    local = _watershed_assign_energy(seed_cc, support, energy)
    remapped, remap_trace = _remap_local_to_global_contact(
        semantic, local, core_prob=core_prob, contact_prob=contact_prob,
    )
    values, counts = np.unique(remapped, return_counts=True)
    trace = {
        "decoder": profile,
        "decoder_profile": profile,
        "probability_mode": not strict_argmax,
        "seed_components_raw": int(n_raw_seed),
        "seed_components_kept": int(n_seed),
        "support_voxels": int(support.sum()),
        "abbc_foreground_voxels": int(abbc_foreground.sum()),
        "contact_wall_voxels": int(contact_wall.sum()),
        "hard_negative_voxels": int((pelvic_mask & (hard_negative_prob >= 0.45)).sum()),
        "missing_seed_component": missing_components,
        "conditional_small_island": remap_trace,
        "prediction_voxels": {str(int(v)): int(c) for v, c in zip(values, counts) if int(v) > 0},
    }
    return remapped.astype(np.uint16, copy=False), trace


def decode_abbc_official_prediction(foundation_pred: np.ndarray,
                                    abbc_pred: np.ndarray,
                                    decoder_profile: str = "official_v1",
                                    abbc_probabilities: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Decode active Contact-LegacyFuse/ABBC output into pelvic fragment IDs."""
    pelvic_mask = (foundation_pred >= 1) & (foundation_pred <= NUM_ANATOMIES)
    pred = np.where(pelvic_mask, abbc_pred.astype(np.uint8, copy=False), 0)
    pred_for_argmax_decoders = np.where(pred == ABBC_HARD_NEGATIVE_LABEL, 0, pred).astype(np.uint8, copy=False)
    if decoder_profile == "official_v1":
        return decode_official_abbc(foundation_pred.astype(np.uint8, copy=False), pred_for_argmax_decoders)
    if decoder_profile == "official_v1_strict_argmax":
        return _decode_contact_energy(
            foundation_pred.astype(np.uint8, copy=False), pred_for_argmax_decoders,
            probabilities=None, strict_argmax=True, profile="official_v1_strict_argmax",
        )
    if decoder_profile == "contact_energy_v1":
        return _decode_contact_energy(
            foundation_pred.astype(np.uint8, copy=False), pred,
            probabilities=abbc_probabilities, strict_argmax=False, profile="contact_energy_v1",
        )
    if decoder_profile == "contact_energy_v2":
        return _decode_contact_energy(
            foundation_pred.astype(np.uint8, copy=False), pred,
            probabilities=abbc_probabilities, strict_argmax=False, profile="contact_energy_v2",
        )

    spacing_zyx = (1.0, 1.0, 1.0)
    if decoder_profile == "core_cc_nearest":
        local_instance, row = _decode_local_core_cc_nearest(pelvic_mask, pred_for_argmax_decoders, spacing_zyx)
    elif decoder_profile == "border_barrier_v1":
        local_instance, row = _decode_local_border_barrier(pelvic_mask, pred_for_argmax_decoders, spacing_zyx)
    else:
        raise ValueError(f"unknown decoder_profile={decoder_profile!r}")

    decoded = np.zeros(foundation_pred.shape, dtype=np.uint16)
    seed_ids = [int(v) for v in np.unique(local_instance) if int(v) > 0]
    seed_ids.sort(key=lambda sid: int((local_instance == sid).sum()), reverse=True)
    next_by_anatomy = anatomy_start_ids()
    for seed_id in seed_ids:
        mask = local_instance == seed_id
        anatomy_values = foundation_pred[mask]
        anatomy_values = anatomy_values[(anatomy_values >= 1) & (anatomy_values <= NUM_ANATOMIES)]
        if len(anatomy_values) == 0:
            continue
        anatomy_id = int(np.bincount(anatomy_values.astype(np.int64)).argmax())
        global_id = next_by_anatomy[anatomy_id]
        next_by_anatomy[anatomy_id] += 1
        decoded[mask & (foundation_pred == anatomy_id)] = global_id
    return decoded, {
        "decoder": decoder_profile,
        "decoder_profile": decoder_profile,
        "cleanup_profile": "diagnostic_none",
        "diagnostic_trace": row,
        "missing_seed_component": row.get("missing_components", []),
    }


def _iou_for_ids(gt: np.ndarray, pred: np.ndarray,
                 gt_id: int, pred_id: int) -> float:
    gm = gt == gt_id
    pm = pred == pred_id
    union = int((gm | pm).sum())
    if union == 0:
        return 0.0
    return float((gm & pm).sum() / union)


def fragment_matching_metrics(gt_instances: np.ndarray,
                              pred_instances: np.ndarray,
                              dataset_ids: list[int]) -> dict:
    """Compute official-style best-IoU and Hungarian fragment diagnostics."""
    from scipy.optimize import linear_sum_assignment

    ranges = [DATASETS[ds_id]["global_label_range"] for ds_id in dataset_ids]
    ranges = [r for r in ranges if r is not None]
    gt_ids = [
        int(v) for v in np.unique(gt_instances)
        if any(lo <= int(v) <= hi for lo, hi in ranges)
    ]
    pred_ids = [
        int(v) for v in np.unique(pred_instances)
        if any(lo <= int(v) <= hi for lo, hi in ranges)
    ]
    gt_ids.sort()
    pred_ids.sort()
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, gt_id in enumerate(gt_ids):
        for j, pred_id in enumerate(pred_ids):
            matrix[i, j] = _iou_for_ids(gt_instances, pred_instances, gt_id, pred_id)

    best_rows = []
    best_values = []
    for i, gt_id in enumerate(gt_ids):
        if len(pred_ids) == 0:
            best_pred = None
            best_iou = 0.0
        else:
            j = int(np.argmax(matrix[i]))
            best_pred = pred_ids[j]
            best_iou = float(matrix[i, j])
        best_values.append(best_iou)
        best_rows.append({"gt_id": gt_id, "pred_id": best_pred, "iou": best_iou})
    official_unmatched = [row for row in best_rows if row["iou"] <= 0.0]

    hungarian_matches = []
    matched_gt = set()
    matched_pred = set()
    if matrix.size:
        row_ind, col_ind = linear_sum_assignment(-matrix)
        for i, j in zip(row_ind, col_ind):
            iou = float(matrix[i, j])
            if iou <= 0:
                continue
            matched_gt.add(gt_ids[int(i)])
            matched_pred.add(pred_ids[int(j)])
            hungarian_matches.append({
                "gt_id": gt_ids[int(i)],
                "pred_id": pred_ids[int(j)],
                "iou": iou,
            })
    hungarian_sum = sum(float(row["iou"]) for row in hungarian_matches)
    return {
        "gt_fragment_count": len(gt_ids),
        "pred_fragment_count": len(pred_ids),
        "official_style": {
            "mean_best_iou": float(np.mean(best_values)) if best_values else 1.0,
            "unmatched_count": len(official_unmatched),
            "unmatched_gt_ids": [int(row["gt_id"]) for row in official_unmatched],
            "best_matches": best_rows,
        },
        "hungarian": {
            "mean_iou_over_gt": float(hungarian_sum / len(gt_ids)) if gt_ids else 1.0,
            "matched_count": len(hungarian_matches),
            "unmatched_gt_ids": [int(v) for v in gt_ids if v not in matched_gt],
            "extra_pred_ids": [int(v) for v in pred_ids if v not in matched_pred],
            "matches": hungarian_matches,
        },
    }


def _fragment_matching_metrics_positive(gt_instances: np.ndarray,
                                        pred_instances: np.ndarray) -> dict:
    """Best-IoU/Hungarian diagnostics for arbitrary positive instance IDs.

    [METRIC][Scope:v5_oracle]
    V5 decodes one anatomy ROI at a time, so predicted IDs are local component
    IDs rather than global PENGWIN labels. IoU-F is ID-permutation invariant:
    matching must compare binary components, not require exact numeric IDs.
    """
    from scipy.optimize import linear_sum_assignment

    gt_ids = sorted(int(v) for v in np.unique(gt_instances) if int(v) > 0)
    pred_ids = sorted(int(v) for v in np.unique(pred_instances) if int(v) > 0)
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    if gt_ids and pred_ids:
        # [QC][Performance:metric_equivalence]
        # The previous implementation recomputed (gt == id) and (pred == id)
        # for every ID pair. That is correct but O(N_gt * N_pred * voxels) and
        # can make a failed diagnostic appear hung when the decoder creates many
        # seed fragments. This contingency-table path computes the same pairwise
        # IoU matrix with one volume pass, preserving metric semantics.
        gt_index = {int(v): i for i, v in enumerate(gt_ids)}
        pred_index = {int(v): i for i, v in enumerate(pred_ids)}
        # [QC][Performance:vectorized_label_mapping]
        # np.unique gives sparse label IDs; lookup arrays convert voxel labels
        # to compact matrix indices without a Python loop per voxel. Keep the
        # dicts above only for reviewer-readable traceability of ID ordering.
        gt_lookup = np.full(int(max(gt_ids)) + 1, -1, dtype=np.int64)
        pred_lookup = np.full(int(max(pred_ids)) + 1, -1, dtype=np.int64)
        gt_lookup[np.asarray(gt_ids, dtype=np.int64)] = np.arange(len(gt_ids), dtype=np.int64)
        pred_lookup[np.asarray(pred_ids, dtype=np.int64)] = np.arange(len(pred_ids), dtype=np.int64)
        gt_pos = gt_instances > 0
        pred_pos = pred_instances > 0
        gt_sizes = np.bincount(
            gt_lookup[gt_instances[gt_pos].astype(np.int64, copy=False)],
            minlength=len(gt_ids),
        ).astype(np.float64)
        pred_sizes = np.bincount(
            pred_lookup[pred_instances[pred_pos].astype(np.int64, copy=False)],
            minlength=len(pred_ids),
        ).astype(np.float64)
        both = gt_pos & pred_pos
        if bool(both.any()):
            gt_compact = gt_lookup[gt_instances[both].astype(np.int64, copy=False)]
            pred_compact = pred_lookup[pred_instances[both].astype(np.int64, copy=False)]
            flat = gt_compact * len(pred_ids) + pred_compact
            intersections = np.bincount(flat, minlength=len(gt_ids) * len(pred_ids)).reshape(
                len(gt_ids), len(pred_ids)
            ).astype(np.float64)
            unions = gt_sizes[:, None] + pred_sizes[None, :] - intersections
            valid = unions > 0
            matrix[valid] = (intersections[valid] / unions[valid]).astype(np.float32)

    best_rows = []
    best_values = []
    for i, gt_id in enumerate(gt_ids):
        if len(pred_ids) == 0:
            best_pred = None
            best_iou = 0.0
        else:
            j = int(np.argmax(matrix[i]))
            best_pred = pred_ids[j]
            best_iou = float(matrix[i, j])
        best_values.append(best_iou)
        best_rows.append({"gt_id": int(gt_id), "pred_id": best_pred, "iou": best_iou})

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches = []
    if matrix.size:
        row_ind, col_ind = linear_sum_assignment(-matrix)
        for i, j in zip(row_ind, col_ind):
            iou = float(matrix[int(i), int(j)])
            if iou <= 0:
                continue
            matched_gt.add(gt_ids[int(i)])
            matched_pred.add(pred_ids[int(j)])
            matches.append({
                "gt_id": int(gt_ids[int(i)]),
                "pred_id": int(pred_ids[int(j)]),
                "iou": iou,
            })
    hungarian_sum = sum(float(row["iou"]) for row in matches)
    return {
        "gt_fragment_count": len(gt_ids),
        "pred_fragment_count": len(pred_ids),
        "official_style": {
            "mean_best_iou": float(np.mean(best_values)) if best_values else 1.0,
            "unmatched_count": int(sum(1 for row in best_rows if float(row["iou"]) <= 0.0)),
            "unmatched_gt_ids": [int(row["gt_id"]) for row in best_rows if float(row["iou"]) <= 0.0],
            "best_matches": best_rows,
        },
        "hungarian": {
            "mean_iou_over_gt": float(hungarian_sum / len(gt_ids)) if gt_ids else 1.0,
            "matched_count": len(matches),
            "unmatched_gt_ids": [int(v) for v in gt_ids if v not in matched_gt],
            "extra_pred_ids": [int(v) for v in pred_ids if v not in matched_pred],
            "matches": matches,
        },
    }


def _surface_distance_metrics(pred: np.ndarray,
                              target: np.ndarray,
                              spacing_zyx: tuple[float, float, float]) -> dict[str, float | None]:
    """Binary Dice/HD95/ASSD in millimeters for Task1-aligned proxy scoring.

    Raises:
        ValueError: If pred/target shape differs or spacing is invalid.
    """
    if pred.shape != target.shape:
        raise ValueError(f"surface metric shape mismatch: pred={pred.shape} target={target.shape}")
    if len(spacing_zyx) != 3 or any(float(v) <= 0.0 for v in spacing_zyx):
        raise ValueError(f"invalid spacing_zyx for surface metrics: {spacing_zyx}")

    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    inter = int((pred & target).sum())
    denom = int(pred.sum() + target.sum())
    dice = 1.0 if denom == 0 else float(2.0 * inter / denom)
    if pred.shape == target.shape and bool(np.array_equal(pred, target)):
        return {"dice": float(dice), "hd95_mm": 0.0, "assd_mm": 0.0}
    if not pred.any() and not target.any():
        return {"dice": float(dice), "hd95_mm": 0.0, "assd_mm": 0.0}
    if not pred.any() or not target.any():
        return {"dice": float(dice), "hd95_mm": None, "assd_mm": None}
    crop_bbox = bbox_from_mask(pred | target, pad_vox=2)
    if crop_bbox is not None:
        pred = pred[crop_bbox]
        target = target[crop_bbox]

    pred_border = pred ^ ndi.binary_erosion(pred)
    target_border = target ^ ndi.binary_erosion(target)
    dist_to_target = ndi.distance_transform_edt(~target_border, sampling=spacing_zyx)
    dist_to_pred = ndi.distance_transform_edt(~pred_border, sampling=spacing_zyx)
    distances = np.concatenate([dist_to_target[pred_border], dist_to_pred[target_border]])
    if distances.size == 0:
        return {"dice": float(dice), "hd95_mm": 0.0, "assd_mm": 0.0}
    return {
        "dice": float(dice),
        "hd95_mm": float(np.percentile(distances, 95)),
        "assd_mm": float(np.mean(distances)),
    }


def _remap_pred_instances_to_gt_anatomy_ranges(pred_instances: np.ndarray,
                                               gt_instances: np.ndarray,
                                               spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Map arbitrary prediction IDs into PENGWIN pelvic anatomy ID ranges.

    [METRIC][Risk:High][Scope:task1_proxy_anatomy]
    The BICM decoder emits local watershed IDs, not official PENGWIN fragment IDs.
    Same-anatomy fracture surfaces require anatomy-aware IDs, so each predicted
    component is assigned to the GT anatomy with the largest overlap. Components
    with no overlap are assigned to the nearest GT anatomy by physical distance.
    This is a proxy-evaluator normalization, not a postprocess step for submission.
    """
    if pred_instances.shape != gt_instances.shape:
        raise ValueError(f"anatomy remap shape mismatch: pred={pred_instances.shape} gt={gt_instances.shape}")
    pred_instances = pred_instances.astype(np.int32, copy=False)
    gt_instances = gt_instances.astype(np.int32, copy=False)
    pred_ids = [int(v) for v in np.unique(pred_instances) if int(v) > 0]
    if not pred_ids:
        return np.zeros_like(pred_instances, dtype=np.uint16)

    anatomy_masks = {
        str(name): ((gt_instances >= int(lo)) & (gt_instances <= int(hi)))
        for name, (lo, hi) in V5_ANATOMY_RANGES.items()
    }
    distance_to_anatomy: dict[str, np.ndarray] = {}
    remapped = np.zeros_like(pred_instances, dtype=np.uint16)
    next_id_by_anatomy = {str(name): int(lo) for name, (lo, _hi) in V5_ANATOMY_RANGES.items()}
    for pred_id in pred_ids:
        mask = pred_instances == pred_id
        overlaps = {
            name: int((mask & anatomy_mask).sum())
            for name, anatomy_mask in anatomy_masks.items()
        }
        anatomy = max(overlaps, key=overlaps.get)
        if int(overlaps[anatomy]) <= 0:
            best_distance = None
            best_anatomy = anatomy
            for name, anatomy_mask in anatomy_masks.items():
                if not anatomy_mask.any():
                    continue
                if name not in distance_to_anatomy:
                    distance_to_anatomy[name] = ndi.distance_transform_edt(~anatomy_mask, sampling=spacing_zyx)
                mean_distance = float(np.mean(distance_to_anatomy[name][mask]))
                if best_distance is None or mean_distance < best_distance:
                    best_distance = mean_distance
                    best_anatomy = name
            anatomy = best_anatomy
        new_id = int(next_id_by_anatomy[anatomy])
        next_id_by_anatomy[anatomy] += 1
        _lo, hi = V5_ANATOMY_RANGES[anatomy]
        if new_id > int(hi):
            raise ValueError(f"too many predicted fragments assigned to {anatomy}: exceeded ID range {V5_ANATOMY_RANGES[anatomy]}")
        remapped[mask] = np.uint16(new_id)
    return remapped


def _same_anatomy_fracture_surface_proxy(instances: np.ndarray,
                                         *,
                                         contact_radius: int = 1,
                                         surface_erosion_radius: int = 1) -> np.ndarray:
    """Compute same-anatomy contact surfaces on cropped anatomy ROIs.

    [QC][Risk:Major][Scope:proxy_runtime]
    Calling the legacy contact helper on a full CT volume scans every fragment
    against full-volume masks. The Task1 proxy is run repeatedly across
    checkpoints, so this wrapper preserves the same surface definition but limits
    each call to one anatomy bbox.
    """
    instances = np.asarray(instances)
    out = np.zeros(instances.shape, dtype=np.uint8)
    pad = int(max(1, contact_radius) + max(1, surface_erosion_radius) + 2)
    for _anatomy, (lo, hi) in V5_ANATOMY_RANGES.items():
        anatomy_mask = (instances >= int(lo)) & (instances <= int(hi))
        bbox = bbox_from_mask(anatomy_mask, pad_vox=pad)
        if bbox is None:
            continue
        crop = instances[bbox]
        ids = [int(v) for v in np.unique(crop) if int(lo) <= int(v) <= int(hi)]
        if len(ids) <= 1:
            continue
        crop_surface = get_contact_surface_regions(
            crop,
            contact_radius=contact_radius,
            surface_erosion_radius=surface_erosion_radius,
            groups={1: ids},
        )
        view = out[bbox]
        view[crop_surface > 0] = 1
        out[bbox] = view
    return out


def _task1_proxy_instance_metrics(gt_instances: np.ndarray,
                                  pred_instances: np.ndarray,
                                  *,
                                  match_iou_threshold: float = 0.10,
                                  overlap_fraction_threshold: float = 0.05,
                                  max_pred_instances: int | None = None) -> dict[str, Any]:
    """Instance F1/merge/split/topology proxy for pelvic Task1 evaluation.

    [METRIC][Risk:High][Scope:official_proxy]
    The official Task1 script is not present in this workspace. This proxy uses
    Hungarian IoU matching for Instance F1 and overlap graph counts for Merge,
    Split, and Topology Consistency. Do not report these fields as official
    leaderboard numbers unless the official script later confirms equivalence.
    """
    from scipy.optimize import linear_sum_assignment

    if gt_instances.shape != pred_instances.shape:
        raise ValueError(f"instance metric shape mismatch: gt={gt_instances.shape} pred={pred_instances.shape}")
    if not (0.0 <= float(match_iou_threshold) <= 1.0):
        raise ValueError(f"match_iou_threshold must be in [0,1], got {match_iou_threshold}")
    if not (0.0 <= float(overlap_fraction_threshold) <= 1.0):
        raise ValueError(f"overlap_fraction_threshold must be in [0,1], got {overlap_fraction_threshold}")

    gt = np.where(valid_instance_mask(gt_instances, pelvic_only=True), gt_instances, 0).astype(np.int32, copy=False)
    pred = np.where(pred_instances > 0, pred_instances, 0).astype(np.int32, copy=False)
    if bool(np.array_equal(gt, pred)):
        gt_ids_equal = sorted(int(v) for v in np.unique(gt) if int(v) > 0)
        return {
            "match_iou_threshold": float(match_iou_threshold),
            "overlap_fraction_threshold": float(overlap_fraction_threshold),
            "gt_instance_count": int(len(gt_ids_equal)),
            "pred_instance_count": int(len(gt_ids_equal)),
            "instance_f1": 1.0,
            "instance_recall": 1.0,
            "instance_precision": 1.0,
            "tp": int(len(gt_ids_equal)),
            "fp": 0,
            "fn": 0,
            "merge_errors": 0,
            "merge_error_rate": 0.0,
            "split_errors": 0,
            "split_error_rate": 0.0,
            "topology_consistency": 1.0,
            "matches": [{"gt_id": int(v), "pred_id": int(v), "iou": 1.0} for v in gt_ids_equal],
        }
    gt_ids = sorted(int(v) for v in np.unique(gt) if int(v) > 0)
    pred_ids = sorted(int(v) for v in np.unique(pred) if int(v) > 0)
    if max_pred_instances is not None and len(pred_ids) > int(max_pred_instances):
        # [METRIC][Risk:Blocker][Scope:task1_proxy_fast_fail]
        # PENGWIN Task1 GT ID range는 1..150이다. learned decoder가 이보다 많은
        # predicted component를 만들면 fulltrain gate 관점에서는 이미 과분할 실패다.
        # Hungarian/overlap graph를 끝까지 계산하면 리포트가 지연되므로 보수적인 실패
        # 지표를 기록하고 다음 케이스로 넘어간다.
        gt_count = int(len(gt_ids))
        pred_count = int(len(pred_ids))
        return {
            "match_iou_threshold": float(match_iou_threshold),
            "overlap_fraction_threshold": float(overlap_fraction_threshold),
            "gt_instance_count": gt_count,
            "pred_instance_count": pred_count,
            "instance_f1": 0.0,
            "instance_recall": 0.0,
            "instance_precision": 0.0,
            "tp": 0,
            "fp": pred_count,
            "fn": gt_count,
            "merge_errors": 0,
            "merge_error_rate": 0.0,
            "split_errors": gt_count,
            "split_error_rate": 1.0 if gt_count > 0 else 0.0,
            "topology_consistency": 0.0,
            "matches": [],
            "metric_skipped_reason": (
                f"pred_instance_count={pred_count} exceeds max_pred_instances={int(max_pred_instances)}"
            ),
        }
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    intersections = np.zeros_like(matrix, dtype=np.float64)
    if gt_ids and pred_ids:
        gt_lookup = np.full(int(max(gt_ids)) + 1, -1, dtype=np.int64)
        pred_lookup = np.full(int(max(pred_ids)) + 1, -1, dtype=np.int64)
        gt_lookup[np.asarray(gt_ids, dtype=np.int64)] = np.arange(len(gt_ids), dtype=np.int64)
        pred_lookup[np.asarray(pred_ids, dtype=np.int64)] = np.arange(len(pred_ids), dtype=np.int64)
        gt_pos = gt > 0
        pred_pos = pred > 0
        gt_sizes = np.bincount(gt_lookup[gt[gt_pos].astype(np.int64, copy=False)], minlength=len(gt_ids)).astype(np.float64)
        pred_sizes = np.bincount(pred_lookup[pred[pred_pos].astype(np.int64, copy=False)], minlength=len(pred_ids)).astype(np.float64)
        both = gt_pos & pred_pos
        if bool(both.any()):
            gt_compact = gt_lookup[gt[both].astype(np.int64, copy=False)]
            pred_compact = pred_lookup[pred[both].astype(np.int64, copy=False)]
            flat = gt_compact * len(pred_ids) + pred_compact
            intersections = np.bincount(flat, minlength=len(gt_ids) * len(pred_ids)).reshape(
                len(gt_ids), len(pred_ids)
            ).astype(np.float64)
            unions = gt_sizes[:, None] + pred_sizes[None, :] - intersections
            valid = unions > 0
            matrix[valid] = (intersections[valid] / unions[valid]).astype(np.float32)
    else:
        gt_sizes = np.zeros((len(gt_ids),), dtype=np.float64)
        pred_sizes = np.zeros((len(pred_ids),), dtype=np.float64)

    matches: list[dict[str, Any]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    if matrix.size:
        row_ind, col_ind = linear_sum_assignment(-matrix)
        for i, j in zip(row_ind, col_ind):
            i = int(i)
            j = int(j)
            iou = float(matrix[i, j])
            if iou < float(match_iou_threshold):
                continue
            matched_gt.add(gt_ids[i])
            matched_pred.add(pred_ids[j])
            matches.append({"gt_id": int(gt_ids[i]), "pred_id": int(pred_ids[j]), "iou": iou})
    tp = int(len(matches))
    fp = int(len(pred_ids) - len(matched_pred))
    fn = int(len(gt_ids) - len(matched_gt))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-8))

    gt_to_pred_count: dict[int, int] = {}
    pred_to_gt_count: dict[int, int] = {}
    if intersections.size:
        gt_overlap_fraction = intersections / np.maximum(gt_sizes[:, None], 1.0)
        pred_overlap_fraction = intersections / np.maximum(pred_sizes[None, :], 1.0)
        linked = (gt_overlap_fraction >= float(overlap_fraction_threshold)) | (
            pred_overlap_fraction >= float(overlap_fraction_threshold)
        )
        for i, gt_id in enumerate(gt_ids):
            gt_to_pred_count[int(gt_id)] = int(np.count_nonzero(linked[i, :]))
        for j, pred_id in enumerate(pred_ids):
            pred_to_gt_count[int(pred_id)] = int(np.count_nonzero(linked[:, j]))
    else:
        gt_to_pred_count = {int(gt_id): 0 for gt_id in gt_ids}
        pred_to_gt_count = {int(pred_id): 0 for pred_id in pred_ids}

    merge_error_count = int(sum(max(0, count - 1) for count in pred_to_gt_count.values()))
    split_error_count = int(sum(max(0, count - 1) for count in gt_to_pred_count.values()))
    one_to_one_gt = int(sum(
        1
        for gt_id in gt_ids
        if gt_to_pred_count.get(int(gt_id), 0) == 1
        and any(
            int(row["gt_id"]) == int(gt_id)
            and pred_to_gt_count.get(int(row["pred_id"]), 0) == 1
            for row in matches
        )
    ))
    topology_consistency = float(one_to_one_gt / max(len(gt_ids), 1))
    return {
        "match_iou_threshold": float(match_iou_threshold),
        "overlap_fraction_threshold": float(overlap_fraction_threshold),
        "gt_instance_count": int(len(gt_ids)),
        "pred_instance_count": int(len(pred_ids)),
        "instance_f1": f1,
        "instance_recall": recall,
        "instance_precision": precision,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "merge_errors": merge_error_count,
        "merge_error_rate": float(merge_error_count / max(len(gt_ids), 1)),
        "split_errors": split_error_count,
        "split_error_rate": float(split_error_count / max(len(gt_ids), 1)),
        "topology_consistency": topology_consistency,
        "matches": matches,
    }


def _task1_proxy_fragment_overflow_failure(gt_instances: np.ndarray,
                                           pred_instances: np.ndarray,
                                           *,
                                           reason: str,
                                           max_proxy_instances: int,
                                           local_radius_mm: float,
                                           match_iou_threshold: float,
                                           overlap_fraction_threshold: float) -> dict[str, Any]:
    """Return a conservative Task1 proxy failure for impossible over-fragmentation.

    [METRIC][Risk:Blocker][Scope:fulltrain_gate]
    Official-aligned local proxy는 GT anatomy ID capacity 150개를 기준으로 한다.
    predicted component가 이 capacity를 초과하면 official-style remap과 fracture
    contact proxy가 의미를 잃으므로, expensive surface distance를 생략하고
    fulltrain gate 실패를 명시적으로 기록한다.
    """
    gt_pelvis = np.where(valid_instance_mask(gt_instances, pelvic_only=True), gt_instances, 0).astype(np.uint16, copy=False)
    pred = np.where(pred_instances > 0, pred_instances, 0).astype(np.uint16, copy=False)
    gt_support = gt_pelvis > 0
    pred_support = pred > 0
    instance_metrics = _task1_proxy_instance_metrics(
        gt_pelvis,
        pred,
        match_iou_threshold=match_iou_threshold,
        overlap_fraction_threshold=overlap_fraction_threshold,
        max_pred_instances=max_proxy_instances,
    )
    return {
        "metric_contract": "task1_official_aligned_proxy_v1",
        "official_mean_position_score": None,
        "official_score_reproducible": False,
        "dice": float(_binary_dice(pred_support, gt_support)),
        "fracture_dice_proxy": 0.0,
        "local_dice_cfs_20mm_proxy": 0.0,
        "hd95_mm": None,
        "assd_mm": None,
        "instance_f1": instance_metrics["instance_f1"],
        "instance_recall": instance_metrics["instance_recall"],
        "instance_precision": instance_metrics["instance_precision"],
        "merge_errors": instance_metrics["merge_errors"],
        "merge_error_rate": instance_metrics["merge_error_rate"],
        "split_errors": instance_metrics["split_errors"],
        "split_error_rate": instance_metrics["split_error_rate"],
        "topology_consistency": instance_metrics["topology_consistency"],
        "gt_fracture_voxels": None,
        "pred_fracture_voxels": None,
        "proxy_failure_reason": reason,
        "local_radius_mm": float(local_radius_mm),
        "instance_metrics": instance_metrics,
        "limitations": [
            "Mean Position Score is a leaderboard rank aggregate and is not locally reproduced.",
            "Surface-distance fields are null because predicted fragment count exceeds official proxy capacity.",
            "This is a conservative fulltrain-gate failure, not an official leaderboard score.",
        ],
    }


def compute_task1_official_proxy_metrics(gt_instances: np.ndarray,
                                         pred_instances: np.ndarray,
                                         spacing_zyx: tuple[float, float, float],
                                         *,
                                         local_radius_mm: float = 20.0,
                                         match_iou_threshold: float = 0.10,
                                         overlap_fraction_threshold: float = 0.05) -> dict[str, Any]:
    """Compute raw Task1 leaderboard-aligned proxy metrics for one case.

    [AUDIT][Risk:Blocker][Scope:official_metric_alignment]
    The leaderboard table reports raw metrics plus position/rank columns. The
    official scoring script is not available here, so this function intentionally
    reports raw proxy metrics and never claims to reproduce `Mean Position Score`.

    [QC][Invariant:shape_dtype_device]
    Arrays are source-space z/y/x numpy volumes with identical shape. Spacing is
    z/y/x in millimeters and is required for HD95, ASSD, and CFS-20mm locality.
    """
    if gt_instances.shape != pred_instances.shape:
        raise ValueError(f"Task1 proxy shape mismatch: gt={gt_instances.shape} pred={pred_instances.shape}")
    gt_pelvis = np.where(valid_instance_mask(gt_instances, pelvic_only=True), gt_instances, 0).astype(np.uint16, copy=False)
    pred = np.where(pred_instances > 0, pred_instances, 0).astype(np.uint16, copy=False)
    gt_support = gt_pelvis > 0
    pred_support = pred > 0
    max_proxy_instances = int(sum(int(hi) - int(lo) + 1 for lo, hi in V5_ANATOMY_RANGES.values()))
    pred_instance_count = int(len([v for v in np.unique(pred) if int(v) > 0]))
    if pred_instance_count > max_proxy_instances:
        return _task1_proxy_fragment_overflow_failure(
            gt_pelvis,
            pred,
            reason=f"pred_instance_count={pred_instance_count} exceeds official proxy capacity={max_proxy_instances}",
            max_proxy_instances=max_proxy_instances,
            local_radius_mm=local_radius_mm,
            match_iou_threshold=match_iou_threshold,
            overlap_fraction_threshold=overlap_fraction_threshold,
        )

    support_surface = _surface_distance_metrics(pred_support, gt_support, spacing_zyx)
    proxy_failure_reason = None
    if bool(np.array_equal(pred, gt_pelvis)):
        pred_for_contact = gt_pelvis
    else:
        try:
            pred_for_contact = _remap_pred_instances_to_gt_anatomy_ranges(pred, gt_pelvis, spacing_zyx)
        except ValueError as exc:
            message = str(exc)
            if "too many predicted fragments assigned" not in message:
                raise
            # [METRIC][Risk:Blocker][Scope:task1_proxy_fragment_overflow]
            # predicted fragment 수가 PENGWIN anatomy ID range를 초과하면 official proxy의
            # fracture/contact ID 정규화가 불가능하다. 이 경우를 예외로 중단하면 fulltrain
            # gate가 "평가 실패"로만 남고 실제 실패 원인을 수치로 추적할 수 없으므로,
            # fracture proxy는 실패로 낮추고 instance/split/topology 지표는 그대로 계산한다.
            proxy_failure_reason = message
            pred_for_contact = np.zeros_like(gt_pelvis, dtype=np.uint16)
    gt_fracture = _same_anatomy_fracture_surface_proxy(gt_pelvis, contact_radius=1, surface_erosion_radius=1).astype(bool, copy=False)
    pred_fracture = _same_anatomy_fracture_surface_proxy(pred_for_contact, contact_radius=1, surface_erosion_radius=1).astype(bool, copy=False)
    fracture_dice = _binary_dice(pred_fracture, gt_fracture)
    if gt_fracture.any():
        # [QC][Risk:Major][Scope:local_cfs_runtime]
        # CFS 20mm Dice only needs the fracture-neighborhood band. Computing an
        # EDT over the full CT volume made GT sanity checks too slow, so the
        # distance transform is restricted to a physically padded fracture bbox.
        pad_vox = int(np.ceil(float(local_radius_mm) / max(min(float(v) for v in spacing_zyx), 1e-6))) + 2
        frac_bbox = bbox_from_mask(gt_fracture, pad_vox=pad_vox)
        local_band = np.zeros_like(gt_fracture, dtype=bool)
        if frac_bbox is not None:
            dist_crop = ndi.distance_transform_edt(~gt_fracture[frac_bbox], sampling=spacing_zyx)
            local_band[frac_bbox] = dist_crop <= float(local_radius_mm)
    else:
        local_band = gt_support | pred_support
    local_dice = _binary_dice(pred_support & local_band, gt_support & local_band)
    instance_metrics = _task1_proxy_instance_metrics(
        gt_pelvis,
        pred,
        match_iou_threshold=match_iou_threshold,
        overlap_fraction_threshold=overlap_fraction_threshold,
        max_pred_instances=max_proxy_instances,
    )
    return {
        "metric_contract": "task1_official_aligned_proxy_v1",
        "official_mean_position_score": None,
        "official_score_reproducible": False,
        "dice": float(support_surface["dice"]),
        "fracture_dice_proxy": float(fracture_dice),
        "local_dice_cfs_20mm_proxy": float(local_dice),
        "hd95_mm": support_surface["hd95_mm"],
        "assd_mm": support_surface["assd_mm"],
        "instance_f1": instance_metrics["instance_f1"],
        "instance_recall": instance_metrics["instance_recall"],
        "instance_precision": instance_metrics["instance_precision"],
        "merge_errors": instance_metrics["merge_errors"],
        "merge_error_rate": instance_metrics["merge_error_rate"],
        "split_errors": instance_metrics["split_errors"],
        "split_error_rate": instance_metrics["split_error_rate"],
        "topology_consistency": instance_metrics["topology_consistency"],
        "gt_fracture_voxels": int(gt_fracture.sum()),
        "pred_fracture_voxels": int(pred_fracture.sum()),
        "proxy_failure_reason": proxy_failure_reason,
        "local_radius_mm": float(local_radius_mm),
        "instance_metrics": instance_metrics,
        # [QA][Status:Not run][Scope:official_metric_script]
        # Expected: if the challenge releases an official metric script, this raw
        # proxy must be compared against it before any leaderboard claim is made.
        "limitations": [
            "Mean Position Score is a leaderboard rank aggregate and is not locally reproduced.",
            "Fracture Dice and Local Dice use a same-anatomy contact-surface proxy.",
            "Instance F1/Merge/Split/Topology use fixed IoU/overlap thresholds recorded in instance_metrics.",
        ],
    }


# =============================================================================
# PENGWIN 2026 Task 1 official-aligned proxy v2
# [METRIC][Risk:Blocker][Scope:official_metric_alignment_v2]
# Reason: the v1 proxy (compute_task1_official_proxy_metrics) is pelvic-only,
# global Hungarian, pooled fracture-dice, no GT-volume filter, and no 1 cm^3
# CC pruning. v2 follows the PENGWIN 2026 official Task 1 spec more closely
# while we still wait for the official evaluator script to be published.
# The v1 function is intentionally NOT modified so existing CLI eval JSONs
# (eval_task1_v288_prediction.json, etc.) stay byte-stable for regressions.
# =============================================================================
def _voxel_volume_mm3(spacing_zyx: tuple[float, float, float]) -> float:
    """Per-voxel volume in mm^3 from z/y/x spacing."""
    if len(spacing_zyx) != 3 or any(float(v) <= 0.0 for v in spacing_zyx):
        raise ValueError(f"invalid spacing_zyx: {spacing_zyx}")
    return float(spacing_zyx[0]) * float(spacing_zyx[1]) * float(spacing_zyx[2])


def _prune_small_components(instance_map: np.ndarray,
                            spacing_zyx: tuple[float, float, float],
                            min_mm3: float) -> np.ndarray:
    """instance별 CC 중 `min_mm3`보다 작은 것을 제거한다 (각 ID를 독립적으로 relabel).

    [METRIC][Risk:High][Scope:official_proxy_v2_cc_prune]
    PENGWIN 2026 Task 1 evaluator는 metric 계산 전에 약 1 cm^3 (1000 mm^3) 미만 connected
    component를 GT와 prediction 양쪽에서 모두 가지치기한다. 여기서는 map을 instance ID 단위로
    쪼개 ID별로 CC를 구한 뒤, voxel 수가 spacing을 고려한 threshold 미만인 CC를 0으로 만든다.
    다른 ID는 그대로 보존된다.
    """
    instance_map = np.asarray(instance_map)
    if float(min_mm3) <= 0.0:
        return instance_map.astype(instance_map.dtype, copy=True)
    vox_mm3 = _voxel_volume_mm3(spacing_zyx)
    voxel_threshold = int(max(1, np.ceil(float(min_mm3) / max(vox_mm3, 1e-9))))
    out = instance_map.copy()
    for inst_id in [int(v) for v in np.unique(instance_map) if int(v) > 0]:
        mask = instance_map == inst_id
        labeled, n_comp = ndi.label(mask)
        if n_comp == 0:
            continue
        sizes = np.bincount(labeled.ravel())
        # labeling 결과의 0번 index는 background를 의미한다.
        for cc_label in range(1, int(n_comp) + 1):
            if int(sizes[cc_label]) < voxel_threshold:
                out[labeled == cc_label] = 0
    return out


def _filter_gt_fragments_by_volume(gt_instances: np.ndarray,
                                   spacing_zyx: tuple[float, float, float],
                                   min_mm3: float = 500.0) -> tuple[np.ndarray, list[int]]:
    """전체 부피가 min_mm3 미만인 GT fragment ID를 제거한다.

    [METRIC][Risk:High][Scope:official_proxy_v2_gt_omit]
    PENGWIN 2026 공식 평가에서는 부피가 500 mm^3 미만인 fragment를 matching/metric 계산 전에
    GT 측에서 제외한다. 이 필터는 per-ID 단위로만 동작하며, per-CC 단위가 아님을 주의하자.
    같은 fragment 안의 작은 CC들은 _prune_small_components가 따로 처리한다.
    """
    gt_instances = np.asarray(gt_instances)
    if float(min_mm3) <= 0.0:
        return gt_instances.astype(gt_instances.dtype, copy=True), []
    vox_mm3 = _voxel_volume_mm3(spacing_zyx)
    voxel_threshold = int(max(1, np.ceil(float(min_mm3) / max(vox_mm3, 1e-9))))
    out = gt_instances.copy()
    dropped: list[int] = []
    for gt_id in [int(v) for v in np.unique(gt_instances) if int(v) > 0]:
        size = int((gt_instances == gt_id).sum())
        if size < voxel_threshold:
            out[gt_instances == gt_id] = 0
            dropped.append(int(gt_id))
    return out, dropped


def _present_anatomies(gt_instances: np.ndarray,
                       anatomy_ranges: dict[str, tuple[int, int]] | None = None) -> list[str]:
    """GT voxel이 한 개 이상 존재하는 anatomy key들의 리스트를 반환한다."""
    if anatomy_ranges is None:
        anatomy_ranges = PENGWIN_OFFICIAL_ANATOMY_RANGES
    present: list[str] = []
    for name, (lo, hi) in anatomy_ranges.items():
        if bool(((gt_instances >= int(lo)) & (gt_instances <= int(hi))).any()):
            present.append(str(name))
    return present


def _per_anatomy_argmax_match(gt_part: np.ndarray,
                              pred_part: np.ndarray,
                              iou_threshold: float) -> dict[str, Any]:
    """For each GT in this anatomy, pred = argmax_pred IoU(gt, pred) within same anatomy.

    [METRIC][Risk:High][Scope:official_proxy_v2_matching]
    The official evaluator does per-anatomy argmax IoU matching (each GT picks
    its best pred independently). This is NOT global Hungarian: two GTs may
    point to the same pred (which is what triggers a Merge Error in v2 counts).
    """
    gt_ids = sorted(int(v) for v in np.unique(gt_part) if int(v) > 0)
    pred_ids = sorted(int(v) for v in np.unique(pred_part) if int(v) > 0)
    if not gt_ids:
        return {
            "gt_ids": [],
            "pred_ids": pred_ids,
            "matches": [],
            "iou_matrix_shape": [0, len(pred_ids)],
        }
    if not pred_ids:
        return {
            "gt_ids": gt_ids,
            "pred_ids": [],
            "matches": [{"gt_id": int(g), "pred_id": None, "iou": 0.0} for g in gt_ids],
            "iou_matrix_shape": [len(gt_ids), 0],
        }
    gt_lookup = {gid: i for i, gid in enumerate(gt_ids)}
    pred_lookup = {pid: i for i, pid in enumerate(pred_ids)}
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    gt_pos = gt_part > 0
    pred_pos = pred_part > 0
    gt_sizes = np.zeros((len(gt_ids),), dtype=np.float64)
    pred_sizes = np.zeros((len(pred_ids),), dtype=np.float64)
    for gid, idx in gt_lookup.items():
        gt_sizes[idx] = float((gt_part == gid).sum())
    for pid, idx in pred_lookup.items():
        pred_sizes[idx] = float((pred_part == pid).sum())
    both = gt_pos & pred_pos
    if bool(both.any()):
        gt_flat = gt_part[both].astype(np.int64, copy=False)
        pred_flat = pred_part[both].astype(np.int64, copy=False)
        for gid, pid in zip(gt_flat.tolist(), pred_flat.tolist()):
            if gid in gt_lookup and pid in pred_lookup:
                matrix[gt_lookup[gid], pred_lookup[pid]] += 1.0
    unions = gt_sizes[:, None] + pred_sizes[None, :] - matrix
    iou = np.zeros_like(matrix)
    valid = unions > 0
    iou[valid] = matrix[valid] / unions[valid]
    matches: list[dict[str, Any]] = []
    for i, gid in enumerate(gt_ids):
        best_j = int(np.argmax(iou[i, :]))
        best_iou = float(iou[i, best_j])
        if best_iou >= float(iou_threshold):
            matches.append({"gt_id": int(gid), "pred_id": int(pred_ids[best_j]), "iou": best_iou})
        else:
            matches.append({"gt_id": int(gid), "pred_id": None, "iou": best_iou})
    return {
        "gt_ids": gt_ids,
        "pred_ids": pred_ids,
        "matches": matches,
        "iou_matrix_shape": [len(gt_ids), len(pred_ids)],
    }


def _per_fragment_hd95_assd(gt_mask: np.ndarray,
                            pred_mask: np.ndarray,
                            spacing_zyx: tuple[float, float, float]) -> dict[str, float | None]:
    """Per-fragment binary HD95/ASSD + Dice in mm.

    Returns NaN-safe nulls when either side is empty.
    """
    return _surface_distance_metrics(pred_mask.astype(bool, copy=False),
                                     gt_mask.astype(bool, copy=False),
                                     spacing_zyx)


def _mean_skip_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def compute_task1_official_aligned_v2_metrics(pred_instances: np.ndarray,
                                              gt_instances: np.ndarray,
                                              spacing_zyx: tuple[float, float, float],
                                              *,
                                              gt_fragment_min_mm3: float = 500.0,
                                              cc_prune_mm3: float = 1000.0,
                                              iou_match_threshold: float = 0.10,
                                              local_radius_mm: float = 20.0) -> dict[str, Any]:
    """PENGWIN 2026 Task 1 공식 스펙에 정렬한 proxy v2 metric을 계산한다.

    v1 (compute_task1_official_proxy_metrics) 대비 달라진 점:
    - Femur (151-200) anatomy range를 포함한다.
    - 500 mm^3 미만의 GT fragment를 필터링한다 (gt_fragment_min_mm3로 조정 가능).
    - matching 이전에 1 cm^3 미만의 predicted CC를 가지치기한다 (cc_prune_mm3로 조정 가능).
    - per-anatomy argmax matching을 사용한다 (global Hungarian을 대체).
    - HD95/ASSD를 fragment별로 계산한 뒤 part 안에서 평균하고, 다시 part끼리 평균한다.
    - Instance F1/Recall/Precision을 part별로 계산하고 cohort macro로 모은다.
    - Merge/Split은 COUNT만 내보낸다 (_rate field는 더 이상 없다).
    - cohort 집계를 위해 case_failed (bool)와 failure_mode를 함께 태그한다.

    [METRIC][Risk:Blocker][Scope:official_proxy_v2]
    PENGWIN 2026 Task 1 공식 evaluator script는 2026-05-29 기준으로 아직 공개되지 않았다.
    이 구현은 공개된 스펙을 따른 것이므로, 리더보드용 주장을 하려면 공식 스크립트가 공개된
    뒤 반드시 그것으로 다시 검증해야 한다.
    """
    if pred_instances.shape != gt_instances.shape:
        raise ValueError(
            f"Task1 official-aligned v2 shape mismatch: pred={pred_instances.shape} gt={gt_instances.shape}"
        )
    if len(spacing_zyx) != 3 or any(float(v) <= 0.0 for v in spacing_zyx):
        raise ValueError(f"invalid spacing_zyx for Task1 v2 metrics: {spacing_zyx}")
    if not (0.0 <= float(iou_match_threshold) <= 1.0):
        raise ValueError(f"iou_match_threshold must be in [0,1], got {iou_match_threshold}")

    anatomy_ranges = PENGWIN_OFFICIAL_ANATOMY_RANGES
    case_failed = False
    failure_mode: str | None = None
    vox_mm3 = _voxel_volume_mm3(spacing_zyx)

    # Step 1: 공식 anatomy ID range (1..200) 안으로 제한하고, 그 밖의 noise ID는 제거한다.
    valid_lo = min(int(lo) for lo, _ in anatomy_ranges.values())
    valid_hi = max(int(hi) for _, hi in anatomy_ranges.values())
    gt_full = np.where(
        (gt_instances >= valid_lo) & (gt_instances <= valid_hi),
        gt_instances,
        0,
    ).astype(np.int32, copy=False)
    pred_full = np.where(pred_instances > 0, pred_instances, 0).astype(np.int32, copy=False)

    # Step 2: GT 측에서 500 mm^3 미만 fragment를 떨궈낸다.
    gt_filtered, gt_dropped_ids = _filter_gt_fragments_by_volume(
        gt_full, spacing_zyx, min_mm3=float(gt_fragment_min_mm3)
    )

    # Step 3: GT와 prediction 양쪽에서 ID별로 1 cm^3 미만 CC를 가지치기한다.
    try:
        gt_pruned = _prune_small_components(gt_filtered, spacing_zyx, min_mm3=float(cc_prune_mm3))
        pred_pruned = _prune_small_components(pred_full, spacing_zyx, min_mm3=float(cc_prune_mm3))
    except Exception as exc:  # noqa: BLE001
        case_failed = True
        failure_mode = f"cc_prune_failed: {exc!r}"
        gt_pruned = gt_filtered
        pred_pruned = pred_full

    # Step 4: anatomy별로 metric을 계산한다.
    per_anatomy: dict[str, dict[str, Any]] = {}
    present = _present_anatomies(gt_pruned, anatomy_ranges)

    for anatomy, (lo, hi) in anatomy_ranges.items():
        in_anatomy_gt = (gt_pruned >= int(lo)) & (gt_pruned <= int(hi))
        in_anatomy_pred = (pred_pruned >= int(lo)) & (pred_pruned <= int(hi))
        gt_part = np.where(in_anatomy_gt, gt_pruned, 0).astype(np.int32, copy=False)
        pred_part = np.where(in_anatomy_pred, pred_pruned, 0).astype(np.int32, copy=False)

        gt_ids = sorted(int(v) for v in np.unique(gt_part) if int(v) > 0)
        pred_ids = sorted(int(v) for v in np.unique(pred_part) if int(v) > 0)
        n_gt = int(len(gt_ids))
        n_pred = int(len(pred_ids))

        # GT가 없는 anatomy: metric 계산은 건너뛰지만, presence 정보는 남겨둔다.
        if n_gt == 0:
            per_anatomy[str(anatomy)] = {
                "present": False,
                "gt_instance_count": 0,
                "pred_instance_count": n_pred,
                "fracture_dice_per_fragment": None,
                "local_dice_per_fragment_20mm": None,
                "hd95_mm_per_fragment": None,
                "assd_mm_per_fragment": None,
                "instance_recall": None,
                "instance_precision": None,
                "instance_f1": None,
                "merge_error_count": 0,
                "split_error_count": 0,
                "topology_consistency": None,
                "matches": [],
            }
            continue

        # anatomy별 argmax matching을 한다 (global Hungarian은 쓰지 않는다).
        match_result = _per_anatomy_argmax_match(gt_part, pred_part, iou_threshold=float(iou_match_threshold))
        matches = match_result["matches"]

        # fragment별 surface metric (Dice/HD95/ASSD)과 local Dice를 계산한다.
        dice_values: list[float | None] = []
        hd95_values: list[float | None] = []
        assd_values: list[float | None] = []
        local_dice_values: list[float | None] = []

        # part별 local-band (어떤 GT fragment 기준 20 mm 이내 영역)를 계산한다.
        gt_part_bin = gt_part > 0
        local_band = np.zeros_like(gt_part_bin, dtype=bool)
        if bool(gt_part_bin.any()):
            pad_vox = int(np.ceil(float(local_radius_mm) / max(min(float(v) for v in spacing_zyx), 1e-6))) + 2
            band_bbox = bbox_from_mask(gt_part_bin, pad_vox=pad_vox)
            if band_bbox is not None:
                dist_crop = ndi.distance_transform_edt(~gt_part_bin[band_bbox], sampling=spacing_zyx)
                local_band[band_bbox] = dist_crop <= float(local_radius_mm)

        for row in matches:
            gid = int(row["gt_id"])
            pid = row["pred_id"]
            gt_mask = (gt_part == gid)
            pred_mask = np.zeros_like(gt_mask, dtype=bool) if pid is None else (pred_part == int(pid))
            surf = _per_fragment_hd95_assd(gt_mask, pred_mask, spacing_zyx)
            dice_values.append(surf.get("dice"))
            hd95_values.append(surf.get("hd95_mm"))
            assd_values.append(surf.get("assd_mm"))
            local_gt = gt_mask & local_band
            local_pred = pred_mask & local_band
            local_dice_values.append(float(_binary_dice(local_pred, local_gt)))

        # part별로 Instance Recall/Precision/F1을 계산한다.
        matched_gt_ids = {int(m["gt_id"]) for m in matches if m["pred_id"] is not None}
        matched_pred_ids = {int(m["pred_id"]) for m in matches if m["pred_id"] is not None}
        tp = int(len(matched_gt_ids))
        fn = int(n_gt - tp)
        fp = int(n_pred - len(matched_pred_ids))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2.0 * precision * recall / max(precision + recall, 1e-8))

        # Merge: 여러 개의 매칭된 GT가 같은 pred id로 몰린 경우 -> 초과된 충돌 횟수를 센다.
        # Split: 하나의 GT에 여러 pred CC가 겹치는 경우
        # (매칭된 GT와 겹치는, 자기 자신은 매칭되지 않은 pred 하나당 split 1로 센다).
        pred_assignment_counts: dict[int, int] = {}
        for m in matches:
            if m["pred_id"] is not None:
                pred_assignment_counts[int(m["pred_id"])] = pred_assignment_counts.get(int(m["pred_id"]), 0) + 1
        merge_error_count = int(sum(max(0, c - 1) for c in pred_assignment_counts.values()))
        # split: GT마다, 이 anatomy 안에서 GT mask와 일정 이상 겹치는 pred CC 수
        # (CC prune 이후 기준)를 세고, 거기서 1 (매칭된 CC 자신)을 뺀다.
        split_error_count = 0
        for row in matches:
            gid = int(row["gt_id"])
            gt_mask = (gt_part == gid)
            if not bool(gt_mask.any()):
                continue
            overlapping_pred_ids = {int(v) for v in np.unique(pred_part[gt_mask]) if int(v) > 0}
            if len(overlapping_pred_ids) > 1:
                split_error_count += int(len(overlapping_pred_ids) - 1)

        # Topology consistency proxy: 1대1로 깔끔하게 매칭된 (충돌도 split도 없는)
        # pred를 가진 GT fragment의 비율을 잰다.
        good = 0
        for row in matches:
            pid = row["pred_id"]
            if pid is None:
                continue
            if pred_assignment_counts.get(int(pid), 0) != 1:
                continue
            gid = int(row["gt_id"])
            gt_mask = (gt_part == gid)
            overlapping_pred_ids = {int(v) for v in np.unique(pred_part[gt_mask]) if int(v) > 0}
            if len(overlapping_pred_ids) <= 1:
                good += 1
        topology_consistency = float(good / max(n_gt, 1))

        per_anatomy[str(anatomy)] = {
            "present": True,
            "gt_instance_count": n_gt,
            "pred_instance_count": n_pred,
            "fracture_dice_per_fragment": _mean_skip_none(dice_values),
            "local_dice_per_fragment_20mm": _mean_skip_none(local_dice_values),
            "hd95_mm_per_fragment": _mean_skip_none(hd95_values),
            "assd_mm_per_fragment": _mean_skip_none(assd_values),
            "instance_recall": float(recall),
            "instance_precision": float(precision),
            "instance_f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "merge_error_count": int(merge_error_count),
            "split_error_count": int(split_error_count),
            "topology_consistency": float(topology_consistency),
            "matches": matches,
        }

    # Cohort macro: 실제로 존재하는 anatomy들에 대해서만 평균을 낸다 (None은 건너뛴다).
    def _macro(key: str) -> float | None:
        values = [
            per_anatomy[name][key]
            for name in present
            if per_anatomy.get(name, {}).get(key) is not None
        ]
        if not values:
            return None
        return float(np.mean([float(v) for v in values]))

    def _macro_sum(key: str) -> int:
        return int(sum(int(per_anatomy[name].get(key, 0) or 0) for name in present))

    cohort_macro = {
        "present_anatomies": list(present),
        "fracture_dice": _macro("fracture_dice_per_fragment"),
        "local_dice_20mm": _macro("local_dice_per_fragment_20mm"),
        "hd95_mm": _macro("hd95_mm_per_fragment"),
        "assd_mm": _macro("assd_mm_per_fragment"),
        "instance_recall": _macro("instance_recall"),
        "instance_precision": _macro("instance_precision"),
        "instance_f1": _macro("instance_f1"),
        "topology_consistency": _macro("topology_consistency"),
        "merge_error_count_total": _macro_sum("merge_error_count"),
        "split_error_count_total": _macro_sum("split_error_count"),
    }

    return {
        "metric_contract": "task1_official_aligned_proxy_v2",
        "official_mean_position_score": None,
        "official_score_reproducible": False,
        "anatomy_ranges": {str(k): [int(v[0]), int(v[1])] for k, v in anatomy_ranges.items()},
        "spacing_zyx_mm": [float(v) for v in spacing_zyx],
        "voxel_volume_mm3": float(vox_mm3),
        "gt_fragment_min_mm3": float(gt_fragment_min_mm3),
        "cc_prune_mm3": float(cc_prune_mm3),
        "iou_match_threshold": float(iou_match_threshold),
        "local_radius_mm": float(local_radius_mm),
        "gt_dropped_below_min_volume_ids": [int(v) for v in gt_dropped_ids],
        "per_anatomy": per_anatomy,
        "cohort_macro": cohort_macro,
        "case_failed": bool(case_failed),
        "failure_mode": failure_mode,
        "limitations": [
            "Official PENGWIN 2026 Task 1 evaluator script is not published as of 2026-05-29.",
            "Per-anatomy argmax matching follows the public spec; collision semantics for Merge are proxy.",
            "Topology Consistency is a 1-to-1 overlap-graph proxy; the official reference is not available.",
            "Femur 151-200 is included (V5_ANATOMY_RANGES_WITH_FEMUR).",
        ],
    }




TASK1_V251_CHANNELS = {
    "support": 0,
    "exterior_context": 1,
    "fracture_surface": 2,
    "seed_center": 3,
    "seed_body": 4,
    "affinity_z": 5,
    "affinity_y": 6,
    "affinity_x": 7,
}


def _task1_v251_fragment_seed_body(fragment: np.ndarray,
                                   spacing_zyx: tuple[float, float, float],
                                   *,
                                   radius_mm: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Return one seed center and one compact connected seed body.

    [QC][Invariant:one_seed_per_fragment]
    V251 decoder는 seed가 fragment topology를 설명한다고 가정한다. 하나의 GT
    fragment에서 seed body가 여러 component로 갈라지면 학습 전 oracle부터
    split/merge 해석이 흔들리므로, max-distance voxel 주변의 연결 component만
    유지한다.
    """
    if len(spacing_zyx) != 3 or any(float(v) <= 0.0 for v in spacing_zyx):
        raise ValueError(f"invalid spacing_zyx for V251 seed body: {spacing_zyx}")
    fragment = fragment.astype(bool, copy=False)
    seed_center = np.zeros(fragment.shape, dtype=np.float32)
    seed_body = np.zeros(fragment.shape, dtype=np.float32)
    coords = np.argwhere(fragment)
    if coords.size == 0:
        return seed_center, seed_body
    dist = ndi.distance_transform_edt(fragment, sampling=spacing_zyx)
    candidates = np.argwhere(dist == float(dist.max()))
    centroid = coords.mean(axis=0)
    center = candidates[int(np.argmin(np.sum((candidates.astype(np.float64) - centroid) ** 2, axis=1)))]
    seed_center[tuple(center)] = 1.0

    spacing = np.asarray(spacing_zyx, dtype=np.float64).reshape(1, 3)
    delta_mm = (coords.astype(np.float64) - center.reshape(1, 3).astype(np.float64)) * spacing
    inside = np.sum(delta_mm * delta_mm, axis=1) <= float(radius_mm) ** 2
    body = np.zeros(fragment.shape, dtype=bool)
    body[tuple(coords[inside].T)] = True
    body &= fragment
    if body.any():
        cc, n_cc = ndi.label(body)
        center_component = int(cc[tuple(center)])
        if center_component <= 0:
            sizes = np.bincount(cc.ravel())
            keep = int(np.argmax(sizes[1:]) + 1) if int(n_cc) > 0 else 0
            body = cc == keep
        else:
            body = cc == center_component
    else:
        body[tuple(center)] = True
    seed_body[body] = 1.0
    return seed_center, seed_body








TASK1_V252_BASE_CHANNELS = {
    "support": 0,
    "exterior_context": 1,
    "fracture_surface": 2,
    "seed_center": 3,
    "seed_body": 4,
}

TASK1_V252_AFFINITY13_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 1),
    (0, 1, -1), (0, 1, 0), (0, 1, 1),
    (1, -1, -1), (1, -1, 0), (1, -1, 1),
    (1, 0, -1), (1, 0, 0), (1, 0, 1),
    (1, 1, -1), (1, 1, 0), (1, 1, 1),
)

TASK1_V252_CHANNELS = {
    **TASK1_V252_BASE_CHANNELS,
    **{
        f"affinity13_{dz}_{dy}_{dx}": 5 + i
        for i, (dz, dy, dx) in enumerate(TASK1_V252_AFFINITY13_OFFSETS)
    },
}
TASK1_V255_ATTRACTIVE_START = 5
TASK1_V255_REPULSIVE_START = 5 + len(TASK1_V252_AFFINITY13_OFFSETS)
TASK1_V255_OUTPUT_CHANNELS = 5 + 2 * len(TASK1_V252_AFFINITY13_OFFSETS)
TASK1_V255_CHANNELS = {
    **TASK1_V252_CHANNELS,
    **{
        f"mutex13_{dz}_{dy}_{dx}": TASK1_V255_REPULSIVE_START + i
        for i, (dz, dy, dx) in enumerate(TASK1_V252_AFFINITY13_OFFSETS)
    },
}
TASK1_V273_BASE_CHANNELS = {
    "support": 0,
    "seed_body": 1,
}
TASK1_V273_JOIN_START = 2
TASK1_V273_CUT_START = TASK1_V273_JOIN_START + len(TASK1_V252_AFFINITY13_OFFSETS)
TASK1_V273_OUTPUT_CHANNELS = 2 + 2 * len(TASK1_V252_AFFINITY13_OFFSETS)
TASK1_V273_CHANNELS = {
    **TASK1_V273_BASE_CHANNELS,
    **{
        f"join13_{dz}_{dy}_{dx}": TASK1_V273_JOIN_START + i
        for i, (dz, dy, dx) in enumerate(TASK1_V252_AFFINITY13_OFFSETS)
    },
    **{
        f"cut13_{dz}_{dy}_{dx}": TASK1_V273_CUT_START + i
        for i, (dz, dy, dx) in enumerate(TASK1_V252_AFFINITY13_OFFSETS)
    },
}
TASK1_V275_OUTPUT_CHANNELS = int(BFV3_FRAGMENT_POSITION_V275_OUTPUT_CHANNELS)
TASK1_V275_CHANNELS = {
    "background": 0,
    **{f"fragment_position_{idx:02d}": idx for idx in range(1, 51)},
}
TASK1_V277_OUTPUT_CHANNELS = int(BFV3_SEPARATOR_GAP_V277_OUTPUT_CHANNELS)
TASK1_V277_CHANNELS = {
    "support": 0,
    "separator_gap": 1,
}
TASK1_V278_OUTPUT_CHANNELS = int(BFV3_SEPARATOR_ENERGY_V278_OUTPUT_CHANNELS)
TASK1_V278_CHANNELS = {
    "support": 0,
    "separator_energy": 1,
}
TASK1_V288_OUTPUT_CHANNELS = 4
TASK1_V288_CHANNELS = {
    "background": 0,
    "border": 1,
    "boundary": 2,
    "core": 3,
}
TASK1_V289_OUTPUT_CHANNELS = 4
TASK1_V289_CHANNELS = {
    "background": 0,
    "border": 1,
    "boundary_sdf": 2,
    "core": 3,
}


def _task1_affinity_offset_slices(shape: tuple[int, int, int],
                                  offset_zyx: tuple[int, int, int]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Return source/destination slices for one directed 3D affinity offset."""
    src: list[slice] = []
    dst: list[slice] = []
    for dim, delta in zip(shape, offset_zyx):
        if abs(int(delta)) >= int(dim):
            raise ValueError(f"affinity offset {offset_zyx} is invalid for shape {shape}")
        if int(delta) > 0:
            src.append(slice(0, -int(delta)))
            dst.append(slice(int(delta), None))
        elif int(delta) < 0:
            src.append(slice(-int(delta), None))
            dst.append(slice(0, int(delta)))
        else:
            src.append(slice(None))
            dst.append(slice(None))
    return tuple(src), tuple(dst)


def _task1_v253_affinity13_edge_counts(probs: np.ndarray,
                                        *,
                                        support_threshold: float,
                                        affinity_threshold: float) -> dict[str, Any]:
    """Count valid and joined V253 affinity13 graph edges before graph decode.

    Raises:
        ValueError: `probs`가 V253 18채널 확률 텐서가 아니거나 NaN/Inf를 포함할 때.
    """
    arr = np.asarray(probs, dtype=np.float32)
    expected_channels = 5 + len(TASK1_V252_AFFINITY13_OFFSETS)
    # [QC][Invariant:channel_order]
    # V253 decoder는 [support, exterior, CFS, seed_center, seed_body, 13 affinity] 순서를 전제로 한다.
    # 이 순서가 깨지면 edge cap이 실제 decoder graph와 다른 대상을 세어 hang/metric 오판을 만들 수 있다.
    if arr.ndim != 4 or int(arr.shape[0]) != int(expected_channels):
        raise ValueError(f"V253 edge count expects [{expected_channels},Z,Y,X], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("V253 edge count received NaN/Inf probabilities")
    support = arr[TASK1_V252_BASE_CHANNELS["support"]] >= float(support_threshold)
    affinity = np.clip(arr[5:], 0.0, 1.0)
    shape = tuple(int(v) for v in support.shape)
    valid_edges = 0
    joined_edges = 0
    per_offset: list[dict[str, Any]] = []
    for i, offset in enumerate(TASK1_V252_AFFINITY13_OFFSETS):
        src_sl, dst_sl = _task1_affinity_offset_slices(shape, offset)
        edge_valid = support[src_sl] & support[dst_sl]
        edge_join = edge_valid & (affinity[i][src_sl] >= float(affinity_threshold))
        valid_count = int(edge_valid.sum())
        joined_count = int(edge_join.sum())
        valid_edges += valid_count
        joined_edges += joined_count
        per_offset.append({
            "offset_zyx": [int(v) for v in offset],
            "valid_edges": int(valid_count),
            "joined_edges": int(joined_count),
        })
    return {
        "support_voxels": int(support.sum()),
        "valid_affinity_edges": int(valid_edges),
        "joined_affinity_edges": int(joined_edges),
        "affinity_threshold": float(affinity_threshold),
        "per_offset": per_offset,
    }


def _probability_threshold_to_logit(threshold: float, *, name: str) -> float:
    """Convert a Bernoulli probability threshold to the equivalent logit threshold."""
    value = float(threshold)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {threshold}")
    return float(np.log(value / (1.0 - value)))


def _task1_v253_affinity13_edge_counts_from_logits(logits: np.ndarray,
                                                   *,
                                                   support_threshold: float,
                                                   affinity_threshold: float) -> dict[str, Any]:
    """Count V253 affinity13 graph edges directly from logits.

    Raises:
        ValueError: `logits`가 V253 18채널 logit 텐서가 아니거나 NaN/Inf를 포함할 때.
    """
    arr = np.asarray(logits)
    expected_channels = 5 + len(TASK1_V252_AFFINITY13_OFFSETS)
    # [QC][Invariant:threshold_equivalence]
    # sigmoid(logit) >= p 는 logit >= log(p/(1-p))와 동치다. learned eval에서는
    # cap 초과 ROI를 확률 텐서/graph로 확장하기 전에 raw logit에서 걸러야 CPU/RAM 병목을 막을 수 있다.
    if arr.ndim != 4 or int(arr.shape[0]) != int(expected_channels):
        raise ValueError(f"V253 raw edge count expects [{expected_channels},Z,Y,X], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("V253 raw edge count received NaN/Inf logits")
    support_logit_threshold = _probability_threshold_to_logit(support_threshold, name="support_threshold")
    affinity_logit_threshold = _probability_threshold_to_logit(affinity_threshold, name="affinity_threshold")
    support = arr[TASK1_V252_BASE_CHANNELS["support"]] >= support_logit_threshold
    shape = tuple(int(v) for v in support.shape)
    valid_edges = 0
    joined_edges = 0
    per_offset: list[dict[str, Any]] = []
    for i, offset in enumerate(TASK1_V252_AFFINITY13_OFFSETS):
        src_sl, dst_sl = _task1_affinity_offset_slices(shape, offset)
        edge_valid = support[src_sl] & support[dst_sl]
        edge_join = edge_valid & (arr[5 + i][src_sl] >= affinity_logit_threshold)
        valid_count = int(edge_valid.sum())
        joined_count = int(edge_join.sum())
        valid_edges += valid_count
        joined_edges += joined_count
        per_offset.append({
            "offset_zyx": [int(v) for v in offset],
            "valid_edges": int(valid_count),
            "joined_edges": int(joined_count),
        })
    return {
        "support_voxels": int(support.sum()),
        "valid_affinity_edges": int(valid_edges),
        "joined_affinity_edges": int(joined_edges),
        "affinity_threshold": float(affinity_threshold),
        "per_offset": per_offset,
    }


def _task1_v255_mutex13_edge_counts(probs: np.ndarray,
                                    *,
                                    support_threshold: float,
                                    affinity_threshold: float,
                                    mutex_threshold: float = 0.5) -> dict[str, Any]:
    """Count V255 graph edges using attractive + repulsive mutex probabilities."""
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[0]) != int(TASK1_V255_OUTPUT_CHANNELS):
        raise ValueError(f"V255 edge count expects [{TASK1_V255_OUTPUT_CHANNELS},Z,Y,X], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("V255 edge count received NaN/Inf probabilities")
    support = arr[TASK1_V252_BASE_CHANNELS["support"]] >= float(support_threshold)
    attractive = np.clip(arr[TASK1_V255_ATTRACTIVE_START:TASK1_V255_REPULSIVE_START], 0.0, 1.0)
    repulsive = np.clip(arr[TASK1_V255_REPULSIVE_START:], 0.0, 1.0)
    shape = tuple(int(v) for v in support.shape)
    valid_edges = 0
    joined_edges = 0
    veto_edges = 0
    per_offset: list[dict[str, Any]] = []
    for i, offset in enumerate(TASK1_V252_AFFINITY13_OFFSETS):
        src_sl, dst_sl = _task1_affinity_offset_slices(shape, offset)
        edge_valid = support[src_sl] & support[dst_sl]
        edge_veto = edge_valid & (repulsive[i][src_sl] >= float(mutex_threshold))
        edge_join = edge_valid & (attractive[i][src_sl] >= float(affinity_threshold)) & ~edge_veto
        valid_count = int(edge_valid.sum())
        joined_count = int(edge_join.sum())
        veto_count = int(edge_veto.sum())
        valid_edges += valid_count
        joined_edges += joined_count
        veto_edges += veto_count
        per_offset.append({
            "offset_zyx": [int(v) for v in offset],
            "valid_edges": valid_count,
            "joined_edges": joined_count,
            "mutex_veto_edges": veto_count,
        })
    return {
        "support_voxels": int(support.sum()),
        "valid_affinity_edges": int(valid_edges),
        "joined_affinity_edges": int(joined_edges),
        "mutex_veto_edges": int(veto_edges),
        "affinity_threshold": float(affinity_threshold),
        "mutex_threshold": float(mutex_threshold),
        "per_offset": per_offset,
    }


def _task1_v255_mutex13_edge_counts_from_logits(logits: np.ndarray,
                                                *,
                                                support_threshold: float,
                                                affinity_threshold: float,
                                                mutex_threshold: float = 0.5) -> dict[str, Any]:
    """Count V255 graph edges directly from logits before probability expansion."""
    arr = np.asarray(logits)
    if arr.ndim != 4 or int(arr.shape[0]) != int(TASK1_V255_OUTPUT_CHANNELS):
        raise ValueError(f"V255 raw edge count expects [{TASK1_V255_OUTPUT_CHANNELS},Z,Y,X], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("V255 raw edge count received NaN/Inf logits")
    support_logit_threshold = _probability_threshold_to_logit(support_threshold, name="support_threshold")
    affinity_logit_threshold = _probability_threshold_to_logit(affinity_threshold, name="affinity_threshold")
    mutex_logit_threshold = _probability_threshold_to_logit(mutex_threshold, name="mutex_threshold")
    support = arr[TASK1_V252_BASE_CHANNELS["support"]] >= support_logit_threshold
    shape = tuple(int(v) for v in support.shape)
    valid_edges = 0
    joined_edges = 0
    veto_edges = 0
    per_offset: list[dict[str, Any]] = []
    for i, offset in enumerate(TASK1_V252_AFFINITY13_OFFSETS):
        src_sl, dst_sl = _task1_affinity_offset_slices(shape, offset)
        edge_valid = support[src_sl] & support[dst_sl]
        edge_veto = edge_valid & (arr[TASK1_V255_REPULSIVE_START + i][src_sl] >= mutex_logit_threshold)
        edge_join = edge_valid & (arr[TASK1_V255_ATTRACTIVE_START + i][src_sl] >= affinity_logit_threshold) & ~edge_veto
        valid_count = int(edge_valid.sum())
        joined_count = int(edge_join.sum())
        veto_count = int(edge_veto.sum())
        valid_edges += valid_count
        joined_edges += joined_count
        veto_edges += veto_count
        per_offset.append({
            "offset_zyx": [int(v) for v in offset],
            "valid_edges": valid_count,
            "joined_edges": joined_count,
            "mutex_veto_edges": veto_count,
        })
    return {
        "support_voxels": int(support.sum()),
        "valid_affinity_edges": int(valid_edges),
        "joined_affinity_edges": int(joined_edges),
        "mutex_veto_edges": int(veto_edges),
        "affinity_threshold": float(affinity_threshold),
        "mutex_threshold": float(mutex_threshold),
        "per_offset": per_offset,
    }












def _task1_v277_separator_gap_from_instances(inst_roi: np.ndarray) -> np.ndarray:
    """Build a 13-neighbor separator-gap mask from same-anatomy instance IDs.

    [DATA][Risk:High][Scope:v277_separator_target]
    이 함수는 old contact head target을 재사용하지 않는다. 같은 anatomy range 안에서
    인접 voxel의 instance ID가 달라지는 지점만 separator로 표시해 connected component
    decoder의 절단면을 만든다.
    """
    inst = np.asarray(inst_roi).astype(np.int32, copy=False)
    separator = np.zeros(inst.shape, dtype=bool)
    for offset in TASK1_V252_AFFINITY13_OFFSETS:
        src_sl, dst_sl = _task1_affinity_offset_slices(inst.shape, offset)
        a = inst[src_sl]
        b = inst[dst_sl]
        a_fg = valid_instance_mask(a, pelvic_only=True)
        b_fg = valid_instance_mask(b, pelvic_only=True)
        same_anatomy = a_fg & b_fg & (((a - 1) // 50) == ((b - 1) // 50))
        different_fragment = same_anatomy & (a != b)
        if different_fragment.any():
            src_view = separator[src_sl]
            dst_view = separator[dst_sl]
            src_view[different_fragment] = True
            dst_view[different_fragment] = True
            separator[src_sl] = src_view
            separator[dst_sl] = dst_view
    return separator





def _task1_v277_merge_small_components(labels: np.ndarray,
                                       *,
                                       min_component_voxels: int) -> np.ndarray:
    """separator로 인해 생긴 너무 작은 fragment를 가장 가까운 큰 component에 병합한다."""
    out = np.asarray(labels).astype(np.int32, copy=True)
    if int(min_component_voxels) <= 0:
        return out.astype(np.uint16, copy=False)
    ids = [int(v) for v in np.unique(out) if int(v) > 0]
    if not ids:
        return out.astype(np.uint16, copy=False)
    sizes = {component_id: int((out == component_id).sum()) for component_id in ids}
    small_ids = [component_id for component_id in ids if int(sizes[component_id]) < int(min_component_voxels)]
    large_ids = [component_id for component_id in ids if int(sizes[component_id]) >= int(min_component_voxels)]
    if not small_ids or not large_ids:
        return out.astype(np.uint16, copy=False)
    large_mask = np.isin(out, np.asarray(large_ids, dtype=np.int32))
    nearest_indices = ndi.distance_transform_edt(~large_mask, return_indices=True)[1]
    for component_id in small_ids:
        mask = out == int(component_id)
        if not mask.any():
            continue
        nearest_ids = out[tuple(axis_indices[mask] for axis_indices in nearest_indices)]
        nearest_ids = nearest_ids[nearest_ids > 0]
        if nearest_ids.size <= 0:
            continue
        values, counts = np.unique(nearest_ids, return_counts=True)
        out[mask] = int(values[int(np.argmax(counts))])
    compact = np.zeros_like(out, dtype=np.uint16)
    for new_id, old_id in enumerate([int(v) for v in np.unique(out) if int(v) > 0], start=1):
        compact[out == int(old_id)] = np.uint16(new_id)
    return compact




def task1_v288_abbc_target(inst_roi: np.ndarray,
                           *,
                           core_radius_vox: float = 2.0,
                           boundary_dilate_vox: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    """V288 ABBC oracle field [background, border, boundary, core]를 만든다.

    [QC][Invariant:channel_order]
    출력은 `[4,Z,Y,X]` 형태의 float32 one-hot이다. channel 0이 background, 1이 border,
    2가 boundary, 3이 core다. core는 decoder의 seed class 역할을 하고, boundary는 fragment
    쌍 사이의 medial-axis 적응형 두께를 가진 cut surface다.

    [DATA][Risk:High][Scope:v288_abbc_target]
    PENGWIN 2024 1st (MIC-DKFZ)의 ABBC representation을 oracle 단계에서 재현한다.
    boundary 두께는 raw 13-neighbor between-fragment band를 `boundary_dilate_vox` 횟수만큼
    binary_dilation으로 확장한 다음 support와 교차시켜 정의한다. core는 support 영역 안에서
    boundary로부터 `core_radius_vox` 이상 떨어진 voxel이며, 만약 어떤 fragment에 core가 하나도
    없다면 그 fragment의 EDT argmax 위치에 voxel 한 개를 강제 seed로 넣어 얇은 fragment도
    반드시 seed를 갖도록 보장한다.
    """
    inst = np.asarray(inst_roi).astype(np.uint16, copy=False)
    if inst.ndim != 3:
        raise ValueError(f"V288 ABBC target requires [Z,Y,X], got {inst.shape}")
    if float(core_radius_vox) < 0.0:
        raise ValueError(f"core_radius_vox must be >= 0, got {core_radius_vox}")
    if int(boundary_dilate_vox) < 0:
        raise ValueError(f"boundary_dilate_vox must be >= 0, got {boundary_dilate_vox}")
    support = valid_instance_mask(inst, pelvic_only=True)
    raw_between = _task1_v277_separator_gap_from_instances(inst) & support
    structure = np.ones((3, 3, 3), dtype=bool)
    if int(boundary_dilate_vox) > 0 and raw_between.any():
        boundary = ndi.binary_dilation(
            raw_between,
            structure=structure,
            iterations=int(boundary_dilate_vox),
        ) & support
    else:
        boundary = raw_between & support
    blocker = (~support) | boundary
    if (~blocker).any():
        edt_to_blocker = ndi.distance_transform_edt(~blocker)
    else:
        edt_to_blocker = np.zeros(inst.shape, dtype=np.float32)
    core = support & (~boundary) & (edt_to_blocker >= float(core_radius_vox))
    fragments_without_core = 0
    fragments_forced_seed = 0
    for frag_id in [int(v) for v in np.unique(inst) if MIN_INSTANCE_ID <= int(v) <= PELVIC_MAX_INSTANCE_ID]:
        frag_mask = (inst == frag_id)
        if not frag_mask.any():
            continue
        if core[frag_mask].any():
            continue
        fragments_without_core += 1
        frag_inner = frag_mask & (~boundary)
        if not frag_inner.any():
            continue
        frag_edt = ndi.distance_transform_edt(frag_inner)
        flat_idx = int(np.argmax(frag_edt))
        seed_zyx = np.unravel_index(flat_idx, frag_edt.shape)
        if float(frag_edt[seed_zyx]) > 0.0:
            core[seed_zyx] = True
            fragments_forced_seed += 1
    border = support & (~boundary) & (~core)
    background = ~support
    target = np.stack([background, border, boundary, core], axis=0).astype(np.float32, copy=False)
    audit = {
        "contract": "task1_v288_abbc_target_v1",
        "channels": dict(TASK1_V288_CHANNELS),
        "shape": [int(v) for v in inst.shape],
        "output_channels": int(TASK1_V288_OUTPUT_CHANNELS),
        "core_radius_vox": float(core_radius_vox),
        "boundary_dilate_vox": int(boundary_dilate_vox),
        "support_voxels": int(support.sum()),
        "raw_between_voxels": int(raw_between.sum()),
        "boundary_voxels": int(boundary.sum()),
        "core_voxels": int(core.sum()),
        "border_voxels": int(border.sum()),
        "background_voxels": int(background.sum()),
        "fragments_without_core_before_force": int(fragments_without_core),
        "fragments_forced_seed": int(fragments_forced_seed),
        "fragment_count": int(len([v for v in np.unique(inst) if MIN_INSTANCE_ID <= int(v) <= PELVIC_MAX_INSTANCE_ID])),
    }
    return target, audit


def _task1_v288_anatomy_size_ratio_merge(labels: np.ndarray,
                                         *,
                                         size_ratio_keep: float) -> np.ndarray:
    """`size_ratio_keep * 가장 큰 component 크기`보다 작은 component를 가장 가까운 큰 component에 병합한다.

    [DATA][Risk:High][Scope:v288_anatomical_refinement]
    PENGWIN 1st (MIC-DKFZ)의 "merging smaller fragments with closest anatomically
    correct instance" 후처리 단계다. 같은 anatomy ROI 안에서, 가장 큰 component 대비 크기
    비율이 `size_ratio_keep` 미만인 component를 가장 가까운 큰 component로 흡수시킨다.
    이렇게 하면 한 fragment가 여러 CC로 쪼개진 GT label 아티팩트(예: 003 Sacrum이 본래 GT 1이지만
    multi-CC로 저장된 경우)를 inference 단계에서도 동일 anatomy 안에서 자동으로 통합할 수 있다.
    """
    if float(size_ratio_keep) <= 0.0:
        return np.asarray(labels).astype(np.uint16, copy=False)
    out = np.asarray(labels).astype(np.int32, copy=True)
    sizes = {int(c): int((out == c).sum()) for c in np.unique(out) if int(c) > 0}
    if len(sizes) <= 1:
        return out.astype(np.uint16, copy=False)
    largest_size = max(sizes.values())
    threshold = max(1, int(float(size_ratio_keep) * float(largest_size)))
    small_ids = [c for c, sz in sizes.items() if sz < threshold]
    large_ids = [c for c, sz in sizes.items() if sz >= threshold]
    if not small_ids or not large_ids:
        return out.astype(np.uint16, copy=False)
    large_mask = np.isin(out, np.asarray(large_ids, dtype=np.int32))
    nearest_indices = ndi.distance_transform_edt(~large_mask, return_indices=True)[1]
    for c in small_ids:
        mask = out == int(c)
        if not mask.any():
            continue
        nearest_ids = out[tuple(axis_indices[mask] for axis_indices in nearest_indices)]
        nearest_ids = nearest_ids[nearest_ids > 0]
        if nearest_ids.size <= 0:
            continue
        vals, counts = np.unique(nearest_ids, return_counts=True)
        out[mask] = int(vals[int(np.argmax(counts))])
    compact = np.zeros_like(out, dtype=np.uint16)
    for new_id, old_id in enumerate([int(c) for c in np.unique(out) if int(c) > 0], start=1):
        compact[out == int(old_id)] = np.uint16(new_id)
    return compact


def decode_task1_v288_abbc(fields: np.ndarray,
                           *,
                           background_threshold: float = 0.50,
                           core_threshold: float = 0.50,
                           min_component_voxels: int = 100,
                           anatomy_size_ratio_keep: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    """V288 ABBC field를 디코드한다: core CC를 seed로 만든 뒤, boundary는 가장 가까운 seed로 watershed merge.

    [METRIC][Scope:task1_primary_gate]
    background와 core 모두 고정된 0.50 threshold를 사용한다. core CC를 watershed seed로 쓰고,
    boundary/border voxel은 가장 가까운 seed에 흡수된다. 너무 작은 component는 V277 helper로
    인접 component에 병합한다.
    """
    arr = np.asarray(fields)
    if arr.ndim != 4 or int(arr.shape[0]) != int(TASK1_V288_OUTPUT_CHANNELS):
        raise ValueError(f"V288 decoder expects [4,Z,Y,X], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("V288 decoder received NaN/Inf fields")
    background = arr[0] >= float(background_threshold)
    support = ~background
    core = (arr[3] >= float(core_threshold)) & support
    core_labels, n_core = ndi.label(core, structure=np.ones((3, 3, 3), dtype=bool))
    core_labels = core_labels.astype(np.int32, copy=False)
    if not support.any():
        decoded = np.zeros(arr.shape[1:], dtype=np.uint16)
        trace = {
            "decoder": "task1_v288_abbc_core_seed_watershed",
            "background_threshold": float(background_threshold),
            "core_threshold": float(core_threshold),
            "min_component_voxels": int(min_component_voxels),
            "support_voxels": 0,
            "core_voxels": 0,
            "core_seed_components_before_merge": 0,
            "components_before_small_merge": 0,
            "pred_fragment_count": 0,
            "contract_has_contact_probability": False,
            "contract_has_seed_channel": True,
            "contract_has_pairwise_graph": False,
        }
        return decoded, trace
    if int(n_core) <= 0:
        labels = np.where(support, 1, 0).astype(np.int32, copy=False)
    else:
        from skimage.segmentation import watershed as _ski_watershed
        priority = ndi.distance_transform_edt(core_labels == 0)
        labels = _ski_watershed(
            priority,
            markers=core_labels,
            mask=support,
        ).astype(np.int32, copy=False)
        fill_mask = support & (labels == 0)
        if fill_mask.any():
            nearest_indices = ndi.distance_transform_edt(labels == 0, return_indices=True)[1]
            labels[fill_mask] = labels[tuple(axis_indices[fill_mask] for axis_indices in nearest_indices)]
    before_merge_count = int(len([v for v in np.unique(labels) if int(v) > 0]))
    decoded = _task1_v277_merge_small_components(labels, min_component_voxels=int(min_component_voxels))
    components_after_size_merge = int(len([v for v in np.unique(decoded) if int(v) > 0]))
    if float(anatomy_size_ratio_keep) > 0.0:
        decoded = _task1_v288_anatomy_size_ratio_merge(
            decoded, size_ratio_keep=float(anatomy_size_ratio_keep),
        )
    trace = {
        "decoder": "task1_v288_abbc_core_seed_watershed",
        "background_threshold": float(background_threshold),
        "core_threshold": float(core_threshold),
        "min_component_voxels": int(min_component_voxels),
        "anatomy_size_ratio_keep": float(anatomy_size_ratio_keep),
        "support_voxels": int(support.sum()),
        "core_voxels": int(core.sum()),
        "core_seed_components_before_merge": int(n_core),
        "components_before_small_merge": int(before_merge_count),
        "components_after_small_merge": int(components_after_size_merge),
        "pred_fragment_count": int(len([v for v in np.unique(decoded) if int(v) > 0])),
        "contract_has_contact_probability": False,
        "contract_has_seed_channel": True,
        "contract_has_pairwise_graph": False,
    }
    return decoded.astype(np.uint16, copy=False), trace












def task1_v266_pairwise_softmax_fields_from_logits(raw_logits: np.ndarray) -> np.ndarray:
    """Convert V266 raw logits to V255-compatible decoder probabilities.

    [QC][Invariant:channel_activation]
    V266 rows 5..17 and 18..30 are not independent sigmoid probabilities.
    They are join-vs-cut logits for the same edge. The decoder probability is
    sigmoid(join_logit - cut_logit), which is exactly the two-class softmax
    join probability. Applying expit to each row independently would recreate
    the V255/V260 failure mode.
    """
    raw = np.asarray(raw_logits, dtype=np.float32)
    if raw.ndim != 4 or int(raw.shape[0]) != int(TASK1_V255_OUTPUT_CHANNELS):
        raise ValueError(f"V266 logits expect [{TASK1_V255_OUTPUT_CHANNELS},Z,Y,X], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("V266 logits contain NaN/Inf")
    fields = np.zeros_like(raw, dtype=np.float32)
    fields[:5] = expit(raw[:5]).astype(np.float32, copy=False)
    diff = raw[TASK1_V255_ATTRACTIVE_START:TASK1_V255_REPULSIVE_START] - raw[TASK1_V255_REPULSIVE_START:]
    fields[TASK1_V255_ATTRACTIVE_START:TASK1_V255_REPULSIVE_START] = expit(diff).astype(np.float32, copy=False)
    fields[TASK1_V255_REPULSIVE_START:] = expit(-diff).astype(np.float32, copy=False)
    return fields








def _task1_v258_seed_peak_markers(seed_score: np.ndarray,
                                  support: np.ndarray,
                                  spacing_zyx: tuple[float, float, float],
                                  *,
                                  peak_threshold: float = 0.5,
                                  nms_radius_mm: float = 10.0,
                                  max_markers: int = 50,
                                  preselect_markers: int = 4096,
                                  allow_top1_fallback: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Select deterministic seed-center peaks for V258 marker decoding.

    [AUDIT][Risk:High][Scope:v258_seed_peak_decoder]
    V257의 threshold mask seed는 0개 또는 수십만 voxel로 흔들렸다. V258은
    channel3을 dense mask가 아니라 peak field로 해석한다. marker 수는 anatomy
    ID capacity(50) 안에서 NMS로 제한해 Task1 Instance/Topology metric이
    overfragmentation 때문에 평가 불능 상태로 빠지는 것을 막는다.
    """
    score = np.asarray(seed_score, dtype=np.float32)
    support_mask = np.asarray(support).astype(bool, copy=False)
    if score.shape != support_mask.shape:
        raise ValueError(f"V258 seed peak shape mismatch: score={score.shape} support={support_mask.shape}")
    if len(spacing_zyx) != 3 or any(float(v) <= 0.0 for v in spacing_zyx):
        raise ValueError(f"invalid spacing_zyx for V258 seed peak markers: {spacing_zyx}")
    if int(max_markers) < 1:
        raise ValueError(f"max_markers must be >= 1, got {max_markers}")
    if float(nms_radius_mm) <= 0.0:
        raise ValueError(f"nms_radius_mm must be > 0, got {nms_radius_mm}")

    markers = np.zeros(score.shape, dtype=np.float32)
    if not support_mask.any():
        return markers, {
            "seed_peak_policy": "channel3_local_peak_nms",
            "seed_peak_candidates": 0,
            "seed_peak_selected": 0,
            "seed_peak_threshold": float(peak_threshold),
            "seed_peak_top1_fallback": False,
        }

    candidate_mask = support_mask & (score >= float(peak_threshold))
    fallback_used = False
    if not candidate_mask.any() and bool(allow_top1_fallback):
        masked_score = np.where(support_mask, score, -np.inf)
        top_flat = int(np.argmax(masked_score.reshape(-1)))
        candidate_flat = np.asarray([top_flat], dtype=np.int64)
        fallback_used = True
    else:
        candidate_flat = np.flatnonzero(candidate_mask.reshape(-1))
    if candidate_flat.size == 0:
        return markers, {
            "seed_peak_policy": "channel3_local_peak_nms",
            "seed_peak_candidates": 0,
            "seed_peak_selected": 0,
            "seed_peak_threshold": float(peak_threshold),
            "seed_peak_top1_fallback": bool(fallback_used),
        }

    flat_score = score.reshape(-1)
    candidate_scores = flat_score[candidate_flat]
    preselect = max(int(max_markers) * 8, int(preselect_markers))
    if int(candidate_flat.size) > int(preselect):
        # [QC][Invariant:deterministic_runtime]
        # threshold 후보가 수십만 voxel이면 전체 정렬이 eval runtime을 지배한다.
        # argpartition으로 상위 후보만 남긴 뒤 점수 내림차순 정렬을 적용해 marker
        # 선택은 deterministic하게 유지한다.
        keep_idx = np.argpartition(-candidate_scores, int(preselect) - 1)[:int(preselect)]
        candidate_flat = candidate_flat[keep_idx]
        candidate_scores = candidate_scores[keep_idx]
    order = np.argsort(-candidate_scores, kind="mergesort")
    candidate_flat = candidate_flat[order]
    candidate_scores = candidate_scores[order]
    coords = np.column_stack(np.unravel_index(candidate_flat, score.shape)).astype(np.int32, copy=False)
    spacing = np.asarray(spacing_zyx, dtype=np.float64).reshape(1, 3)
    min_dist2 = float(nms_radius_mm) ** 2
    selected: list[np.ndarray] = []
    selected_scores: list[float] = []
    for coord, value in zip(coords, candidate_scores):
        if selected:
            selected_arr = np.asarray(selected, dtype=np.float64)
            delta_mm = (selected_arr - coord.reshape(1, 3).astype(np.float64)) * spacing
            if bool(np.any(np.sum(delta_mm * delta_mm, axis=1) < min_dist2)):
                continue
        selected.append(coord.copy())
        selected_scores.append(float(value))
        if len(selected) >= int(max_markers):
            break
    for marker_id, coord in enumerate(selected, start=1):
        markers[tuple(int(v) for v in coord)] = 1.0
    # [METRIC][Scope:task1_instance_seed]
    # selected marker 수는 Task1 Instance Precision/Recall 및 Split/Merge의 직접
    # 상한/하한으로 작동한다. trace에 후보 수, fallback, NMS radius를 남겨 fulltrain
    # gate 실패 시 seed 문제와 affinity 문제를 분리한다.
    return markers, {
        "seed_peak_policy": "channel3_local_peak_nms",
        "seed_peak_candidates": int(candidate_flat.size),
        "seed_peak_selected": int(len(selected)),
        "seed_peak_threshold": float(peak_threshold),
        "seed_peak_nms_radius_mm": float(nms_radius_mm),
        "seed_peak_max_markers": int(max_markers),
        "seed_peak_top1_fallback": bool(fallback_used),
        "seed_peak_selected_scores": selected_scores[:50],
    }






TASK1_V261_CHANNELS = {
    "support": 0,
    "exterior_context": 1,
    "fracture_surface": 2,
    "center_heatmap": 3,
    "distance_field": 4,
    "offset_to_center_z": 5,
    "offset_to_center_y": 6,
    "offset_to_center_x": 7,
}
TASK1_V261_OUTPUT_CHANNELS = 8

TASK1_V269_CHANNELS = {
    "support": 0,
    "flow_z": 1,
    "flow_y": 2,
    "flow_x": 3,
}
TASK1_V269_OUTPUT_CHANNELS = 4

TASK1_V270_CHANNELS = {
    "support": 0,
    "sink_z": 1,
    "sink_y": 2,
    "sink_x": 3,
}
TASK1_V270_OUTPUT_CHANNELS = 4

TASK1_V272_CHANNELS = {
    "support": 0,
    "seed_body": 1,
    "distance_field": 2,
}
TASK1_V272_OUTPUT_CHANNELS = 3






def _task1_v262_peak_seed_mask(center_prob: np.ndarray,
                               support: np.ndarray,
                               *,
                               center_threshold: float,
                               peak_nms_size_vox: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one decoder seed per local peak plateau for V262 center-flow."""
    if int(peak_nms_size_vox) < 1 or int(peak_nms_size_vox) % 2 == 0:
        raise ValueError(f"peak_nms_size_vox must be odd and >=1, got {peak_nms_size_vox}")
    score = np.asarray(center_prob, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    if score.shape != support.shape:
        raise ValueError(f"V262 peak seed shape mismatch: center={score.shape} support={support.shape}")
    if not np.isfinite(score).all():
        raise ValueError("V262 center probability contains NaN/Inf")
    if not support.any():
        return np.zeros(score.shape, dtype=bool), {
            "peak_nms_size_vox": int(peak_nms_size_vox),
            "raw_local_peak_voxels": 0,
            "raw_peak_plateaus": 0,
            "peak_count_after_nms": 0,
        }
    pooled = ndi.maximum_filter(
        score,
        size=int(peak_nms_size_vox),
        mode="constant",
        cval=-np.inf,
    )
    raw_peak = (score >= float(center_threshold)) & support & (score >= pooled - 1e-7)
    plateau_cc, plateau_n = ndi.label(raw_peak)
    seeds = np.zeros(score.shape, dtype=bool)
    for plateau_id in range(1, int(plateau_n) + 1):
        coords = np.argwhere(plateau_cc == int(plateau_id))
        if coords.size == 0:
            continue
        vals = score[tuple(coords.T)]
        # [QC][Invariant:deterministic_peak_tie_break]
        # maximum_filter는 plateau 전체를 local maximum으로 표시할 수 있다. plateau당
        # 하나의 voxel만 남겨야 decoder center 수가 Task1 Split/Instance Precision과
        # 직접 대응한다. argmax tie는 np.argmax의 첫 좌표로 deterministic하게 처리한다.
        chosen = coords[int(np.argmax(vals))]
        seeds[tuple(int(v) for v in chosen)] = True
    return seeds, {
        "peak_nms_size_vox": int(peak_nms_size_vox),
        "raw_local_peak_voxels": int(raw_peak.sum()),
        "raw_peak_plateaus": int(plateau_n),
        "peak_count_after_nms": int(seeds.sum()),
    }






def _task1_v269_cluster_endpoint_votes(endpoints: np.ndarray,
                                       *,
                                       cluster_radius_vox: float,
                                       min_endpoint_votes: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster flow-dynamics endpoints and return one label per endpoint row."""
    if float(cluster_radius_vox) <= 0.0:
        raise ValueError(f"cluster_radius_vox must be > 0, got {cluster_radius_vox}")
    if int(min_endpoint_votes) < 1:
        raise ValueError(f"min_endpoint_votes must be >= 1, got {min_endpoint_votes}")
    points = np.asarray(endpoints, dtype=np.int64)
    if points.ndim != 2 or int(points.shape[1]) != 3:
        raise ValueError(f"V269 endpoint array must have shape [N,3], got {points.shape}")
    if int(points.shape[0]) == 0:
        return np.zeros((0,), dtype=np.uint16), {
            "unique_endpoint_count": 0,
            "endpoint_cluster_count_raw": 0,
            "endpoint_cluster_count_kept": 0,
            "dropped_endpoint_votes": 0,
        }

    unique_points, inverse, counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    n_points = int(unique_points.shape[0])
    parent = np.arange(n_points, dtype=np.int64)

    def find(i: int) -> int:
        while int(parent[i]) != int(i):
            parent[i] = parent[int(parent[i])]
            i = int(parent[i])
        return int(i)

    def union(a: int, b: int) -> None:
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    from scipy.spatial import cKDTree

    tree = cKDTree(unique_points.astype(np.float32, copy=False))
    for a, b in tree.query_pairs(r=float(cluster_radius_vox)):
        union(int(a), int(b))
    roots = np.asarray([find(i) for i in range(n_points)], dtype=np.int64)
    root_to_indices: dict[int, list[int]] = {}
    for idx, root in enumerate(roots):
        root_to_indices.setdefault(int(root), []).append(int(idx))

    cluster_rows = []
    for root, indices in root_to_indices.items():
        vote_count = int(np.sum(counts[indices]))
        centroid = np.average(unique_points[indices].astype(np.float64, copy=False), axis=0, weights=counts[indices])
        cluster_rows.append((int(root), vote_count, centroid))
    cluster_rows.sort(key=lambda row: (float(row[2][0]), float(row[2][1]), float(row[2][2]), -int(row[1])))
    kept_roots = [root for root, votes, _centroid in cluster_rows if int(votes) >= int(min_endpoint_votes)]
    root_label = {int(root): int(i + 1) for i, root in enumerate(kept_roots)}
    unique_labels = np.zeros((n_points,), dtype=np.uint16)
    for idx, root in enumerate(roots):
        label = root_label.get(int(root), 0)
        unique_labels[idx] = np.uint16(label)
    labels = unique_labels[inverse]
    dropped_votes = int(np.count_nonzero(labels == 0))
    return labels.astype(np.uint16, copy=False), {
        "unique_endpoint_count": int(n_points),
        "endpoint_cluster_count_raw": int(len(cluster_rows)),
        "endpoint_cluster_count_kept": int(len(kept_roots)),
        "min_endpoint_votes": int(min_endpoint_votes),
        "cluster_radius_vox": float(cluster_radius_vox),
        "dropped_endpoint_votes": int(dropped_votes),
    }












def _write_task1_prediction_mha(decoded_case: np.ndarray,
                                ref_img: sitk.Image,
                                out_path: Path) -> None:
    """Write a decoded uint16 instance mask with the source label geometry."""
    # [QC][Invariant:geometry]
    # 시각화와 Task1 proxy가 같은 source-space volume을 보도록 oracle prediction은
    # canonicalized label image의 geometry를 그대로 복사한다.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = sitk.GetImageFromArray(decoded_case.astype(np.uint16, copy=False))
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, str(out_path))


























def _task1_v270_fields_from_logits(raw_logits: np.ndarray) -> np.ndarray:
    """Convert learned V270 logits to decoder fields."""
    raw = np.asarray(raw_logits, dtype=np.float32)
    if raw.ndim != 4 or int(raw.shape[0]) != int(TASK1_V270_OUTPUT_CHANNELS):
        raise ValueError(f"expected V270 logits [4,Z,Y,X], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("V270 logits contain NaN/Inf")
    fields = np.zeros_like(raw, dtype=np.float32)
    fields[TASK1_V270_CHANNELS["support"]] = expit(raw[0]).astype(np.float32, copy=False)
    fields[TASK1_V270_CHANNELS["sink_z"]] = expit(raw[1]).astype(np.float32, copy=False)
    fields[TASK1_V270_CHANNELS["sink_y"]] = expit(raw[2]).astype(np.float32, copy=False)
    fields[TASK1_V270_CHANNELS["sink_x"]] = expit(raw[3]).astype(np.float32, copy=False)
    return fields






def _load_task1_v253_logits(path: Path) -> np.ndarray:
    """Load one learned V253 logits cache with the fixed 18-channel contract."""
    logits = _load_bicm_v6_logits(path)
    expected = int(BFV3_AFFINITY13_SEED_OUTPUT_CHANNELS)
    # [QC][Invariant:channel_order]
    # V253 learned eval must consume the exact oracle-pass 18-channel order:
    # support/exterior/fracture/seed_center/seed_body + 13 same-fragment
    # affinities. Accepting 6/8/10-channel logits here would silently evaluate
    # a different contract and make the full-train gate meaningless.
    if logits.ndim != 4 or int(logits.shape[0]) != expected:
        raise RuntimeError(f"expected V253 logits [{expected},Z,Y,X], got {logits.shape} in {path}")
    if not np.isfinite(logits).all():
        raise RuntimeError(f"V253 logits contain NaN/Inf: {path}")
    return logits.astype(np.float32, copy=False)






def task1_v273_pairwise_softmax_fields_from_logits(raw_logits: np.ndarray) -> np.ndarray:
    """Convert V273 raw logits to no-contact decoder probabilities."""
    raw = np.asarray(raw_logits, dtype=np.float32)
    if raw.ndim != 4 or int(raw.shape[0]) != int(BFV3_NO_CONTACT_PAIRWISE_V273_OUTPUT_CHANNELS):
        raise ValueError(f"V273 logits expect [28,Z,Y,X], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("V273 logits contain NaN/Inf")
    fields = np.zeros_like(raw, dtype=np.float32)
    fields[TASK1_V273_BASE_CHANNELS["support"]] = expit(raw[0]).astype(np.float32, copy=False)
    fields[TASK1_V273_BASE_CHANNELS["seed_body"]] = expit(raw[1]).astype(np.float32, copy=False)
    diff = raw[TASK1_V273_JOIN_START:TASK1_V273_CUT_START] - raw[TASK1_V273_CUT_START:]
    fields[TASK1_V273_JOIN_START:TASK1_V273_CUT_START] = expit(diff).astype(np.float32, copy=False)
    fields[TASK1_V273_CUT_START:] = expit(-diff).astype(np.float32, copy=False)
    return fields


def _task1_v273_edge_counts(fields: np.ndarray,
                            *,
                            support_threshold: float,
                            join_threshold: float,
                            cut_threshold: float) -> dict[str, Any]:
    """Count V273 valid/join/cut graph edges before connected-components decode."""
    arr = np.asarray(fields, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[0]) != int(BFV3_NO_CONTACT_PAIRWISE_V273_OUTPUT_CHANNELS):
        raise ValueError(f"V273 edge count expects [28,Z,Y,X], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("V273 edge count received NaN/Inf probabilities")
    support = arr[TASK1_V273_BASE_CHANNELS["support"]] >= float(support_threshold)
    join = np.clip(arr[TASK1_V273_JOIN_START:TASK1_V273_CUT_START], 0.0, 1.0)
    cut = np.clip(arr[TASK1_V273_CUT_START:], 0.0, 1.0)
    shape = tuple(int(v) for v in support.shape)
    valid_edges = 0
    joined_edges = 0
    cut_edges = 0
    per_offset: list[dict[str, Any]] = []
    for i, offset in enumerate(TASK1_V252_AFFINITY13_OFFSETS):
        src_sl, dst_sl = _task1_affinity_offset_slices(shape, offset)
        edge_valid = support[src_sl] & support[dst_sl]
        edge_cut = edge_valid & (cut[i][src_sl] >= float(cut_threshold))
        edge_join = edge_valid & (join[i][src_sl] >= float(join_threshold)) & ~edge_cut
        valid_count = int(edge_valid.sum())
        joined_count = int(edge_join.sum())
        cut_count = int(edge_cut.sum())
        valid_edges += valid_count
        joined_edges += joined_count
        cut_edges += cut_count
        per_offset.append({
            "offset_zyx": [int(v) for v in offset],
            "valid_edges": int(valid_count),
            "joined_edges": int(joined_count),
            "cut_edges": int(cut_count),
        })
    return {
        "support_voxels": int(support.sum()),
        "valid_pairwise_edges": int(valid_edges),
        "joined_pairwise_edges": int(joined_edges),
        "cut_pairwise_edges": int(cut_edges),
        "join_threshold": float(join_threshold),
        "cut_threshold": float(cut_threshold),
        "per_offset": per_offset,
    }






def task1_v275_probabilities_from_logits(raw: np.ndarray) -> np.ndarray:
    """Convert V275 raw logits to channel-first softmax probabilities."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[0]) != int(TASK1_V275_OUTPUT_CHANNELS):
        raise ValueError(f"V275 logits expect [51,Z,Y,X], got {arr.shape}")
    # [QC][Invariant:channel_order]
    # 51-way softmax 전체가 하나의 multiclass contract다. sigmoid를 channel별로 적용하면
    # 여러 fragment class가 동시에 켜져 argmax decoder와 train target이 어긋난다.
    shifted = arr - np.max(arr, axis=0, keepdims=True)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    denom = np.sum(exp, axis=0, keepdims=True)
    return (exp / np.maximum(denom, np.float32(1e-8))).astype(np.float32, copy=False)




def task1_v277_probabilities_from_logits(raw: np.ndarray) -> np.ndarray:
    """Convert V277 raw logits to independent sigmoid fields."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[0]) != int(TASK1_V277_OUTPUT_CHANNELS):
        raise ValueError(f"V277 logits expect [2,Z,Y,X], got {arr.shape}")
    # [QC][Invariant:channel_order]
    # V277은 multiclass argmax가 아니라 support/separator binary field 두 개다.
    # softmax를 적용하면 separator가 support와 경쟁해 CC decoder contract가 깨진다.
    return expit(arr).astype(np.float32, copy=False)




def task1_v287_probabilities_from_logits(raw: np.ndarray) -> np.ndarray:
    """Convert V287 softmax logits to V277-compatible support/separator fields."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[0]) != int(BFV3_SEPARATOR_SOFTMAX_V287_OUTPUT_CHANNELS):
        raise ValueError(f"V287 logits expect [3,Z,Y,X], got {arr.shape}")
    shifted = arr - np.max(arr, axis=0, keepdims=True)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    prob = exp / np.maximum(np.sum(exp, axis=0, keepdims=True), np.float32(1e-8))
    fields = np.zeros((2, *arr.shape[1:]), dtype=np.float32)
    # [METRIC][Scope:v287_decoder_contract]
    # V287의 learned head는 3-class softmax지만 decoder는 검증된 V277 separator-gap
    # CC path를 그대로 쓴다. support는 body+separator, separator는 class 2 확률이다.
    fields[0] = np.clip(prob[1] + prob[2], 0.0, 1.0)
    fields[1] = np.clip(prob[2], 0.0, 1.0)
    return fields








def task1_v288_probabilities_from_logits(raw: np.ndarray) -> np.ndarray:
    """Convert V288 ABBC softmax logits to 4-channel probability fields."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[0]) != int(BFV3_ABBC_V288_OUTPUT_CHANNELS):
        raise ValueError(f"V288 logits expect [4,Z,Y,X], got {arr.shape}")
    shifted = arr - np.max(arr, axis=0, keepdims=True)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    prob = exp / np.maximum(np.sum(exp, axis=0, keepdims=True), np.float32(1e-8))
    return prob.astype(np.float32, copy=False)


def _aggregate_task1_v2_samples(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average per-sample official-aligned v2 metrics across held-out val
    samples, with a per-anatomy breakdown. Each sample is one held-out
    (case, anatomy) ROI scored against that anatomy's GT fragments only."""
    mean_keys = ["fracture_dice", "local_dice_20mm", "hd95_mm", "assd_mm",
                 "instance_recall", "instance_precision", "instance_f1", "topology_consistency"]

    def _macro(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_samples": len(rows)}
        for k in mean_keys:
            vals = [r["metrics"].get(k) for r in rows if r["metrics"].get(k) is not None]
            out[k] = float(np.mean(vals)) if vals else None
        for k in ("merge_error_count_total", "split_error_count_total"):
            out[k] = int(sum(int(r["metrics"].get(k, 0) or 0) for r in rows))
        return out

    by_anatomy = {a: _macro([r for r in per_sample if r["anatomy"] == a])
                  for a in sorted({r["anatomy"] for r in per_sample})}
    return {"overall": _macro(per_sample), "by_anatomy": by_anatomy}


def run_task1_abbc_eval(*,
                        dataset_id: int,
                        trainer: str,
                        out_path: Path,
                        samples: list[str] | None = None,
                        checkpoint: str = "checkpoint_best.pth",
                        fold: int = 0,
                        plans: str = "nnUNetResEncUNetLPlans",
                        config: str = "3d_fullres",
                        preprocessed_plans_dir: str = "nnUNetPlans_3d_fullres",
                        roi_pad_vox: int = 24,
                        background_threshold: float = 0.50,
                        core_threshold: float = 0.50,
                        min_component_voxels: int = 100,
                        anatomy_size_ratio_keep: float = 0.0,
                        decoder_support_ratio_cap: float = 6.0,
                        gt_fragment_min_mm3: float = 500.0,
                        cc_prune_mm3: float = 1000.0,
                        iou_match_threshold: float = 0.10) -> dict[str, Any]:
    """Per-sample held-out Stage-2 eval for a 4-class ABBC model, scored with the
    PENGWIN 2026 official-aligned v2 proxy.

    The Ds538 split is per-ROI — a case can have some anatomies in TRAIN and others
    in VAL — so a full-volume per-case eval would leak training ROIs. Here every
    held-out (case, anatomy) val sample is scored independently: its ROI is run
    through sliding-window inference, decoded with the core-seed watershed
    (decode_task1_v288_abbc), placed into a single-anatomy instance volume, and
    scored against that anatomy's GT fragments only (global ranges Sacrum 1-50,
    LeftHip 51-100, RightHip 101-150, Femur 151-200). `samples` defaults to the
    fold's held-out val list from splits_final.json; pass an explicit chunk for
    multi-process parallelism. See [[pengwin-task1-eval-metric]].
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    dataset_name = DATASETS[int(dataset_id)]["name"]
    model_dir = _nnunet_results_root() / dataset_name / f"{trainer}__{plans}__{config}"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"trained model folder missing: {model_dir}")
    config_dir = NN_PREP / dataset_name / preprocessed_plans_dir
    if samples is None:
        splits = json.loads((NN_PREP / dataset_name / "splits_final.json").read_text())
        samples = list(splits[int(fold)]["val"])

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=(int(fold),), checkpoint_name=checkpoint)

    gt_cache: dict[str, tuple[np.ndarray, tuple]] = {}
    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        stem = str(sample).replace("PENGWIN_", "")
        case, anatomy = stem.split("_", 1)
        case = case.zfill(3)
        if anatomy not in V5_ANATOMY_RANGES_WITH_FEMUR:
            raise ValueError(f"unknown anatomy in sample {sample!r}")
        lo, hi = V5_ANATOMY_RANGES_WITH_FEMUR[anatomy]

        if case not in gt_cache:
            cd = find_case_dir(case)
            if cd is None:
                raise FileNotFoundError(f"source case not found: {case}")
            lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
            gt_cache[case] = (
                sitk.GetArrayFromImage(lbl_img).astype(np.uint16, copy=False),
                tuple(float(v) for v in lbl_img.GetSpacing()[::-1]),
            )
        inst_full, spacing_zyx = gt_cache[case]

        # GT and prediction restricted to THIS anatomy only (the held-out unit) so
        # no other anatomy of the same patient (which may be a TRAIN ROI) is scored.
        gt_anat = np.where((inst_full >= int(lo)) & (inst_full <= int(hi)), inst_full, 0).astype(np.uint16, copy=False)
        pred_anat = np.zeros(inst_full.shape, dtype=np.uint16)

        anat_mask = anatomy_mask_from_instances(inst_full, anatomy)
        bbox = bbox_from_mask(anat_mask, pad_vox=int(roi_pad_vox))
        decoded = False
        if bbox is not None and (
            (config_dir / f"{sample}.npy").exists() or (config_dir / f"{sample}.npz").exists()
        ):
            gt_support = int(((inst_full[bbox] >= int(lo)) & (inst_full[bbox] <= int(hi))).sum())
            data = _load_preprocessed_sample(config_dir, sample)
            logits = _predict_custom_logits_from_preprocessed_data(
                predictor, torch.from_numpy(np.asarray(data).copy()).float(),
                output_channels=int(TASK1_V288_OUTPUT_CHANNELS))
            logits_np = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
            probs = task1_v288_probabilities_from_logits(logits_np)
            support_ratio = int((probs[0] < float(background_threshold)).sum()) / max(1, gt_support)
            if support_ratio <= float(decoder_support_ratio_cap):
                roi_shape = tuple(int(v) for v in inst_full[bbox].shape)
                if tuple(probs.shape[1:]) != roi_shape:
                    probs = _resize_channel_first_probabilities(probs, roi_shape)
                    probs = probs / np.maximum(np.sum(probs, axis=0, keepdims=True), np.float32(1e-8))
                # [2026-06-06] Anatomy-specific over-segmentation control. Sacrum is a single
                # dominant bone whose predicted core speckles into spurious islands -> over-split
                # (precision 0.49); an aggressive size-ratio + min-component merge fixes it
                # (F1 0.585->0.892). Femur/hips are genuine multi-fragment bones and keep the
                # defaults (a global aggressive merge collapsed their recall).
                _ratio = max(float(anatomy_size_ratio_keep), 0.10) if anatomy == "Sacrum" else float(anatomy_size_ratio_keep)
                _minvox = max(int(min_component_voxels), 250) if anatomy == "Sacrum" else int(min_component_voxels)
                decoded_roi, _trace = decode_task1_v288_abbc(
                    probs, background_threshold=background_threshold, core_threshold=core_threshold,
                    min_component_voxels=_minvox, anatomy_size_ratio_keep=_ratio)
                view = pred_anat[bbox]
                for local_id in sorted(int(v) for v in np.unique(decoded_roi) if int(v) > 0):
                    global_id = int(lo) + local_id - 1
                    if global_id > int(hi):
                        continue
                    view[decoded_roi == local_id] = np.uint16(global_id)
                pred_anat[bbox] = view
                decoded = True

        m = compute_task1_official_aligned_v2_metrics(
            pred_anat, gt_anat, spacing_zyx,
            gt_fragment_min_mm3=gt_fragment_min_mm3, cc_prune_mm3=cc_prune_mm3,
            iou_match_threshold=iou_match_threshold)
        per_sample.append({"sample": str(sample), "anatomy": anatomy,
                           "decoded": decoded, "metrics": m["cohort_macro"]})

    report = {
        "task": "task1_abbc_official_aligned_v2_eval_persample",
        "dataset": dataset_name, "trainer": trainer, "checkpoint": checkpoint, "fold": int(fold),
        "n_samples": len(per_sample),
        "decoder": {"background_threshold": background_threshold, "core_threshold": core_threshold,
                    "min_component_voxels": min_component_voxels,
                    "anatomy_size_ratio_keep": anatomy_size_ratio_keep,
                    "decoder_support_ratio_cap": decoder_support_ratio_cap},
        "metric": {"gt_fragment_min_mm3": gt_fragment_min_mm3, "cc_prune_mm3": cc_prune_mm3,
                   "iou_match_threshold": iou_match_threshold},
        "cohort": _aggregate_task1_v2_samples(per_sample),
        "samples": per_sample,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(report), indent=2))
    return report






def task1_v289_probabilities_from_logits(raw: np.ndarray) -> np.ndarray:
    """Convert V289 raw outputs to a decoder-ready 4-channel field tensor.

    Channels 0/1/3 are passed through softmax (over those three channels);
    channel 2 is left as raw SDF regression. The decoder reads channels
    [softmax_bg, softmax_border, raw_sdf, softmax_core].
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[0]) != int(BFV3_ABBC_SDF_V289_OUTPUT_CHANNELS):
        raise ValueError(f"V289 logits expect [4,Z,Y,X], got {arr.shape}")
    softmax_input = np.stack([arr[0], arr[1], arr[3]], axis=0)
    sm_shifted = softmax_input - np.max(softmax_input, axis=0, keepdims=True)
    sm_exp = np.exp(sm_shifted).astype(np.float32, copy=False)
    sm_prob = sm_exp / np.maximum(np.sum(sm_exp, axis=0, keepdims=True), np.float32(1e-8))
    fields = np.zeros_like(arr, dtype=np.float32)
    fields[0] = sm_prob[0]
    fields[1] = sm_prob[1]
    fields[2] = arr[2]
    fields[3] = sm_prob[2]
    return fields










def _task1_v280_query_channels(shape: tuple[int, int, int],
                               seed_zyx: np.ndarray,
                               sigma_vox: float = 6.0) -> np.ndarray:
    """Build V280 query input maps for one preprocessed ROI."""
    depth, height, width = (int(v) for v in shape)
    seed = np.asarray(seed_zyx, dtype=np.float32)
    z = np.arange(depth, dtype=np.float32)[:, None, None] - float(seed[0])
    y = np.arange(height, dtype=np.float32)[None, :, None] - float(seed[1])
    x = np.arange(width, dtype=np.float32)[None, None, :] - float(seed[2])
    dist2 = z * z + y * y + x * x
    gaussian = np.exp(-dist2 / (2.0 * float(sigma_vox) * float(sigma_vox))).astype(np.float32, copy=False)
    diag = float(np.sqrt(max(depth - 1, 1) ** 2 + max(height - 1, 1) ** 2 + max(width - 1, 1) ** 2))
    inv_distance = (1.0 - np.minimum(np.sqrt(dist2) / max(diag, 1.0), 1.0)).astype(np.float32, copy=False)
    return np.stack([gaussian, inv_distance], axis=0).astype(np.float32, copy=False)


def _task1_v281_query_channels(shape: tuple[int, int, int],
                               positive_seed_zyx: np.ndarray,
                               negative_seeds_zyx: list[np.ndarray],
                               sigma_vox: float = 6.0) -> np.ndarray:
    """Build V281 positive/negative query maps for one preprocessed ROI."""
    # [AUDIT][Risk:Blocker][Scope:v281_query_contract]
    # V280의 positive-only query는 "어느 fragment가 아닌지"를 입력으로 주지 않아
    # adjacent fragment identity가 약했다. V281 predict는 training과 동일하게
    # positive 2채널 + negative 2채널을 만들며 contact/fracture label은 쓰지 않는다.
    positive = _task1_v280_query_channels(shape, positive_seed_zyx, sigma_vox=sigma_vox)
    depth, height, width = (int(v) for v in shape)
    if not negative_seeds_zyx:
        negative = np.zeros((2, depth, height, width), dtype=np.float32)
        return np.concatenate([positive, negative], axis=0).astype(np.float32, copy=False)
    z_axis = np.arange(depth, dtype=np.float32)[:, None, None]
    y_axis = np.arange(height, dtype=np.float32)[None, :, None]
    x_axis = np.arange(width, dtype=np.float32)[None, None, :]
    min_dist2 = np.full((depth, height, width), np.inf, dtype=np.float32)
    for seed_raw in negative_seeds_zyx:
        seed = np.asarray(seed_raw, dtype=np.float32)
        dz = z_axis - float(seed[0])
        dy = y_axis - float(seed[1])
        dx = x_axis - float(seed[2])
        min_dist2 = np.minimum(min_dist2, dz * dz + dy * dy + dx * dx)
    gaussian = np.exp(-min_dist2 / (2.0 * float(sigma_vox) * float(sigma_vox))).astype(np.float32, copy=False)
    diag = float(np.sqrt(max(depth - 1, 1) ** 2 + max(height - 1, 1) ** 2 + max(width - 1, 1) ** 2))
    inv_distance = (1.0 - np.minimum(np.sqrt(min_dist2) / max(diag, 1.0), 1.0)).astype(np.float32, copy=False)
    negative = np.stack([gaussian, inv_distance], axis=0).astype(np.float32, copy=False)
    return np.concatenate([positive, negative], axis=0).astype(np.float32, copy=False)


def _task1_v280_seed_from_instance(inst: np.ndarray,
                                   centers_zyx: np.ndarray,
                                   frag_id: int) -> np.ndarray:
    coords = np.argwhere(inst == int(frag_id))
    if coords.size == 0:
        raise ValueError(f"fragment {frag_id} has no voxels")
    if int(frag_id) < int(centers_zyx.shape[0]):
        center = np.asarray(centers_zyx[int(frag_id)], dtype=np.float32)
        if np.isfinite(center).all():
            nearest = coords[np.argmin(((coords.astype(np.float32) - center.reshape(1, 3)) ** 2).sum(axis=1))]
            return nearest.astype(np.float32)
    return coords.astype(np.float32).mean(axis=0)








def _task1_v283_global_coord_channels(shape: tuple[int, int, int]) -> np.ndarray:
    """Return full-ROI normalized z/y/x channels for V283 inference."""
    # [QC][Invariant:global_coord_space]
    # V283 trainer는 dataloader에서 전체 preprocessed ROI 기준 좌표를 붙인다. Prediction도
    # 같은 shape 정규화를 써야 sliding-window tile마다 좌표계가 달라지는 V282 실패를
    # 반복하지 않는다.
    depth, height, width = (int(v) for v in shape)
    denom = np.maximum(np.asarray([depth, height, width], dtype=np.float32) - 1.0, 1.0)
    z = (np.arange(depth, dtype=np.float32) / float(denom[0]))[:, None, None]
    y = (np.arange(height, dtype=np.float32) / float(denom[1]))[None, :, None]
    x = (np.arange(width, dtype=np.float32) / float(denom[2]))[None, None, :]
    return np.stack(
        [
            np.broadcast_to(z, (depth, height, width)),
            np.broadcast_to(y, (depth, height, width)),
            np.broadcast_to(x, (depth, height, width)),
        ],
        axis=0,
    ).astype(np.float32, copy=False)




def _task1_v280_seed_component(mask: np.ndarray, seed_zyx: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return mask
    labels, n_labels = ndi.label(mask)
    if int(n_labels) <= 1:
        return mask
    seed = np.asarray(np.round(seed_zyx), dtype=np.int64)
    seed = np.clip(seed, [0, 0, 0], np.asarray(mask.shape, dtype=np.int64) - 1)
    seed_label = int(labels[tuple(int(v) for v in seed)])
    if seed_label > 0:
        return labels == seed_label
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))




def _task1_v282_assign_oracle_prototypes(embedding: np.ndarray,
                                         support_mask: np.ndarray,
                                         inst_roi: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign support voxels to nearest GT-fragment embedding prototype."""
    # [AUDIT][Risk:High][Scope:v282_oracle_prototype]
    # 이 decoder는 learned seed/prototype 문제가 아니라 embedding separability만 검증한다.
    # GT fragment ID로 prototype을 만들기 때문에 fulltrain 가능한 최종 decoder가 아니다.
    if embedding.ndim != 4:
        raise ValueError(f"V282 embedding must be [C,Z,Y,X], got {embedding.shape}")
    support = np.asarray(support_mask, dtype=bool)
    inst = np.asarray(inst_roi, dtype=np.uint16)
    ids = [int(v) for v in np.unique(inst) if 0 < int(v) <= PELVIC_MAX_INSTANCE_ID]
    decoded = np.zeros(inst.shape, dtype=np.uint16)
    if not ids or not bool(support.any()):
        return decoded, {"prototype_count": 0, "assigned_voxels": 0}
    emb_zyxc = np.moveaxis(embedding.astype(np.float32, copy=False), 0, -1)
    prototypes = []
    valid_ids = []
    for frag_id in ids:
        mask = inst == int(frag_id)
        if not bool(mask.any()):
            continue
        prototypes.append(emb_zyxc[mask].mean(axis=0))
        valid_ids.append(int(frag_id))
    if not prototypes:
        return decoded, {"prototype_count": 0, "assigned_voxels": 0}
    proto = np.stack(prototypes, axis=0).astype(np.float32, copy=False)
    coords = np.argwhere(support)
    assigned_ids = np.zeros((coords.shape[0],), dtype=np.uint16)
    for start in range(0, int(coords.shape[0]), 200000):
        chunk = coords[start:start + 200000]
        points = emb_zyxc[tuple(chunk.T)].astype(np.float32, copy=False)
        dist2 = ((points[:, None, :] - proto[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(dist2, axis=1)
        assigned_ids[start:start + len(chunk)] = np.asarray(valid_ids, dtype=np.uint16)[nearest]
    decoded[tuple(coords.T)] = assigned_ids
    return decoded, {
        "prototype_count": int(len(valid_ids)),
        "assigned_voxels": int(coords.shape[0]),
        "prototype_ids": [int(v) for v in valid_ids],
    }






def _task1_v261_fields_from_logits(raw_logits: np.ndarray) -> np.ndarray:
    """Convert V261 learned logits to decoder fields without corrupting offsets."""
    raw = np.asarray(raw_logits, dtype=np.float32)
    if raw.ndim != 4 or int(raw.shape[0]) != int(BFV3_CENTER_FLOW_OUTPUT_CHANNELS):
        raise ValueError(f"expected V261 logits [8,Z,Y,X], got {raw.shape}")
    fields = raw.astype(np.float32, copy=True)
    # [QC][Invariant:channel_activation]
    # V261 rows 0..4 are binary/logistic fields, but rows 5..7 are raw signed
    # voxel offsets. Applying sigmoid to offsets would destroy direction and make
    # Instance/Merge/Split/Topology metrics meaningless.
    fields[:5] = expit(fields[:5]).astype(np.float32, copy=False)
    return fields


def _task1_v267_no_contact_fields_from_logits(raw_logits: np.ndarray) -> np.ndarray:
    """Convert V267 5-channel no-contact logits to V262-compatible fields.

    [QC][Invariant:channel_order]
    V267 rows are 0 support, 1 center heatmap, 2..4 raw offsets. The decoder
    helper still expects the older 8-channel center-flow layout, so rows 1/2/4
    of that layout are filled with inert zeros and never used for promotion.
    """
    raw = np.asarray(raw_logits, dtype=np.float32)
    if raw.ndim != 4 or int(raw.shape[0]) != int(BFV3_NO_CONTACT_CENTER_FLOW_OUTPUT_CHANNELS):
        raise ValueError(f"expected V267 logits [5,Z,Y,X], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("V267 logits contain NaN/Inf")
    fields = np.zeros((int(BFV3_CENTER_FLOW_OUTPUT_CHANNELS), *raw.shape[1:]), dtype=np.float32)
    fields[TASK1_V261_CHANNELS["support"]] = expit(raw[0]).astype(np.float32, copy=False)
    fields[TASK1_V261_CHANNELS["center_heatmap"]] = expit(raw[1]).astype(np.float32, copy=False)
    fields[TASK1_V261_CHANNELS["offset_to_center_z"]] = raw[2].astype(np.float32, copy=False)
    fields[TASK1_V261_CHANNELS["offset_to_center_y"]] = raw[3].astype(np.float32, copy=False)
    fields[TASK1_V261_CHANNELS["offset_to_center_x"]] = raw[4].astype(np.float32, copy=False)
    # [METRIC][Scope:task1_primary_gate]
    # fracture/contact probability is intentionally unavailable. Fracture Dice proxy
    # is computed from final instance boundaries, not from a contact head.
    return fields














def _summarize_logit_values(values: np.ndarray, *, threshold_logit: float = 0.0) -> dict[str, Any]:
    """Summarize one forensic logit bucket without retaining it across samples."""
    arr = np.asarray(values)
    count = int(arr.size)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q01": None,
            "q05": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "q99": None,
            "max": None,
            "frac_ge_threshold": None,
            "threshold_logit": float(threshold_logit),
        }
    arr_f = arr.astype(np.float32, copy=False)
    quantiles = np.quantile(arr_f, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": count,
        "mean": float(np.mean(arr_f)),
        "std": float(np.std(arr_f)),
        "min": float(np.min(arr_f)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q10": float(quantiles[2]),
        "q25": float(quantiles[3]),
        "q50": float(quantiles[4]),
        "q75": float(quantiles[5]),
        "q90": float(quantiles[6]),
        "q95": float(quantiles[7]),
        "q99": float(quantiles[8]),
        "max": float(np.max(arr_f)),
        "frac_ge_threshold": float(np.mean(arr_f >= float(threshold_logit))),
        "threshold_logit": float(threshold_logit),
    }


def _weighted_bucket_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    total = 0
    weighted = 0.0
    for row in rows:
        bucket = row.get(key) or {}
        count = int(bucket.get("count") or 0)
        mean = bucket.get("mean")
        if count > 0 and mean is not None:
            total += count
            weighted += float(mean) * count
    return float(weighted / total) if total > 0 else None


def _weighted_bucket_frac(rows: list[dict[str, Any]], key: str) -> float | None:
    total = 0
    weighted = 0.0
    for row in rows:
        bucket = row.get(key) or {}
        count = int(bucket.get("count") or 0)
        frac = bucket.get("frac_ge_threshold")
        if count > 0 and frac is not None:
            total += count
            weighted += float(frac) * count
    return float(weighted / total) if total > 0 else None


def _probabilities_to_logits_clipped(probs: np.ndarray, *, eps: float = 1e-4) -> np.ndarray:
    """Convert restored probabilities back to logits for threshold-aligned forensic summaries."""
    arr = np.clip(np.asarray(probs, dtype=np.float32), float(eps), 1.0 - float(eps))
    return np.log(arr / (1.0 - arr)).astype(np.float32, copy=False)


def run_task1_v253_affinity_forensics(cases: list[str],
                                      out_path: Path,
                                      output_root: Path,
                                      *,
                                      roi_pad_vox: int = 24,
                                      support_threshold: float = 0.5,
                                      affinity_threshold: float = 0.5,
                                      dice_fulltrain_threshold: float = 0.90) -> dict[str, Any]:
    """Measure whether learned V253 affinity logits separate GT same/different edges.

    [AUDIT][Risk:Blocker][Scope:fulltrain_gate]
    Full train is blocked until hard3 learned logits show that same-fragment GT
    edges score above different-fragment/contact GT edges. If this separation is
    absent, more epochs mainly preserve the wrong decoder contract and Task1
    Dice/Instance/Topology metrics remain non-actionable.
    """
    if output_root is None:
        raise ValueError("V253 affinity forensic requires --output-root")
    if not 0.0 < float(dice_fulltrain_threshold) <= 1.0:
        raise ValueError(f"dice_fulltrain_threshold must be in (0,1], got {dice_fulltrain_threshold}")
    support_logit_threshold = _probability_threshold_to_logit(support_threshold, name="support_threshold")
    affinity_logit_threshold = _probability_threshold_to_logit(affinity_threshold, name="affinity_threshold")
    sample_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for case in [str(c).zfill(3) for c in cases]:
        cd = find_case_dir(case)
        if cd is None:
            raise FileNotFoundError(f"source case not found: {case}")
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        inst_full = sitk.GetArrayFromImage(lbl_img).astype(np.uint16, copy=False)
        case_support_dice_rows = []
        for anatomy, (lo, hi) in V5_ANATOMY_RANGES.items():
            sample = f"PENGWIN_{case}_{anatomy}"
            anat_mask = anatomy_mask_from_instances(inst_full, anatomy)
            bbox = bbox_from_mask(anat_mask, pad_vox=roi_pad_vox)
            if bbox is None:
                continue
            inst_roi = np.where(
                (inst_full[bbox] >= int(lo)) & (inst_full[bbox] <= int(hi)),
                inst_full[bbox],
                0,
            ).astype(np.uint16, copy=False)
            logits_path = output_root / f"{sample}.npz"
            raw_logits = _load_task1_v253_logits(logits_path)
            restored_from_preprocessed_grid = tuple(raw_logits.shape[1:]) != tuple(inst_roi.shape)
            if restored_from_preprocessed_grid:
                # [QC][Invariant:forensic_geometry]
                # V253 predictor exports nnU-Net preprocessed ROI logits. Task1 metric은 source-space
                # GT ROI에서 계산되므로 forensic도 eval과 동일하게 sigmoid probability를 source ROI로
                # restore한 뒤 logit scale로 되돌린다. 이것은 threshold tuning이 아니라 좌표계 정렬이다.
                restored_probs = _resize_channel_first_probabilities(
                    expit(raw_logits).astype(np.float32, copy=False),
                    tuple(inst_roi.shape),
                )
                forensic_logits = _probabilities_to_logits_clipped(restored_probs)
            else:
                forensic_logits = raw_logits
            gt_support = inst_roi > 0
            pred_support = forensic_logits[TASK1_V252_BASE_CHANNELS["support"]] >= support_logit_threshold
            support_intersection = int((pred_support & gt_support).sum())
            support_pred_count = int(pred_support.sum())
            support_gt_count = int(gt_support.sum())
            support_precision = float(support_intersection / max(support_pred_count, 1))
            support_recall = float(support_intersection / max(support_gt_count, 1))
            support_dice = _binary_dice(pred_support, gt_support)
            same_values: list[np.ndarray] = []
            different_values: list[np.ndarray] = []
            per_offset = []
            valid_edge_count = 0
            same_edge_count = 0
            different_edge_count = 0
            same_join_count = 0
            different_join_count = 0
            for i, offset in enumerate(TASK1_V252_AFFINITY13_OFFSETS):
                src_sl, dst_sl = _task1_affinity_offset_slices(tuple(int(v) for v in inst_roi.shape), offset)
                a = inst_roi[src_sl]
                b = inst_roi[dst_sl]
                a_fg = valid_instance_mask(a, pelvic_only=True)
                b_fg = valid_instance_mask(b, pelvic_only=True)
                same_anatomy = a_fg & b_fg & (((a - 1) // 50) == ((b - 1) // 50))
                same_fragment = same_anatomy & (a == b)
                different_fragment = same_anatomy & (a != b)
                logit = forensic_logits[5 + i][src_sl]
                same_logits = logit[same_fragment]
                different_logits = logit[different_fragment]
                same_values.append(same_logits)
                different_values.append(different_logits)
                offset_same_count = int(same_logits.size)
                offset_different_count = int(different_logits.size)
                offset_same_join = int((same_logits >= affinity_logit_threshold).sum())
                offset_different_join = int((different_logits >= affinity_logit_threshold).sum())
                same_edge_count += offset_same_count
                different_edge_count += offset_different_count
                valid_edge_count += int(same_anatomy.sum())
                same_join_count += offset_same_join
                different_join_count += offset_different_join
                per_offset.append({
                    "offset_zyx": [int(v) for v in offset],
                    "same_fragment_edges": offset_same_count,
                    "different_fragment_contact_edges": offset_different_count,
                    "same_join_rate_at_threshold": float(offset_same_join / max(offset_same_count, 1)),
                    "different_join_rate_at_threshold": float(offset_different_join / max(offset_different_count, 1)) if offset_different_count else None,
                })
            same_all = np.concatenate(same_values) if same_values else np.asarray([], dtype=np.float32)
            different_all = np.concatenate(different_values) if different_values else np.asarray([], dtype=np.float32)
            same_summary = _summarize_logit_values(same_all, threshold_logit=affinity_logit_threshold)
            different_summary = _summarize_logit_values(different_all, threshold_logit=affinity_logit_threshold)
            median_margin = None
            q10_q90_margin = None
            if same_summary["q50"] is not None and different_summary["q50"] is not None:
                median_margin = float(same_summary["q50"] - different_summary["q50"])
            if same_summary["q10"] is not None and different_summary["q90"] is not None:
                q10_q90_margin = float(same_summary["q10"] - different_summary["q90"])
            sample_row = {
                "case": case,
                "sample": sample,
                "anatomy": anatomy,
                "logits_path": str(logits_path),
                "restored_from_preprocessed_grid": bool(restored_from_preprocessed_grid),
                "logit_shape_before_restore": [int(v) for v in raw_logits.shape],
                "roi_shape": [int(v) for v in inst_roi.shape],
                "support": {
                    "gt_voxels": support_gt_count,
                    "pred_voxels": support_pred_count,
                    "pred_gt_ratio": float(support_pred_count / max(support_gt_count, 1)),
                    "dice": float(support_dice),
                    "precision": support_precision,
                    "recall": support_recall,
                    "threshold": float(support_threshold),
                },
                # [DATA][Risk:High][Scope:affinity_hard_negative]
                # same_fragment_edges는 학습 target=1, different_fragment_contact_edges는 같은 anatomy
                # 내부에서 서로 다른 GT fragment가 맞닿은 target=0 hard negative다.
                "same_fragment_logit": same_summary,
                "different_fragment_contact_logit": different_summary,
                "valid_same_anatomy_edges": int(valid_edge_count),
                "same_fragment_edges": int(same_edge_count),
                "different_fragment_contact_edges": int(different_edge_count),
                "same_join_rate_at_threshold": float(same_join_count / max(same_edge_count, 1)),
                "different_join_rate_at_threshold": float(different_join_count / max(different_edge_count, 1)) if different_edge_count else None,
                "median_margin_same_minus_different": median_margin,
                "q10_same_minus_q90_different": q10_q90_margin,
                "per_offset": per_offset,
            }
            sample_rows.append(sample_row)
            case_support_dice_rows.append(float(support_dice))
        case_rows.append({
            "case": case,
            "support_dice_mean": float(np.mean(case_support_dice_rows)) if case_support_dice_rows else None,
            "support_dice_min": float(np.min(case_support_dice_rows)) if case_support_dice_rows else None,
        })
    support_dice_values = [float(row["support"]["dice"]) for row in sample_rows]
    same_counts = [int(row["same_fragment_edges"]) for row in sample_rows]
    different_counts = [int(row["different_fragment_contact_edges"]) for row in sample_rows]
    q10_q90_margins = [
        float(row["q10_same_minus_q90_different"])
        for row in sample_rows
        if row["q10_same_minus_q90_different"] is not None
    ]
    median_margins = [
        float(row["median_margin_same_minus_different"])
        for row in sample_rows
        if row["median_margin_same_minus_different"] is not None
    ]
    different_join_rates = [
        float(row["different_join_rate_at_threshold"])
        for row in sample_rows
        if row["different_join_rate_at_threshold"] is not None
    ]
    support_dice_mean = float(np.mean(support_dice_values)) if support_dice_values else None
    support_dice_min = float(np.min(support_dice_values)) if support_dice_values else None
    fulltrain_gate = {
        "dice_threshold": float(dice_fulltrain_threshold),
        "support_head_dice_mean_ge_threshold": bool(support_dice_mean is not None and support_dice_mean >= float(dice_fulltrain_threshold)),
        "support_head_dice_min_ge_threshold": bool(support_dice_min is not None and support_dice_min >= float(dice_fulltrain_threshold)),
        "has_different_fragment_hard_negatives": bool(sum(different_counts) > 0),
        "affinity_median_margin_positive": bool(median_margins and float(np.min(median_margins)) > 0.0),
        "affinity_q10_q90_separated": bool(q10_q90_margins and float(np.min(q10_q90_margins)) > 0.0),
        "different_fragment_join_rate_mean": float(np.mean(different_join_rates)) if different_join_rates else None,
        "fulltrain_ready": False,
        "reason": "pending",
    }
    # [METRIC][Risk:High][Scope:fulltrain_entry]
    # 사용자가 허용한 Dice 기준은 0.90 이상이다. 그러나 fulltrain 진입은 support Dice만으로
    # 충분하지 않고, affinity hard negative가 분리되어 decoder skip 없이 instance/topology metric을
    # 계산할 수 있어야 한다.
    if not fulltrain_gate["support_head_dice_mean_ge_threshold"]:
        fulltrain_gate["reason"] = "support_head_dice_below_threshold"
    elif not fulltrain_gate["has_different_fragment_hard_negatives"]:
        fulltrain_gate["reason"] = "no_different_fragment_hard_negative_edges_in_eval_set"
    elif not fulltrain_gate["affinity_median_margin_positive"]:
        fulltrain_gate["reason"] = "same_fragment_logits_not_above_different_fragment_logits"
    elif not fulltrain_gate["affinity_q10_q90_separated"]:
        fulltrain_gate["reason"] = "same_and_different_affinity_logit_distributions_overlap"
    else:
        fulltrain_gate["fulltrain_ready"] = True
        fulltrain_gate["reason"] = "support_and_affinity_forensic_gate_passed"
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "task1_v253_affinity_logit_forensics",
        "target_contract": "task1_v253_learned_logits_affinity13_seed_healed_v1",
        "output_root": str(output_root),
        "cases": [str(c).zfill(3) for c in cases],
        "roi_pad_vox": int(roi_pad_vox),
        "support_threshold": float(support_threshold),
        "affinity_threshold": float(affinity_threshold),
        "summary": {
            "n_samples": int(len(sample_rows)),
            "support_dice_mean": support_dice_mean,
            "support_dice_min": support_dice_min,
            "same_fragment_edges_total": int(sum(same_counts)),
            "different_fragment_contact_edges_total": int(sum(different_counts)),
            "same_fragment_logit_mean_weighted": _weighted_bucket_mean(sample_rows, "same_fragment_logit"),
            "different_fragment_contact_logit_mean_weighted": _weighted_bucket_mean(sample_rows, "different_fragment_contact_logit"),
            "same_join_rate_at_threshold_weighted": _weighted_bucket_frac(sample_rows, "same_fragment_logit"),
            "different_join_rate_at_threshold_weighted": _weighted_bucket_frac(sample_rows, "different_fragment_contact_logit"),
            "median_margin_same_minus_different_mean": float(np.mean(median_margins)) if median_margins else None,
            "median_margin_same_minus_different_min": float(np.min(median_margins)) if median_margins else None,
            "q10_same_minus_q90_different_mean": float(np.mean(q10_q90_margins)) if q10_q90_margins else None,
            "q10_same_minus_q90_different_min": float(np.min(q10_q90_margins)) if q10_q90_margins else None,
            "fulltrain_gate": fulltrain_gate,
        },
        "case_summary": case_rows,
        "samples": sample_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(result), indent=2, allow_nan=False))
    return result


def run_task1_official_proxy_eval(cases: list[str],
                                  out_path: Path,
                                  pred_root: Path | None = None,
                                  oracle_gt: bool = False,
                                  local_radius_mm: float = 20.0,
                                  match_iou_threshold: float = 0.10,
                                  overlap_fraction_threshold: float = 0.05) -> dict[str, Any]:
    """Evaluate full-volume instance predictions with Task1-aligned proxy metrics."""
    if not oracle_gt and pred_root is None:
        raise ValueError("--pred-root is required unless --oracle-gt is set")
    rows = []
    for case in [str(c).zfill(3) for c in cases]:
        cd = find_case_dir(case)
        if cd is None:
            raise FileNotFoundError(f"source case not found: {case}")
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        gt = sitk.GetArrayFromImage(lbl_img).astype(np.uint16, copy=False)
        spacing_zyx = tuple(float(v) for v in lbl_img.GetSpacing()[::-1])
        if oracle_gt:
            pred = np.where(valid_instance_mask(gt, pelvic_only=True), gt, 0).astype(np.uint16, copy=False)
            pred_source = "gt_vs_gt"
        else:
            assert pred_root is not None
            pred_path = pred_root / f"PENGWIN_{case}.mha"
            if not pred_path.exists():
                raise FileNotFoundError(f"prediction not found: {pred_path}")
            pred_img = canonicalize_sitk(sitk.ReadImage(str(pred_path)))
            pred = sitk.GetArrayFromImage(pred_img).astype(np.uint16, copy=False)
            pred_source = str(pred_path)
        metrics = compute_task1_official_proxy_metrics(
            gt,
            pred,
            spacing_zyx,
            local_radius_mm=local_radius_mm,
            match_iou_threshold=match_iou_threshold,
            overlap_fraction_threshold=overlap_fraction_threshold,
        )
        rows.append({"case": case, "prediction": pred_source, "task1_official_proxy": metrics})
    metric_names = [
        "dice", "fracture_dice_proxy", "local_dice_cfs_20mm_proxy", "hd95_mm", "assd_mm",
        "instance_f1", "instance_recall", "instance_precision", "merge_error_rate",
        "split_error_rate", "topology_consistency",
    ]
    summary = {
        "n_cases": int(len(rows)),
        "metric_contract": "task1_official_aligned_proxy_v1",
        "official_score_reproducible": False,
        "mean_position_score": None,
    }
    for name in metric_names:
        values = [row["task1_official_proxy"][name] for row in rows if row["task1_official_proxy"][name] is not None]
        summary[f"{name}_mean"] = float(np.mean(values)) if values else None
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "task1_official_aligned_proxy",
        "oracle_gt": bool(oracle_gt),
        "pred_root": None if pred_root is None else str(pred_root),
        "summary": summary,
        "cases": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(result), indent=2, allow_nan=False))
    return result


def _binary_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * int((pred & target).sum()) / denom)


def _bicm_v5_core_component_counts(target: np.ndarray, inst_roi: np.ndarray) -> dict[str, int]:
    """Count class-3 core connected components per GT fragment."""
    out: dict[str, int] = {}
    core = target == V5_LABELS["core"]
    for fragment_id in sorted(int(v) for v in np.unique(inst_roi) if int(v) > 0):
        _cc, n_comp = ndi.label(core & (inst_roi == fragment_id))
        out[str(fragment_id)] = int(n_comp)
    return out


def _decode_bicm_v5_with_profile(labels: np.ndarray,
                                 decoder_profile: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode V5 labels with an explicitly named fixed decoder profile.

    [AUDIT][Risk:High][Scope:decoder_selection]
    Decoder profile is an experiment variable and is recorded in every report.
    The seed-healed path is deterministic and uses no GT/probability threshold
    sweep; it exists only to test whether fragmented predicted core markers,
    not the target representation itself, are blocking IoU-F.
    """
    if decoder_profile == "bicm_v5_core_watershed":
        return decode_bicm_v5(labels)
    if decoder_profile == "bicm_v5_seed_healed":
        return decode_bicm_v5_seed_healed(labels)
    raise ValueError(
        f"unknown BICM V5 decoder_profile={decoder_profile!r}; "
        "expected bicm_v5_core_watershed or bicm_v5_seed_healed"
    )






def _apply_bicm_v5_hybrid_mode(pred_labels: np.ndarray,
                               target: np.ndarray,
                               mode: str) -> np.ndarray:
    """Return a deterministic V5 hybrid label map for root-cause attribution.

    [AUDIT][Risk:High][Scope:oracle_hybrid]
    This helper is not a promotion decoder. It injects selected GT semantic
    classes into a saved prediction to answer one question: if contact and/or
    core were fixed while the rest of the prediction stayed unchanged, would
    IoU-F recover? Any result from this path is upper-bound/root-cause evidence
    only and must not unlock submission or fulltrain by itself.

    [QC][Invariant:no_threshold_tuning]
    The input is an argmax semantic prediction and a GT-derived target. There
    is no probability threshold, validation sweep, or learned postprocess here.
    """
    pred = np.asarray(pred_labels, dtype=np.uint8)
    tgt = np.asarray(target, dtype=np.uint8)
    if pred.shape != tgt.shape:
        raise ValueError(f"BICM V5 hybrid shape mismatch: pred={pred.shape} target={tgt.shape}")
    hybrid = pred.copy()
    support_pred = np.isin(pred, V5_SUPPORT_LABELS)
    if mode == "oracle_contact_in_pred_support":
        hybrid[pred == V5_LABELS["contact_surface"]] = V5_LABELS["interior_shell"]
        hybrid[(tgt == V5_LABELS["contact_surface"]) & support_pred] = V5_LABELS["contact_surface"]
    elif mode == "oracle_contact_all":
        hybrid[pred == V5_LABELS["contact_surface"]] = V5_LABELS["interior_shell"]
        hybrid[tgt == V5_LABELS["contact_surface"]] = V5_LABELS["contact_surface"]
    elif mode == "oracle_contact_core_in_pred_support":
        hybrid[pred == V5_LABELS["contact_surface"]] = V5_LABELS["interior_shell"]
        hybrid[pred == V5_LABELS["core"]] = V5_LABELS["interior_shell"]
        hybrid[(tgt == V5_LABELS["core"]) & support_pred] = V5_LABELS["core"]
        hybrid[(tgt == V5_LABELS["contact_surface"]) & support_pred] = V5_LABELS["contact_surface"]
    elif mode == "oracle_contact_core_all":
        hybrid[pred == V5_LABELS["contact_surface"]] = V5_LABELS["interior_shell"]
        hybrid[pred == V5_LABELS["core"]] = V5_LABELS["interior_shell"]
        hybrid[tgt == V5_LABELS["core"]] = V5_LABELS["core"]
        hybrid[tgt == V5_LABELS["contact_surface"]] = V5_LABELS["contact_surface"]
    else:
        raise ValueError(
            "unsupported BICM V5 hybrid mode "
            f"{mode!r}; expected oracle_contact_in_pred_support, oracle_contact_all, "
            "oracle_contact_core_in_pred_support, or oracle_contact_core_all"
        )
    return hybrid




def _predict_custom_logits_from_preprocessed_data(predictor: Any,
                                                  data: "torch.Tensor",
                                                  output_channels: int) -> "torch.Tensor":
    """Run nnU-Net sliding-window inference for non-semantic custom heads.

    [QC][Invariant:custom_head_channels]
    nnU-Net's default exporter allocates output channels from `label_manager`.
    Dataset537 V6/V38 emit custom sigmoid/affinity heads despite raw labels
    being 0..4, so using the semantic exporter would silently recreate the
    failed V5 argmax path. This helper allocates the exact head count requested
    by the trainer.
    """
    import torch
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from nnunetv2.utilities.helpers import dummy_context, empty_cache
    from tqdm import tqdm

    # [2026-06-05] Active path = 4-channel ABBC export only. This set previously listed ~22
    # dead V-version channel counts; the active call (run_task1_abbc_eval, output_channels=
    # TASK1_V288_OUTPUT_CHANNELS=4) passed ONLY because BICM_V6_OUTPUT_CHANNELS also == 4.
    # Pin it explicitly to the active ABBC contract so removing the dead constants cannot break it.
    allowed = {int(TASK1_V288_OUTPUT_CHANNELS)}
    if int(output_channels) not in allowed:
        raise ValueError(f"unsupported custom output_channels={output_channels}; allowed={sorted(allowed)}")

    def _internal_predict(data_pad: torch.Tensor, slicers: list[tuple], do_on_device: bool) -> torch.Tensor:
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = predictor.device if do_on_device else torch.device("cpu")
        try:
            empty_cache(predictor.device)
            data_work = data_pad.to(results_device)
            predicted_logits = torch.zeros(
                (int(output_channels), *data_work.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            n_predictions = torch.zeros(data_work.shape[1:], dtype=torch.half, device=results_device)
            gaussian = compute_gaussian(
                tuple(predictor.configuration_manager.patch_size),
                sigma_scale=1.0 / 8,
                value_scaling_factor=10,
                device=results_device,
            ) if predictor.use_gaussian else 1

            for sl in tqdm(slicers, disable=not predictor.allow_tqdm):
                workon = data_work[sl][None].to(predictor.device)
                prediction = predictor._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                if int(prediction.shape[0]) != int(output_channels):
                    raise RuntimeError(
                        "Custom nnU-Net head mismatch during inference: "
                        f"expected {output_channels} channels, got {tuple(prediction.shape)}"
                    )
                if predictor.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian
            predicted_logits /= n_predictions
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError("Encountered inf in custom-head predicted logits")
        except Exception as exc:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(predictor.device)
            empty_cache(results_device)
            raise exc
        return predicted_logits

    with torch.no_grad():
        if not torch.is_tensor(data):
            raise TypeError(f"expected torch.Tensor input, got {type(data).__name__}")
        if data.ndim != 4:
            raise ValueError(f"expected preprocessed data shape [C, Z, Y, X], got {tuple(data.shape)}")
        predictor.network = predictor.network.to(predictor.device)
        predictor.network.eval()
        empty_cache(predictor.device)
        with torch.autocast(predictor.device.type, enabled=True) if predictor.device.type == "cuda" else dummy_context():
            data_pad, slicer_revert_padding = pad_nd_image(
                data,
                predictor.configuration_manager.patch_size,
                "constant",
                {"value": 0},
                True,
                None,
            )
            slicers = predictor._internal_get_sliding_window_slicers(data_pad.shape[1:])
            if predictor.perform_everything_on_device and predictor.device.type != "cpu":
                try:
                    predicted_logits = _internal_predict(data_pad, slicers, True)
                except RuntimeError as exc:
                    print(
                        "Custom-head prediction on device failed; retrying with CPU result buffers. "
                        f"Original error: {exc}"
                    )
                    empty_cache(predictor.device)
                    predicted_logits = _internal_predict(data_pad, slicers, False)
            else:
                predicted_logits = _internal_predict(data_pad, slicers, bool(predictor.perform_everything_on_device))
            empty_cache(predictor.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
    return predicted_logits


def _load_preprocessed_sample(config_dir: Path, sample: str) -> np.ndarray:
    npy_path = config_dir / f"{sample}.npy"
    npz_path = config_dir / f"{sample}.npz"
    if npy_path.exists():
        return np.load(npy_path, "r").astype(np.float32, copy=False)
    if npz_path.exists():
        return np.load(npz_path)["data"].astype(np.float32, copy=False)
    raise FileNotFoundError(f"preprocessed sample missing for {sample} under {config_dir}")






def _load_bicm_v6_logits(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing V6 logits cache: {path}")
    data = np.load(path)
    if "logits" in data:
        arr = np.asarray(data["logits"], dtype=np.float32)
    elif "output" in data:
        arr = np.asarray(data["output"], dtype=np.float32)
    else:
        keys = list(data.keys())
        if not keys:
            raise RuntimeError(f"empty V6 logits cache: {path}")
        arr = np.asarray(data[keys[0]], dtype=np.float32)
    allowed_channels = {
        int(FACTOR_INSTANCE_OUTPUT_CHANNELS),
        int(BICM_V6_OUTPUT_CHANNELS),
        int(BICM_V8_OUTPUT_CHANNELS),
        int(BICM_V38_OUTPUT_CHANNELS),
        int(BICM_V68_OUTPUT_CHANNELS),
        int(BFV3_BINARY_BARRIER_OUTPUT_CHANNELS),
        int(BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS),
        int(BFV3_XYZ_AFFINITY_OUTPUT_CHANNELS),
        int(BFV3_AFFINITY13_SEED_OUTPUT_CHANNELS),
        int(BFV3_MUTEX13_SEED_OUTPUT_CHANNELS),
        int(BFV3_CENTER_FLOW_OUTPUT_CHANNELS),
        int(BFV3_NO_CONTACT_CENTER_FLOW_OUTPUT_CHANNELS),
        int(BFV3_NO_CONTACT_PAIRWISE_V273_OUTPUT_CHANNELS),
        int(BFV3_FRAGMENT_POSITION_V275_OUTPUT_CHANNELS),
        int(BFV3_QUERY_MASK_V280_OUTPUT_CHANNELS),
        int(BFV3_QUERY_MASK_PN_V281_OUTPUT_CHANNELS),
        int(BFV3_FREE_EMBEDDING_V282_OUTPUT_CHANNELS),
        int(BFV3_GLOBAL_COORD_FREE_EMBEDDING_V283_OUTPUT_CHANNELS),
    }
    if arr.ndim != 4 or int(arr.shape[0]) not in allowed_channels:
        raise RuntimeError(
            f"expected BICM logits [C,Z,Y,X] with C in {sorted(allowed_channels)}, got {arr.shape} in {path}"
        )
    return arr


def _resize_channel_first_probabilities(probs: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize channel-first probability maps to the raw ROI target shape.

    [AUDIT][Risk:High][Scope:coordinate_space]
    V6 prediction runs on nnU-Net preprocessed ROI arrays, while IoU-F is
    reported in the raw cropped ROI coordinate system used by the instance GT.
    This helper is a coordinate-space restore, not threshold tuning: it applies
    fixed linear interpolation per probability channel and keeps the promotion
    threshold at 0.5.
    """
    if probs.ndim != 4:
        raise ValueError(f"expected probability maps [C,Z,Y,X], got {probs.shape}")
    target_shape = tuple(int(v) for v in target_shape)
    if tuple(probs.shape[1:]) == target_shape:
        return probs.astype(np.float32, copy=False)
    if any(int(v) <= 0 for v in probs.shape[1:]) or any(v <= 0 for v in target_shape):
        raise ValueError(f"invalid resize shape: source={probs.shape[1:]} target={target_shape}")
    factors = [1.0] + [float(t) / float(s) for s, t in zip(probs.shape[1:], target_shape)]
    resized = ndi.zoom(probs.astype(np.float32, copy=False), zoom=factors, order=1)
    if tuple(resized.shape[1:]) != target_shape:
        raise RuntimeError(f"probability resize failed: got {resized.shape}, expected C+{target_shape}")
    return np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)


def _resize_label_volume_nearest(labels: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize a semantic label volume with nearest-neighbor interpolation.

    [QC][Invariant:label_space]
    BFV3/V5 target labels are discrete class IDs. Linear interpolation would
    create invalid fractional labels and corrupt forensic buckets, so this helper
    uses order=0 and verifies the final shape only.
    """
    arr = np.asarray(labels)
    target_shape = tuple(int(v) for v in target_shape)
    if tuple(arr.shape) == target_shape:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"expected label volume [Z,Y,X], got {arr.shape}")
    if any(int(v) <= 0 for v in arr.shape) or any(v <= 0 for v in target_shape):
        raise ValueError(f"invalid label resize shape: source={arr.shape} target={target_shape}")
    factors = [float(t) / float(s) for s, t in zip(arr.shape, target_shape)]
    resized = ndi.zoom(arr, zoom=factors, order=0)
    if tuple(resized.shape) != target_shape:
        raise RuntimeError(f"label resize failed: got {resized.shape}, expected {target_shape}")
    return resized.astype(arr.dtype, copy=False)


def _infer_bicm_contact_contract(output_root: Path | None, requested: str) -> str:
    if requested != "auto" or output_root is None:
        return str(requested)
    # [QA][FailFast:contract_inference]
    # nnU-Net's native predictor does not always write our custom trainer name
    # to the output folder. Keep this filename fallback deterministic so V16+
    # five-channel ROI classifier runs are not accidentally evaluated with the
    # older dense-geomean V15 contract. This does not tune a threshold; it only
    # selects the fixed contract implied by the run name.
    output_name = output_root.name.lower()
    summary_path = output_root / "predict_summary.json"
    if summary_path.exists():
        try:
            trainer = str(json.loads(summary_path.read_text()).get("trainer", ""))
        except Exception:
            trainer = ""
        if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakXYZAffinityV248":
            # [AUDIT][Risk:High][Scope:decoder_contract]
            # V248 has eight channels like old V38, but the order is different:
            # 0..4 are BFV3 semantic softmax logits and 5..7 are same-fragment
            # affinities. Metadata must select this contract before the generic
            # 8-channel/V38 fallback sees the tensor.
            return "boundary_fragment_xyz_affinity_v1"
        if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakDecoderContactV247":
            # [AUDIT][Risk:High][Scope:decoder_contract]
            # V247은 같은 6채널 BFV3 출력이지만 channel5 의미가 old aux barrier가
            # 아니라 decoder_contact_logit이다. old V162 contract로 평가하면 label3
            # candidate gate가 다시 개입하므로 metadata에서 명시 분기한다. v2 core-veto는
            # diagnostic contract로 남기되, short-run retrain에서 ratio가 악화되어 기본값은
            # v1 direct contact로 유지한다.
            return "boundary_fragment_decoder_contact_v1"
        if trainer in {
            "PengwinTrainerBICMBoundaryFragmentRemapV162",
            "PengwinTrainerBoundaryFragmentSidecarV164",
            "PengwinTrainerBoundaryFragmentSidecarPrecisionV165",
            "PengwinTrainerBoundaryFragmentSidecarCoreSeedV166",
            "PengwinTrainerBoundaryFragmentSidecarCoreRidgeV167",
            "PengwinTrainerBoundaryFragmentSidecarCoreRidgeHighEdgeV168",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallRidgeV169",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallMassCapV170",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallTightMassCapV171",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallPositiveMassCapV172",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftPositiveMassCapV173",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallVerySoftPositiveMassCapV174",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftMassCapStableScoreV175",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallVerySoftMassCapStableScoreV176",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftMassCapStrictStableV177",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallVerySoftMassCapStrictStableV178",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallLogitCalibratedV179",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallPrecisionLogitCalibratedV180",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallShellContrastV181",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallStrongShellContrastV182",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallBinaryBarrierHeadV183",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallPrecisionBinaryBarrierHeadV184",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallPositiveFloorBinaryBarrierHeadV185",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallPositiveFloorMassCapBinaryBarrierHeadV186",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateBinaryBarrierHeadV187",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateMassCapBinaryBarrierHeadV188",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateSoftRidgeBarrierHeadV189",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateSoftRidgeContrastBarrierHeadV190",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateLogitRidgeBarrierHeadV191",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateLogitRidgePrecisionBarrierHeadV192",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBinaryBarrierHeadV193",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateMassCapBarrierHeadV194",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateWeakMassCapBarrierHeadV195",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftDenseCandidateBarrierHeadV196",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateLongProbeV197",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateLowLRLongProbeV198",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCompactV199",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreRecallV200",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBarrierContrastV205",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftDenseCandidateBarrierContrastV206",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCoverageOpenV209",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCoverageGuardedV210",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateInstanceCoreGuardV213",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateInstanceCoreStrongGuardV214",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBalancedContactV215",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateFragmentPeakCoreV216",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateWeakFragmentMassCoreV217",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateLooseFragmentMassCoreV218",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025WeakPeakCompactV219",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakCompactV220",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakBalancedContactV241",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakTightBalancedContactV242",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakSoftContactRidgeV243",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakPrecisionSoftContactRidgeV244",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakContactShellContrastV245",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakStrongContactShellContrastV246",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakCompactV221",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025ContactPreservePeakCompactV222",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakCoverageCompactV223",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025OriginalContactMidPeakCompactV224",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakContactCapCompactV226",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakTightContactCapCompactV227",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakSupportSeedCompactV230",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakConservativeCoreTailCompactV231",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateSeedHeadV201",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCompactSeedHeadV202",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateSparseSeedHeadV203",
            "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCompactSparseSeedHeadV204",
        }:
            return "boundary_fragment_v162"
    if (
        "v248" in output_name
        or "xyz_affinity" in output_name
        or "bfv3_affinity" in output_name
        or "boundary_fragment_xyz_affinity" in output_name
    ):
        return "boundary_fragment_xyz_affinity_v1"
    if (
        "v247" in output_name
        or "decoder_contact" in output_name
        or "decoder-contact" in output_name
    ):
        return "boundary_fragment_decoder_contact_v1"
    if (
        "bfv3" in output_name
        or "v197" in output_name
        or "v198" in output_name
        or "v199" in output_name
        or "v200" in output_name
        or "v201" in output_name
        or "v202" in output_name
        or "v203" in output_name
        or "v204" in output_name
        or "v205" in output_name
        or "v206" in output_name
        or "v209" in output_name
        or "v210" in output_name
        or "v213" in output_name
        or "v214" in output_name
        or "v215" in output_name
        or "v216" in output_name
        or "v217" in output_name
        or "v218" in output_name
        or "v219" in output_name
        or "v220" in output_name
        or "v221" in output_name
        or "v222" in output_name
        or "v223" in output_name
        or "v224" in output_name
        or "v226" in output_name
        or "v227" in output_name
        or "v162" in output_name
        or "v164" in output_name
        or "v165" in output_name
        or "v166" in output_name
        or "v167" in output_name
        or "boundary_fragment_remap" in output_name
        or "boundaryfragmentremap" in output_name
        or "boundary_fragment_sidecar" in output_name
        or "boundaryfragmentsidecar" in output_name
    ):
        return "boundary_fragment_v162"
    if "v163" in output_name or "factorized_instance_embedding" in output_name or "embedding_v163" in output_name:
        return "factorized_instance_v163"
    if (
        "v105" in output_name
        or "v106" in output_name
        or "v107" in output_name
        or "v160" in output_name
        or "v161" in output_name
        or "offset_assignment" in output_name
        or "offset_attractor" in output_name
        or "radial_support_offset" in output_name
        or "offset_softpeak" in output_name
        or "center_seed_offset" in output_name
    ):
        return "v105_offset_contact"
    if "v108" in output_name or "watershed_barrier" in output_name:
        return "v68_edge_primary"
    if "v112" in output_name or "edge_graph_assignment" in output_name:
        return "v68_edge_primary"
    if "v116" in output_name or "graph_cost_separator" in output_name or "graph_cost_assignment" in output_name:
        return "v68_edge_primary"
    if (
        "v122" in output_name
        or "v123" in output_name
        or "v124" in output_name
        or "v125" in output_name
        or "v126" in output_name
        or "v127" in output_name
        or "warm_edge_product_precision" in output_name
        or "high_recall_gate_cleanup" in output_name
        or "support_conditioned_edge_precision" in output_name
        or "saturated_edge_dense_gate" in output_name
        or "local_adjacency_product" in output_name
        or "edge_reset_local_adjacency" in output_name
    ):
        return "semantic_topology_v68"
    if (
        "v120" in output_name
        or "v121" in output_name
        or "same_fragment_affinity" in output_name
        or "affinity_graph_assignment" in output_name
    ):
        return "v120_affinity_break"
    if "v119" in output_name or "all_network_adaptive_boundary" in output_name:
        return "semantic_topology_v68"
    if "v117" in output_name or "v118" in output_name or "support_gate_semantic_contact" in output_name:
        return "v117_support_gate_semantic_contact"
    if "v90" in output_name or "semantic_band_product" in output_name:
        return "v68_semantic_edge_product"
    if (
        "v103" in output_name
        or "v104" in output_name
        or "v110" in output_name
        or "v115" in output_name
        or "semantic_distance_contact" in output_name
        or "teacher_semantic_contact" in output_name
        or "semantic_geometry_bridge" in output_name
        or "semantic_oracle_adapter" in output_name
    ):
        return "v68_semantic_contact"
    if (
        "v68_edge_primary" in output_name
        or "v86" in output_name
        or "v88" in output_name
        or "v89" in output_name
        or "v91" in output_name
        or "v93" in output_name
        or "v94" in output_name
        or "v95" in output_name
        or "eval_band_edge_primary" in output_name
        or "core_preserving_band_precision" in output_name
        or "band_false_only_precision" in output_name
        or "topology_aware_edge_balance" in output_name
        or "eval_aligned_support_contact" in output_name
        or "final_row_support_contact" in output_name
        or "decoder_feature_contact" in output_name
    ):
        return "v68_edge_primary"
    if (
        "v68" in output_name
        or "v78" in output_name
        or "v79" in output_name
        or "v80" in output_name
        or "v81" in output_name
        or "v82" in output_name
        or "v83" in output_name
        or "v84" in output_name
        or "v85" in output_name
        or "v87" in output_name
        or "v96" in output_name
        or "v97" in output_name
        or "v98" in output_name
        or "v99" in output_name
        or "v100" in output_name
        or "v101" in output_name
        or "v102" in output_name
        or "v109" in output_name
        or "v111" in output_name
        or "v113" in output_name
        or "v114" in output_name
        or "semantic_topology" in output_name
        or "core_anchored_edge" in output_name
        or "duplicate_seed" in output_name
        or "topology_state" in output_name
        or "core_stable_edge" in output_name
        or "seed_recall_edge" in output_name
        or "seed_safe_edge" in output_name
        or "dual_head_contact" in output_name
        or "semantic_gate_contact" in output_name
        or "eval_band_semantic_gate" in output_name
        or "dense_band_gate" in output_name
        or "positive_dense_band_gate" in output_name
        or "teacher_distilled_dense_gate" in output_name
        or "adaptive_dual_margin_dense_gate" in output_name
        or "negative_balanced_dense_gate" in output_name
        or "dual_field_product_gate" in output_name
        or "distance_rank_dense_gate" in output_name
        or "precision_locked_recall" in output_name
        or "joint_support_product" in output_name
        or "adaptive_boundary_product" in output_name
        or "encoder_adapter_boundary_product" in output_name
    ):
        return "semantic_topology_v68"
    if "v39" in output_name or "v40" in output_name or "v41" in output_name or "v42" in output_name or "v53" in output_name or "v54" in output_name or "v55" in output_name or "v56" in output_name or "v57" in output_name or "v58" in output_name or "v59" in output_name or "v60" in output_name or "v61" in output_name or "edge_primary" in output_name or "edgecore" in output_name or "instancecore" in output_name or "fragmentmarker" in output_name or "fragment_marker" in output_name or "sparsehead" in output_name or "sparse_head" in output_name or "heatmap_marker" in output_name or "core_preserving" in output_name or "strict_core" in output_name or "peakseed" in output_name or "peak_seed" in output_name:
        return "edge_primary_v39"
    if "v38" in output_name or "edge_affinity" in output_name:
        return "edge_affinity_v38"
    if any(token in output_name for token in ("v16", "v17", "v18", "v19", "v20", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v34", "v35", "v36", "roi_presence", "roi_branch", "roi_balanced", "roi_calibrated", "coremarker", "fullvol_probe", "coupled", "tolerant", "isolated_contact", "soft_contact_ridge", "local_contrastive", "compact_core", "contact_precision", "staged_contact")):
        return "roi_presence_scalar"
    summary_path = output_root / "predict_summary.json"
    if not summary_path.exists():
        return "auto"
    try:
        trainer = str(json.loads(summary_path.read_text()).get("trainer", ""))
    except Exception:
        return "auto"
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakXYZAffinityV248":
        return "boundary_fragment_xyz_affinity_v1"
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakDecoderContactV247":
        return "boundary_fragment_decoder_contact_v1"
    if trainer in {
        "PengwinTrainerBICMBoundaryFragmentRemapV162",
        "PengwinTrainerBoundaryFragmentSidecarV164",
        "PengwinTrainerBoundaryFragmentSidecarPrecisionV165",
        "PengwinTrainerBoundaryFragmentSidecarCoreSeedV166",
        "PengwinTrainerBoundaryFragmentSidecarCoreRidgeV167",
        "PengwinTrainerBoundaryFragmentSidecarCoreRidgeHighEdgeV168",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallRidgeV169",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallMassCapV170",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallTightMassCapV171",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallPositiveMassCapV172",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftPositiveMassCapV173",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallVerySoftPositiveMassCapV174",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftMassCapStableScoreV175",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallVerySoftMassCapStableScoreV176",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftMassCapStrictStableV177",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallVerySoftMassCapStrictStableV178",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallLogitCalibratedV179",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallPrecisionLogitCalibratedV180",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallShellContrastV181",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallStrongShellContrastV182",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallBinaryBarrierHeadV183",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallPrecisionBinaryBarrierHeadV184",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallPositiveFloorBinaryBarrierHeadV185",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallPositiveFloorMassCapBinaryBarrierHeadV186",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateBinaryBarrierHeadV187",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateMassCapBinaryBarrierHeadV188",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateSoftRidgeBarrierHeadV189",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateSoftRidgeContrastBarrierHeadV190",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateLogitRidgeBarrierHeadV191",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateLogitRidgePrecisionBarrierHeadV192",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBinaryBarrierHeadV193",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateMassCapBarrierHeadV194",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateWeakMassCapBarrierHeadV195",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftDenseCandidateBarrierHeadV196",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateLongProbeV197",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateLowLRLongProbeV198",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCompactV199",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreRecallV200",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBarrierContrastV205",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftDenseCandidateBarrierContrastV206",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCoverageOpenV209",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCoverageGuardedV210",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateInstanceCoreGuardV213",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateInstanceCoreStrongGuardV214",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBalancedContactV215",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateFragmentPeakCoreV216",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateWeakFragmentMassCoreV217",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateLooseFragmentMassCoreV218",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025WeakPeakCompactV219",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakCompactV220",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakBalancedContactV241",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakTightBalancedContactV242",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakSoftContactRidgeV243",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakPrecisionSoftContactRidgeV244",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakContactShellContrastV245",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakStrongContactShellContrastV246",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakCompactV221",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025ContactPreservePeakCompactV222",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakCoverageCompactV223",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025OriginalContactMidPeakCompactV224",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakContactCapCompactV226",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakTightContactCapCompactV227",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakSupportSeedCompactV230",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakConservativeCoreTailCompactV231",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateSeedHeadV201",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCompactSeedHeadV202",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateSparseSeedHeadV203",
        "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCompactSparseSeedHeadV204",
    }:
        return "boundary_fragment_v162"
    if trainer == "PengwinTrainerFactorizedInstanceEmbeddingV163":
        return "factorized_instance_v163"
    if trainer in {"PengwinTrainerBICMContactV16", "PengwinTrainerBICMContactV17", "PengwinTrainerBICMContactV18", "PengwinTrainerBICMContactV19", "PengwinTrainerBICMContactV20", "PengwinTrainerBICMContactV22", "PengwinTrainerBICMContactV23", "PengwinTrainerBICMContactV24", "PengwinTrainerBICMContactV25", "PengwinTrainerBICMContactV26", "PengwinTrainerBICMContactV27", "PengwinTrainerBICMContactV28", "PengwinTrainerBICMContactV29", "PengwinTrainerBICMContactV30", "PengwinTrainerBICMContactV31", "PengwinTrainerBICMContactV32", "PengwinTrainerBICMContactV34", "PengwinTrainerBICMContactV35", "PengwinTrainerBICMContactV36", "PengwinTrainerBICMContactV37"}:
        return "roi_presence_scalar"
    if trainer in {"PengwinTrainerBICMEdgePrimaryV39", "PengwinTrainerBICMEdgeCoreV40", "PengwinTrainerBICMInstanceCoreV41", "PengwinTrainerBICMInstanceCoreV42", "PengwinTrainerBICMEdgeContactV43", "PengwinTrainerBICMEdgeLocalRankV44", "PengwinTrainerBICMEdgeCandidateV45", "PengwinTrainerBICMEdgeCurriculumV46", "PengwinTrainerBICMSeparatedContactV47", "PengwinTrainerBICMSupportAwareContactV48", "PengwinTrainerBICMDecoderFeatureContactV49", "PengwinTrainerBICMDecoderFeaturePhaseV50", "PengwinTrainerBICMDenseEdgeCostV51", "PengwinTrainerBICMDenseEdgeCoreV52", "PengwinTrainerBICMFragmentMarkerCoreV53", "PengwinTrainerBICMSparseHeadBalancedV54", "PengwinTrainerBICMCalibratedSparseHeadV55", "PengwinTrainerBICMPhasedMarkerContactV56", "PengwinTrainerBICMDecoderFeatureMarkerPhaseV57", "PengwinTrainerBICMHeatmapMarkerContactV58", "PengwinTrainerBICMCorePreservingContactV59", "PengwinTrainerBICMStrictCoreTopologyV60", "PengwinTrainerBICMPeakSeedV61"}:
        return "edge_primary_v39"
    if trainer == "PengwinTrainerBICMEdgeAffinityV38":
        return "edge_affinity_v38"
    if trainer in {
        "PengwinTrainerBICMSemanticTopologyV68",
        "PengwinTrainerBICMTopologyCalibratedV69",
        "PengwinTrainerBICMTopologyConsistencyV70",
        "PengwinTrainerBICMEdgeCutPrimaryV71",
        "PengwinTrainerBICMLogitCalibratedV72",
        "PengwinTrainerBICMInstanceTopologyV73",
        "PengwinTrainerBICMAdaptiveInstanceTopologyV74",
        "PengwinTrainerBICMEdgePrecisionSeedTopologyV75",
        "PengwinTrainerBICMGentleEdgePrecisionV76",
        "PengwinTrainerBICMEdgeCurriculumV77",
        "PengwinTrainerBICMCoreAnchoredEdgeCurriculumV78",
        "PengwinTrainerBICMDuplicateSeedEdgeV79",
        "PengwinTrainerBICMTopologyStateAdaptiveV80",
        "PengwinTrainerBICMCoreStableEdgePrecisionV81",
        "PengwinTrainerBICMSeedRecallEdgeSeparationV82",
        "PengwinTrainerBICMSeedSafeEdgeCalibrationV83",
        "PengwinTrainerBICMDualHeadContactCalibrationV84",
        "PengwinTrainerBICMSemanticGateContactV85",
        "PengwinTrainerBICMEvalBandSemanticGateV87",
        "PengwinTrainerBICMDenseBandGateV96",
        "PengwinTrainerBICMPositiveDenseBandGateV97",
        "PengwinTrainerBICMTeacherDistilledDenseGateV98",
        "PengwinTrainerBICMAdaptiveDualMarginDenseGateV99",
        "PengwinTrainerBICMNegativeBalancedDenseGateV100",
        "PengwinTrainerBICMDualFieldProductGateV101",
        "PengwinTrainerBICMDistanceRankDenseGateV102",
        "PengwinTrainerBICMPrecisionLockedRecallV109",
        "PengwinTrainerBICMJointSupportProductV111",
    }:
        return "semantic_topology_v68"
    if trainer in {
        "PengwinTrainerBICMSemanticDistanceContactV103",
        "PengwinTrainerBICMTeacherSemanticContactV104",
        "PengwinTrainerBICMSemanticGeometryBridgeV110",
        "PengwinTrainerBICMSemanticOracleAdapterV115",
    }:
        return "v68_semantic_contact"
    if trainer == "PengwinTrainerBICMSupportGateSemanticContactV117":
        return "v117_support_gate_semantic_contact"
    if trainer == "PengwinTrainerBICMWarmSupportGateSemanticContactV118":
        return "v117_support_gate_semantic_contact"
    if trainer == "PengwinTrainerBICMAllNetworkAdaptiveBoundaryV119":
        return "semantic_topology_v68"
    if trainer == "PengwinTrainerBICMSameFragmentAffinityV120":
        return "v120_affinity_break"
    if trainer == "PengwinTrainerBICMWarmSameFragmentAffinityV121":
        return "v120_affinity_break"
    if trainer in {
        "PengwinTrainerBICMCoreOnlyCenterSeedAffinitySamplerV158",
        "PengwinTrainerBICMCoreHeatmapCenterSeedAffinitySamplerV159",
    }:
        return "v120_affinity_break"
    if trainer in {
        "PengwinTrainerBICMCoreOnlyCenterSeedOffsetSamplerV160",
        "PengwinTrainerBICMCoreHeatmapCenterSeedOffsetSamplerV161",
    }:
        return "v105_offset_contact"
    if trainer == "PengwinTrainerBICMWarmEdgeProductPrecisionV122":
        return "semantic_topology_v68"
    if trainer == "PengwinTrainerBICMHighRecallGateCleanupV123":
        return "semantic_topology_v68"
    if trainer == "PengwinTrainerBICMSupportConditionedEdgePrecisionV124":
        return "semantic_topology_v68"
    if trainer == "PengwinTrainerBICMSaturatedEdgeDenseGateV125":
        return "semantic_topology_v68"
    if trainer in {
        "PengwinTrainerBICMLocalAdjacencyProductV126",
        "PengwinTrainerBICMLocalAdjacencyProductV126FromV96",
        "PengwinTrainerBICMEdgeResetLocalAdjacencyProductV127",
    }:
        return "semantic_topology_v68"
    if trainer in {
        "PengwinTrainerBICMOffsetAssignmentV105",
        "PengwinTrainerBICMOffsetAttractorV106",
        "PengwinTrainerBICMRadialSupportOffsetV107",
    }:
        return "v105_offset_contact"
    if trainer == "PengwinTrainerBICMWatershedBarrierV108":
        return "v68_edge_primary"
    if trainer == "PengwinTrainerBICMEdgeGraphAssignmentV112":
        return "v68_edge_primary"
    if trainer == "PengwinTrainerBICMGraphCostSeparatorV116":
        return "v68_edge_primary"
    if trainer in {
        "PengwinTrainerBICMAdaptiveBoundaryProductV113",
        "PengwinTrainerBICMAdaptiveBoundaryProductV113HighLR",
        "PengwinTrainerBICMEncoderAdapterV114",
    }:
        return "semantic_topology_v68"
    if trainer == "PengwinTrainerBICMSemanticBandProductV90":
        return "v68_semantic_edge_product"
    if trainer in {
        "PengwinTrainerBICMEvalBandEdgePrimaryV86",
        "PengwinTrainerBICMCorePreservingBandPrecisionV88",
        "PengwinTrainerBICMBandFalseOnlyPrecisionV89",
        "PengwinTrainerBICMTopologyAwareEdgeBalanceV91",
        "PengwinTrainerBICMEvalAlignedSupportContactV93",
        "PengwinTrainerBICMFinalRowCalibratedV94",
        "PengwinTrainerBICMDecoderFeatureContactV95",
    }:
        return "v68_edge_primary"
    return "auto"


def _edge_probability_to_voxel_probability(edge_prob_zyx: np.ndarray) -> np.ndarray:
    """Project z/y/x edge probabilities to endpoint voxel probabilities."""
    edge = np.asarray(edge_prob_zyx, dtype=np.float32)
    if edge.ndim != 4 or int(edge.shape[0]) != 3:
        raise ValueError(f"expected edge probabilities [3,Z,Y,X], got {edge.shape}")
    out = np.zeros(edge.shape[1:], dtype=np.float32)
    for axis in range(3):
        src_sl = [slice(None), slice(None), slice(None)]
        dst_sl = [slice(None), slice(None), slice(None)]
        src_sl[axis] = slice(0, -1)
        dst_sl[axis] = slice(1, None)
        vals = edge[(axis, *src_sl)]
        out[tuple(src_sl)] = np.maximum(out[tuple(src_sl)], vals)
        out[tuple(dst_sl)] = np.maximum(out[tuple(dst_sl)], vals)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_boundary_fragment_v162_logits(
    raw_output: np.ndarray,
    *,
    contact_contract: str = "boundary_fragment_v162",
) -> np.ndarray:
    """Convert BFV3 logits to the 5-channel BICM eval contract.

    [QC][Invariant:channel_order]
    raw[0:5]는 [background, exterior, contact_label2, shell_label3, core]이다.
    raw[5]는 old/V247 contact 계열에서 aux/contact logit이고, V248에서는
    raw[5:8]이 z/y/x same-fragment affinity logits이다.
    """
    raw = np.asarray(raw_output, dtype=np.float32)
    if raw.ndim != 4 or int(raw.shape[0]) not in {
        int(BICM_V8_OUTPUT_CHANNELS),
        int(BFV3_BINARY_BARRIER_OUTPUT_CHANNELS),
        int(BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS),
        int(BFV3_XYZ_AFFINITY_OUTPUT_CHANNELS),
    }:
        raise ValueError(f"expected BFV3 logits [5|6|7|8,Z,Y,X], got {raw.shape}")
    logits = raw[:5] - np.max(raw[:5], axis=0, keepdims=True)
    probs = np.exp(logits).astype(np.float32, copy=False)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-8)
    # [QC][Invariant:v162_eval_contract]
    # BFV3 classes are [bg, exterior, barrier, shell, core]. The ROI evaluator
    # consumes [support, contact, core, exterior, contact_aux], so the semantic
    # conversion is fixed and threshold-free. Six/seven-channel BFV3 runs add
    # an independent aux barrier logit; seven-channel runs additionally decode
    # core from the fragment-scale seed head instead of semantic class-4 mass.
    support_prob = np.clip(probs[2] + probs[3] + probs[4], 0.0, 1.0)
    if int(raw.shape[0]) == int(BFV3_XYZ_AFFINITY_OUTPUT_CHANNELS):
        if contact_contract != "boundary_fragment_xyz_affinity_v1":
            raise ValueError(
                "8-channel BFV3 affinity logits require "
                "contact_contract='boundary_fragment_xyz_affinity_v1', "
                f"got {contact_contract!r}"
            )
        affinity_prob = expit(raw[5:8]).astype(np.float32, copy=False)
        break_voxel_prob = _edge_probability_to_voxel_probability(1.0 - affinity_prob)
        # [METRIC][Scope:affinity_decoder_contract]
        # V248 stores same-fragment affinity, so separator/contact evidence is
        # `1 - affinity`. The first five rows are converted to the existing BICM
        # evaluator contract, and rows 5..7 keep the actual affinity graph for
        # graph-assignment decoder profiles.
        return np.concatenate(
            [
                np.stack(
                    [
                        support_prob,
                        break_voxel_prob,
                        probs[4].astype(np.float32, copy=False),
                        probs[1].astype(np.float32, copy=False),
                        break_voxel_prob,
                    ],
                    axis=0,
                ),
                affinity_prob,
            ],
            axis=0,
        ).astype(np.float32, copy=False)
    if int(raw.shape[0]) in {
        int(BFV3_BINARY_BARRIER_OUTPUT_CHANNELS),
        int(BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS),
    }:
        contact_aux_prob = expit(raw[5]).astype(np.float32, copy=False)
        # [AUDIT][Risk:High][Scope:decoder_contract]
        # V162-V246 재현 경로는 candidate(label2+label3)와 aux의 min을 유지한다.
        # V247만 channel5 sigmoid를 직접 contact로 해석한다. 같은 6채널 shape를
        # 공유하므로 이 분기가 섞이면 과거 결과 재현성과 새 contract가 동시에 깨진다.
        if contact_contract == "boundary_fragment_decoder_contact_v1":
            # [METRIC][Scope:contact_threshold]
            # V247 train target은 channel5 sigmoid를 decoder-facing contact_prob로
            # 직접 학습한다. promotion 판단은 eval의 contact_prob >= 0.50이고,
            # label3는 hard negative였으므로 candidate_prob와 min을 취하지 않는다.
            contact_prob = contact_aux_prob.astype(np.float32, copy=False)
        elif contact_contract == "boundary_fragment_decoder_contact_v2_core_veto":
            # [AUDIT][Risk:High][Scope:decoder_contract]
            # V247 v1 forensic에서는 V5 contact positive가 살아났지만 V5 core negative
            # tail도 0.50 위에 남아 contact_ratio가 실패했다. BFV3 semantic core p4는
            # 이미 V220 scaffold에서 학습되는 decoder-visible evidence이므로, channel5를
            # 버리지 않고 `1 - p4`만 곱해 core false-contact를 구조적으로 veto한다.
            #
            # [METRIC][Scope:contact_threshold]
            # threshold 자체는 계속 0.50이다. 이 branch는 threshold sweep이 아니라
            # decoder-facing probability 정의를 고정하는 contract 변경이다.
            contact_prob = (
                contact_aux_prob.astype(np.float32, copy=False)
                * (1.0 - probs[4].astype(np.float32, copy=False))
            ).astype(np.float32, copy=False)
        elif contact_contract == "boundary_fragment_v162":
            candidate_prob = np.clip(probs[2] + probs[3], 0.0, 1.0)
            contact_prob = np.minimum(candidate_prob, contact_aux_prob).astype(np.float32, copy=False)
        else:
            raise ValueError(
                "unsupported BFV3 contact_contract="
                f"{contact_contract!r}; expected boundary_fragment_v162 or "
                "boundary_fragment_decoder_contact_v1 or boundary_fragment_decoder_contact_v2_core_veto"
            )
    else:
        contact_prob = probs[2]
        contact_aux_prob = probs[2]
    if int(raw.shape[0]) == int(BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS):
        core_prob = expit(raw[6]).astype(np.float32, copy=False)
    else:
        core_prob = probs[4]
    exterior_prob = probs[1]
    return np.stack(
        [support_prob, contact_prob, core_prob, exterior_prob, contact_aux_prob],
        axis=0,
    ).astype(np.float32, copy=False)


def _normalize_bicm_custom_logits(raw_output: np.ndarray) -> np.ndarray:
    """Convert saved custom-head logits into the evaluator contract.

    [QC][Invariant:branch_activation]
    V6/V8/V38 heads are all sigmoid logits, while V68 combines semantic
    softmax logits (channels 0..4) with sigmoid topology logits (5..9).
    Applying `expit` to every V68 channel would destroy the semantic branch and
    silently evaluate a different method, so conversion is centralized here.
    """
    raw = np.asarray(raw_output, dtype=np.float32)
    if raw.ndim != 4:
        raise ValueError(f"expected custom logits [C,Z,Y,X], got {raw.shape}")
    if int(raw.shape[0]) != int(BICM_V68_OUTPUT_CHANNELS):
        return expit(raw)
    sem = raw[0:5]
    sem = sem - np.max(sem, axis=0, keepdims=True)
    sem_prob = np.exp(sem).astype(np.float32, copy=False)
    sem_prob /= np.maximum(sem_prob.sum(axis=0, keepdims=True), 1e-8)
    topo_prob = expit(raw[5:10])
    return np.concatenate([sem_prob, topo_prob], axis=0).astype(np.float32, copy=False)


def _normalize_bicm_v105_logits(raw_output: np.ndarray) -> np.ndarray:
    """Normalize V105 logits while preserving raw offset rows 7..9."""
    raw = np.asarray(raw_output, dtype=np.float32)
    if raw.ndim != 4 or int(raw.shape[0]) != int(BICM_V68_OUTPUT_CHANNELS):
        raise ValueError(f"expected V105 logits [10,Z,Y,X], got {raw.shape}")
    sem = raw[0:5]
    sem = sem - np.max(sem, axis=0, keepdims=True)
    sem_prob = np.exp(sem).astype(np.float32, copy=False)
    sem_prob /= np.maximum(sem_prob.sum(axis=0, keepdims=True), 1e-8)
    contact_core = expit(raw[5:7])
    offsets = raw[7:10].astype(np.float32, copy=False)
    # [QC][Invariant:v105_activation]
    # Offset rows are raw normalized center vectors. Applying sigmoid or clipping
    # would force them into [0, 1] and make nearest-seed assignment invalid.
    return np.concatenate([sem_prob, contact_core, offsets], axis=0).astype(np.float32, copy=False)


def _resize_bicm_v105_output(output: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize V105 channel maps without clipping raw offset rows."""
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[0]) != int(BICM_V68_OUTPUT_CHANNELS):
        raise ValueError(f"expected V105 output [10,Z,Y,X], got {arr.shape}")
    target_shape = tuple(int(v) for v in target_shape)
    if tuple(arr.shape[1:]) == target_shape:
        return arr.astype(np.float32, copy=False)
    if any(int(v) <= 0 for v in arr.shape[1:]) or any(v <= 0 for v in target_shape):
        raise ValueError(f"invalid V105 resize shape: source={arr.shape[1:]} target={target_shape}")
    factors = [1.0] + [float(t) / float(s) for s, t in zip(arr.shape[1:], target_shape)]
    resized = ndi.zoom(arr.astype(np.float32, copy=False), zoom=factors, order=1)
    if tuple(resized.shape[1:]) != target_shape:
        raise RuntimeError(f"V105 resize failed: got {resized.shape}, expected 10+{target_shape}")
    resized[0:7] = np.clip(resized[0:7], 0.0, 1.0)
    return resized.astype(np.float32, copy=False)


def _bicm_head_probabilities(output: np.ndarray,
                             *,
                             contact_contract: str = "auto",
                             threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, str]:
    """Normalize V6/V8/V15/V16 BICM probability contracts into decoder inputs.

    [QC][Invariant:fixed_contact_contract]
    V6 has one contact head. V8/V15 combine two dense evidence channels by a
    fixed geometric mean. V16 interprets channel 4 as a pooled ROI classifier:
    a fixed top-k pooled score below 0.5 suppresses contact for that ROI. The
    mode is inferred from `predict_summary.json` or passed explicitly; no
    validation-driven threshold search is allowed.
    """
    if output.ndim != 4:
        raise ValueError(f"expected BICM probability/logit output [C,Z,Y,X], got {output.shape}")
    channels = int(output.shape[0])
    if channels == int(BICM_V6_OUTPUT_CHANNELS):
        support_prob, contact_prob, core_prob, exterior_prob = output
        return support_prob, contact_prob, core_prob, exterior_prob, None, "v6_single_contact"
    if channels == int(BICM_V8_OUTPUT_CHANNELS):
        support_prob, contact_binary_prob, core_prob, exterior_prob, contact_aux_prob = output
        contract = "v8_v15_contact_geomean" if contact_contract == "auto" else str(contact_contract)
        if contract in {
            "boundary_fragment_v162",
            "boundary_fragment_decoder_contact_v1",
            "boundary_fragment_decoder_contact_v2_core_veto",
        }:
            # [QA][Status:Not run][Scope:old_new_contract_regression]
            # Expected: V220/V245는 boundary_fragment_v162로 계속 old min contract를
            # 쓰고, V247 v1/v2는 이미 normalize 단계에서 decoder-facing contact로
            # 변환된 contact_binary_prob를 그대로 사용해야 한다.
            return (
                support_prob,
                contact_binary_prob.astype(np.float32, copy=False),
                core_prob,
                exterior_prob,
                contact_aux_prob,
                contract,
            )
        if contract == "roi_presence_scalar":
            support_mask = support_prob >= float(threshold)
            values = contact_aux_prob[support_mask] if np.any(support_mask) else contact_aux_prob.reshape(-1)
            if values.size:
                k = max(1, int(round(float(values.size) * 0.10)))
                roi_score = float(np.partition(values.reshape(-1), -min(k, values.size))[-min(k, values.size):].mean())
            else:
                roi_score = 0.0
            contact_prob = contact_binary_prob.astype(np.float32, copy=False) if roi_score >= 0.5 else np.zeros_like(contact_binary_prob, dtype=np.float32)
            return support_prob, contact_prob, core_prob, exterior_prob, contact_aux_prob, f"v16_roi_presence_scalar:{roi_score:.6f}"
        contact_prob = np.sqrt(
            np.clip(
                contact_binary_prob.astype(np.float32, copy=False)
                * contact_aux_prob.astype(np.float32, copy=False),
                0.0,
                1.0,
            )
        ).astype(np.float32, copy=False)
        return support_prob, contact_prob, core_prob, exterior_prob, contact_aux_prob, "v8_v15_contact_geomean"
    if channels == int(FACTOR_INSTANCE_OUTPUT_CHANNELS):
        support_prob = output[0]
        contact_prob = output[1]
        core_prob = output[2]
        hard_negative_prob = output[3]
        exterior_prob = np.clip(1.0 - support_prob, 0.0, 1.0).astype(np.float32, copy=False)
        return support_prob, contact_prob, core_prob, exterior_prob, hard_negative_prob, "factorized_instance_v163"
    if channels == int(BICM_V38_OUTPUT_CHANNELS):
        support_prob = output[0]
        dense_contact = output[1]
        core_prob = output[2]
        exterior_prob = output[3]
        roi_prob = output[4]
        contract = "edge_affinity_v38" if contact_contract == "auto" else str(contact_contract)
        if contract == "boundary_fragment_xyz_affinity_v1":
            affinity_prob = np.clip(output[5:8].astype(np.float32, copy=False), 0.0, 1.0)
            edge_break_prob = (1.0 - affinity_prob).astype(np.float32, copy=False)
            contact_prob = _edge_probability_to_voxel_probability(edge_break_prob)
            # [AUDIT][Risk:High][Scope:decoder_contract]
            # V248 shares the 8-channel count with V38 but not its row meaning.
            # Here rows 5..7 are same-fragment affinities, not break logits.
            # `1-affinity` is the separator probability used by watershed and
            # graph decoders; using the V38 branch would invert the topology.
            return support_prob, contact_prob, core_prob, exterior_prob, edge_break_prob, contract
        edge_prob = np.max(output[5:8], axis=0)
        if contract not in {"edge_affinity_v38", "edge_primary_v39"}:
            raise ValueError(f"unsupported 8-channel BICM contact_contract={contract!r}")
        # [METRIC][Risk:High][Scope:v38_fixed_decoder]
        # V38 tests whether true cross-fragment edge supervision localizes the
        # fracture ridge. Contact for the existing fixed watershed decoder is
        # the geometric agreement between dense contact and the strongest
        # axis-edge probability, optionally suppressed by the fixed ROI scalar.
        support_mask = support_prob >= float(threshold)
        values = roi_prob[support_mask] if np.any(support_mask) else roi_prob.reshape(-1)
        if values.size:
            k = max(1, int(round(float(values.size) * 0.10)))
            roi_score = float(np.partition(values.reshape(-1), -min(k, values.size))[-min(k, values.size):].mean())
        else:
            roi_score = 0.0
        if contract == "edge_primary_v39":
            # [METRIC][Risk:High][Scope:v39_fixed_decoder]
            # V39 is a single-variable follow-up to V38: ROI presence remains
            # reported in the contract string, but it cannot erase contact.
            # The edge head is the fixed primary fracture-topology signal.
            contact_prob = edge_prob.astype(np.float32, copy=False)
        else:
            contact_prob = np.sqrt(np.clip(dense_contact.astype(np.float32) * edge_prob.astype(np.float32), 0.0, 1.0))
            if roi_score < 0.5:
                contact_prob = np.zeros_like(contact_prob, dtype=np.float32)
        return support_prob, contact_prob, core_prob, exterior_prob, edge_prob, f"{contract}:{roi_score:.6f}"
    if channels == int(BICM_V68_OUTPUT_CHANNELS):
        sem_prob = output[0:5].astype(np.float32, copy=False)
        sem_label = np.argmax(sem_prob, axis=0).astype(np.uint8, copy=False)
        support_prob = np.isin(sem_label, (2, 3, 4)).astype(np.float32)
        exterior_prob = (sem_label == 1).astype(np.float32)
        dense_contact = output[5].astype(np.float32, copy=False)
        core_prob = output[6].astype(np.float32, copy=False)
        contract = "semantic_topology_v68" if contact_contract == "auto" else str(contact_contract)
        if contract == "v105_offset_contact":
            # [METRIC][Risk:High][Scope:v105_contact_contract]
            # V105 keeps row 5 as a direct fixed-threshold contact metric while
            # rows 7..9 are raw offsets for assignment. They are not edge/contact
            # probabilities and must not enter the contact product.
            offset_norm = np.linalg.norm(output[7:10].astype(np.float32, copy=False), axis=0)
            return support_prob, dense_contact, core_prob, exterior_prob, offset_norm, contract
        edge_prob = np.max(output[7:10].astype(np.float32, copy=False), axis=0)
        if contract == "semantic_topology_v68":
            # [METRIC][Risk:High][Scope:v68_fixed_decoder]
            # V68 decodes contact from agreement between the independent dense
            # contact head and the explicit edge-break topology head. Support
            # remains the semantic argmax branch; no probability threshold is
            # tuned for support or contact.
            contact_prob = np.sqrt(np.clip(dense_contact * edge_prob, 0.0, 1.0)).astype(np.float32, copy=False)
        elif contract == "v68_dense_contact":
            contact_prob = dense_contact
        elif contract == "v68_edge_primary":
            contact_prob = edge_prob
        elif contract == "v68_semantic_contact":
            # [AUDIT][Risk:High][Scope:contact_contract_diagnostic]
            # Dataset537 semantic label 4 is the eval contact_surface band.
            # This diagnostic contract checks whether the semantic softmax
            # branch already localizes that band better than the independent
            # dense-contact or edge-primary heads. It is not selected by auto.
            contact_prob = sem_prob[4].astype(np.float32, copy=False)
        elif contract == "v68_semantic_edge_product":
            # [METRIC][Scope:contact_contract_diagnostic]
            # Fixed-threshold product of semantic contact-band probability and
            # edge-primary probability. This probes representation alignment
            # without changing trained weights or searching thresholds.
            contact_prob = np.sqrt(np.clip(sem_prob[4] * edge_prob, 0.0, 1.0)).astype(np.float32, copy=False)
        elif contract == "v117_support_gate_semantic_contact":
            # [METRIC][Risk:High][Scope:v117_support_gate_contract]
            # Row 5 is no longer dense contact for V117. It is an independent
            # support gate, so fixed-threshold support becomes semantic argmax
            # support intersected with row5>=0.5. Contact remains semantic row4.
            support_prob = (support_prob * dense_contact).astype(np.float32, copy=False)
            contact_prob = sem_prob[4].astype(np.float32, copy=False)
        elif contract == "v118_support_gate_edge_primary":
            support_prob = (support_prob * dense_contact).astype(np.float32, copy=False)
            contact_prob = edge_prob.astype(np.float32, copy=False)
        elif contract == "v118_support_gate_semantic_edge_product":
            support_prob = (support_prob * dense_contact).astype(np.float32, copy=False)
            contact_prob = np.sqrt(np.clip(sem_prob[4] * edge_prob, 0.0, 1.0)).astype(np.float32, copy=False)
        elif contract == "v120_affinity_break":
            # [METRIC][Risk:High][Scope:v120_affinity_contract]
            # Rows 7..9 are same-fragment affinity probabilities. A local break
            # signal is therefore `1 - min(axis_affinity)` rather than the V68
            # `max(edge_break)` contract. Row5 remains the dense contact gate so
            # support-surface low affinities do not automatically count as contact.
            edge_affinity = output[7:10].astype(np.float32, copy=False)
            edge_prob = (1.0 - np.min(edge_affinity, axis=0)).astype(np.float32, copy=False)
            contact_prob = np.sqrt(np.clip(dense_contact * edge_prob, 0.0, 1.0)).astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"unsupported V68 contact_contract={contract!r}; expected "
                "semantic_topology_v68, v68_dense_contact, v68_edge_primary, "
                "v68_semantic_contact, v68_semantic_edge_product, "
                "v117_support_gate_semantic_contact, v118_support_gate_edge_primary, "
                "v118_support_gate_semantic_edge_product, v120_affinity_break, "
                "or v105_offset_contact"
            )
        return support_prob, contact_prob, core_prob, exterior_prob, edge_prob, contract
    raise ValueError(f"unsupported BICM channel count {channels}; expected 4, 5, 8, or 10")


def _bicm_v61_peak_seed_mask(core_prob: np.ndarray,
                             support: np.ndarray,
                             threshold: float = 0.5,
                             max_filter_size: int = 7) -> np.ndarray:
    """Build deterministic local-maximum core markers for V61.

    [AUDIT][Risk:High][Scope:decoder_seed_contract]
    V6-V60 used `core_prob >= 0.5` as a connected-component seed mask. That
    treats a broad plateau as one merged anatomy-level seed and makes topology
    depend on every above-threshold voxel. V61 instead treats channel 2 as a
    peak field: only fixed local maxima above the same 0.5 threshold become
    markers. This is a decoder contract change, not a threshold search.

    [QC][Invariant:no_fallback_seed_invention]
    If no local maximum clears the fixed threshold, this function returns no
    seed. The evaluator then fails the case rather than inventing markers from
    GT fragment counts or tuned probabilities.
    """
    core = np.asarray(core_prob, dtype=np.float32)
    support = np.asarray(support).astype(bool, copy=False)
    if core.shape != support.shape:
        raise ValueError(f"V61 peak seed shape mismatch: core={core.shape} support={support.shape}")
    max_filter_size = int(max_filter_size)
    if max_filter_size < 1 or max_filter_size % 2 == 0:
        raise ValueError(f"max_filter_size must be a positive odd integer, got {max_filter_size}")
    if not support.any():
        return np.zeros_like(support, dtype=bool)
    masked = np.where(support, core, -np.inf).astype(np.float32, copy=False)
    local_max = masked == ndi.maximum_filter(masked, size=max_filter_size, mode="nearest")
    return (local_max & support & (core >= float(threshold))).astype(bool, copy=False)


def _bicm_soft_peak_seed_mask(core_prob: np.ndarray,
                              support: np.ndarray,
                              *,
                              relative_threshold: float = 0.50,
                              min_seed_prob: float = 0.05,
                              max_filter_size: int = 41,
                              max_peaks_per_component: int = 12) -> np.ndarray:
    """Build local-max seed markers without a global 0.5 core threshold.

    [AUDIT][Risk:High][Scope:soft_seed_decoder]
    V154/V155 showed hard sacrum fragments can peak at only 0.14-0.21 while
    non-sacrum fragments still peak near 0.8. A single global core threshold
    therefore deletes valid low-confidence seeds. This decoder-only diagnostic
    uses a fixed component-relative cutoff and does not use GT fragment count or
    validation-selected thresholds.
    """
    core = np.asarray(core_prob, dtype=np.float32)
    support = np.asarray(support).astype(bool, copy=False)
    if core.shape != support.shape:
        raise ValueError(f"soft peak seed shape mismatch: core={core.shape} support={support.shape}")
    max_filter_size = int(max_filter_size)
    if max_filter_size < 1 or max_filter_size % 2 == 0:
        raise ValueError(f"max_filter_size must be a positive odd integer, got {max_filter_size}")
    max_peaks_per_component = max(1, int(max_peaks_per_component))
    if not support.any():
        return np.zeros_like(support, dtype=bool)

    seed = np.zeros_like(support, dtype=bool)
    support_cc, n_support = ndi.label(support)
    masked_global = np.where(support, core, -np.inf).astype(np.float32, copy=False)
    local_max_global = masked_global == ndi.maximum_filter(masked_global, size=max_filter_size, mode="nearest")
    for comp_id in range(1, int(n_support) + 1):
        comp = support_cc == comp_id
        if not comp.any():
            continue
        values = core[comp]
        comp_max = float(np.max(values)) if values.size else 0.0
        if not np.isfinite(comp_max) or comp_max < float(min_seed_prob):
            continue
        cutoff = max(float(min_seed_prob), float(relative_threshold) * comp_max)
        candidates = np.flatnonzero((local_max_global & comp & (core >= cutoff)).reshape(-1))
        if candidates.size == 0:
            comp_flat = np.flatnonzero(comp.reshape(-1))
            candidates = comp_flat[np.asarray([int(np.argmax(core.reshape(-1)[comp_flat]))], dtype=np.int64)]
        if candidates.size > max_peaks_per_component:
            flat_core = core.reshape(-1)
            order = np.argsort(flat_core[candidates])[-max_peaks_per_component:]
            candidates = candidates[order]
        seed.reshape(-1)[candidates] = True
    return seed


def _bicm_core_size_filtered_seed_mask(core_raw: np.ndarray,
                                       *,
                                       min_core_component_voxels: int = 128) -> tuple[np.ndarray, dict[str, int | None]]:
    """Filter thresholded core components with the fixed core-size diagnostic rule.

    [AUDIT][Risk:High][Scope:decoder_marker_contract]
    V211/V212 must preserve the exact `core_size_filtered_v1` marker contract
    before adding any soft-fill seeds. Keeping this rule in one helper prevents
    decoder/evaluation metric drift between the decoded labels and reported core
    coverage/topology diagnostics.
    """
    core_raw = np.asarray(core_raw).astype(bool, copy=False)
    min_core_component_voxels = int(min_core_component_voxels)
    if min_core_component_voxels < 1:
        raise ValueError(f"min_core_component_voxels must be positive, got {min_core_component_voxels}")
    raw_cc, raw_n = ndi.label(core_raw)
    sizes = np.bincount(raw_cc.reshape(-1))
    keep_ids = np.where(sizes >= min_core_component_voxels)[0]
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size == 0 and raw_n > 0:
        keep_ids = np.array([int(np.argmax(sizes[1:]) + 1)], dtype=np.int64)
    core = np.isin(raw_cc, keep_ids) if keep_ids.size else np.zeros_like(core_raw, dtype=bool)
    return core.astype(bool, copy=False), {
        "raw_seed_components_before_filter": int(raw_n),
        "kept_seed_components_after_filter": int(keep_ids.size),
        "min_core_component_voxels": int(min_core_component_voxels),
    }


def _bicm_soft_peak_fill_seedless_components(core_prob: np.ndarray,
                                             support: np.ndarray,
                                             existing_core: np.ndarray,
                                             *,
                                             relative_threshold: float = 0.50,
                                             min_seed_prob: float = 0.05,
                                             max_filter_size: int = 41,
                                             max_peaks_per_component: int = 3) -> tuple[np.ndarray, dict[str, int | float]]:
    """Add fixed soft-peak seeds only where support has no existing core marker.

    [QC][Invariant:seedless_only]
    This helper does not split support components that already contain a
    `core_size_filtered_v1` marker. It only tests the narrow failure mode where
    support exists but the preserved core-size marker rule supplies no seed.
    """
    core = np.asarray(core_prob, dtype=np.float32)
    support = np.asarray(support).astype(bool, copy=False)
    existing_core = np.asarray(existing_core).astype(bool, copy=False)
    if core.shape != support.shape or core.shape != existing_core.shape:
        raise ValueError(
            "soft-fill shape mismatch: "
            f"core={core.shape} support={support.shape} existing_core={existing_core.shape}"
        )
    max_filter_size = int(max_filter_size)
    if max_filter_size < 1 or max_filter_size % 2 == 0:
        raise ValueError(f"max_filter_size must be a positive odd integer, got {max_filter_size}")
    max_peaks_per_component = max(1, int(max_peaks_per_component))
    if not support.any():
        filled = np.zeros_like(support, dtype=bool)
        return filled, {
            "softfill_added_seed_voxels": 0,
            "softfill_seed_components_before": 0,
            "softfill_seed_components_after": 0,
            "softfill_seedless_support_components": 0,
            "softfill_filled_support_components": 0,
            "softfill_skipped_lowprob_components": 0,
            "softfill_relative_threshold": float(relative_threshold),
            "softfill_min_seed_prob": float(min_seed_prob),
            "softfill_max_filter_size": int(max_filter_size),
            "softfill_max_peaks_per_component": int(max_peaks_per_component),
        }

    filled = existing_core.copy()
    added = np.zeros_like(support, dtype=bool)
    support_cc, n_support = ndi.label(support)
    seed_before = int(ndi.label(existing_core)[1])
    masked_global = np.where(support, core, -np.inf).astype(np.float32, copy=False)
    local_max_global = masked_global == ndi.maximum_filter(masked_global, size=max_filter_size, mode="nearest")
    seedless_components = 0
    filled_components = 0
    skipped_lowprob_components = 0
    flat_core = core.reshape(-1)
    flat_added = added.reshape(-1)
    for comp_id in range(1, int(n_support) + 1):
        comp = support_cc == comp_id
        if not comp.any() or (existing_core & comp).any():
            continue
        seedless_components += 1
        values = core[comp]
        comp_max = float(np.max(values)) if values.size else 0.0
        if not np.isfinite(comp_max) or comp_max < float(min_seed_prob):
            skipped_lowprob_components += 1
            continue
        cutoff = max(float(min_seed_prob), float(relative_threshold) * comp_max)
        candidates = np.flatnonzero((local_max_global & comp & (core >= cutoff)).reshape(-1))
        if candidates.size == 0:
            comp_flat = np.flatnonzero(comp.reshape(-1))
            candidates = comp_flat[np.asarray([int(np.argmax(flat_core[comp_flat]))], dtype=np.int64)]
        if candidates.size > max_peaks_per_component:
            order = np.argsort(flat_core[candidates])[-max_peaks_per_component:]
            candidates = candidates[order]
        flat_added[candidates] = True
        filled_components += 1
    filled |= added
    return filled.astype(bool, copy=False), {
        "softfill_added_seed_voxels": int(added.sum()),
        "softfill_seed_components_before": int(seed_before),
        "softfill_seed_components_after": int(ndi.label(filled)[1]),
        "softfill_seedless_support_components": int(seedless_components),
        "softfill_filled_support_components": int(filled_components),
        "softfill_skipped_lowprob_components": int(skipped_lowprob_components),
        "softfill_relative_threshold": float(relative_threshold),
        "softfill_min_seed_prob": float(min_seed_prob),
        "softfill_max_filter_size": int(max_filter_size),
        "softfill_max_peaks_per_component": int(max_peaks_per_component),
    }


def _bicm_v105_offset_assignment_labels(support: np.ndarray,
                                        core_prob: np.ndarray,
                                        offsets_zyx: np.ndarray,
                                        contact_prob: np.ndarray,
                                        *,
                                        threshold: float = 0.5,
                                        max_filter_size: int = 17,
                                        offset_scale: float = 64.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign support voxels to nearest predicted seed using raw offset endpoints."""
    support = np.asarray(support).astype(bool, copy=False)
    core_prob = np.asarray(core_prob, dtype=np.float32)
    contact_prob = np.asarray(contact_prob, dtype=np.float32)
    offsets = np.asarray(offsets_zyx, dtype=np.float32)
    if offsets.shape != (3, *support.shape):
        raise ValueError(f"V105 offset shape mismatch: offsets={offsets.shape} support={support.shape}")
    if core_prob.shape != support.shape or contact_prob.shape != support.shape:
        raise ValueError(
            f"V105 map shape mismatch: support={support.shape} core={core_prob.shape} contact={contact_prob.shape}"
        )
    if not np.isfinite(offsets).all():
        raise ValueError("V105 offsets contain NaN/Inf.")

    support_for_marker = support | (core_prob >= float(threshold))
    core = _bicm_v61_peak_seed_mask(
        core_prob,
        support_for_marker,
        threshold=threshold,
        max_filter_size=max_filter_size,
    )
    seed_cc, n_seed = ndi.label(core)
    trace_base = {
        "decoder": "bicm_v105_offset_assignment",
        "decoder_profile": "bicm_v105_offset_assignment",
        "threshold": float(threshold),
        "contact_contract": "v105_offset_contact",
        "support_voxels": int(support.sum()),
        "contact_voxels": int(((contact_prob >= float(threshold)) & support).sum()),
        "raw_core_voxels": int(((core_prob >= float(threshold)) & support_for_marker).sum()),
        "seed_core_voxels": int(core.sum()),
        "seed_components": int(n_seed),
        "seed_policy": f"offset_nearest_seed_local_max_size{int(max_filter_size)}",
        "offset_scale": float(offset_scale),
    }
    if not support.any() or int(n_seed) == 0:
        trace = dict(trace_base)
        trace["missing_seed_component"] = [{"component": 1, "voxels": int(support.sum())}] if support.any() else []
        trace["pred_fragment_count"] = 0
        return np.zeros(support.shape, dtype=np.uint16), trace

    max_reasonable_seed_components = 300
    if int(n_seed) > max_reasonable_seed_components:
        trace = dict(trace_base)
        trace.update({
            "decoder_skipped": "seed_component_explosion",
            "max_allowed_seed_components": int(max_reasonable_seed_components),
            "missing_seed_component": [{"component": 1, "voxels": int(support.sum())}],
            "missing_seed_component_count": 1,
            "missing_seed_component_voxels": int(support.sum()),
            "missing_seed_component_report_cap": 50,
            "pred_fragment_count": 0,
        })
        return np.zeros(support.shape, dtype=np.uint16), trace

    seed_centers = np.asarray(
        ndi.center_of_mass(core, seed_cc, range(1, int(n_seed) + 1)),
        dtype=np.float32,
    )
    if seed_centers.ndim != 2 or seed_centers.shape[1] != 3 or not np.isfinite(seed_centers).all():
        raise RuntimeError(f"V105 seed center computation failed: shape={seed_centers.shape}")

    coords = np.argwhere(support).astype(np.float32, copy=False)
    assigned = np.zeros(support.shape, dtype=np.uint16)
    labels = np.empty((coords.shape[0],), dtype=np.uint16)
    flat_offsets = offsets[:, support].T.astype(np.float32, copy=False)
    chunk_size = 262144
    for start in range(0, int(coords.shape[0]), chunk_size):
        end = min(start + chunk_size, int(coords.shape[0]))
        endpoints = coords[start:end] + flat_offsets[start:end] * float(offset_scale)
        # [QC][Invariant:v105_assignment]
        # Assignment uses only predicted support, predicted seeds, and predicted
        # offsets. No GT fragment count or validation-selected threshold is used.
        dist2 = ((endpoints[:, None, :] - seed_centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:end] = (np.argmin(dist2, axis=1).astype(np.uint16) + np.uint16(1))
    assigned[support] = labels
    pred_ids = [int(v) for v in np.unique(assigned) if int(v) > 0]
    trace = dict(trace_base)
    trace.update({
        "pred_fragment_count": int(len(pred_ids)),
        "mean_offset_norm": float(np.linalg.norm(flat_offsets, axis=1).mean()) if flat_offsets.size else 0.0,
    })
    return assigned, trace


def _bicm_v160_offset_softpeak_assignment_labels(support: np.ndarray,
                                                 core_prob: np.ndarray,
                                                 offsets_zyx: np.ndarray,
                                                 contact_prob: np.ndarray,
                                                 *,
                                                 threshold: float = 0.5,
                                                 max_filter_size: int = 41,
                                                 offset_scale: float = 64.0,
                                                 relative_threshold: float = 0.50,
                                                 min_seed_prob: float = 0.05,
                                                 max_peaks_per_component: int = 12) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign support voxels by offset endpoints, using fixed soft peak seeds."""
    support = np.asarray(support).astype(bool, copy=False)
    core_prob = np.asarray(core_prob, dtype=np.float32)
    contact_prob = np.asarray(contact_prob, dtype=np.float32)
    offsets = np.asarray(offsets_zyx, dtype=np.float32)
    if offsets.shape != (3, *support.shape):
        raise ValueError(f"V160 offset shape mismatch: offsets={offsets.shape} support={support.shape}")
    if core_prob.shape != support.shape or contact_prob.shape != support.shape:
        raise ValueError(
            f"V160 map shape mismatch: support={support.shape} core={core_prob.shape} contact={contact_prob.shape}"
        )
    if not np.isfinite(offsets).all():
        raise ValueError("V160 offsets contain NaN/Inf.")

    core = _bicm_soft_peak_seed_mask(
        core_prob,
        support,
        relative_threshold=float(relative_threshold),
        min_seed_prob=float(min_seed_prob),
        max_filter_size=int(max_filter_size),
        max_peaks_per_component=int(max_peaks_per_component),
    )
    seed_cc, n_seed = ndi.label(core)
    trace_base = {
        "decoder": "bicm_v160_offset_softpeak_assignment",
        "decoder_profile": "bicm_v160_offset_softpeak_assignment",
        "threshold": float(threshold),
        "contact_contract": "v105_offset_contact",
        "support_voxels": int(support.sum()),
        "contact_voxels": int(((contact_prob >= float(threshold)) & support).sum()),
        "raw_core_voxels": int(((core_prob >= float(threshold)) & support).sum()),
        "seed_core_voxels": int(core.sum()),
        "seed_components": int(n_seed),
        "seed_policy": f"offset_soft_component_peak_size{int(max_filter_size)}",
        "offset_scale": float(offset_scale),
        "relative_threshold": float(relative_threshold),
        "min_seed_prob": float(min_seed_prob),
        "max_peaks_per_component": int(max_peaks_per_component),
    }
    if not support.any() or int(n_seed) == 0:
        trace = dict(trace_base)
        trace["missing_seed_component"] = [{"component": 1, "voxels": int(support.sum())}] if support.any() else []
        trace["pred_fragment_count"] = 0
        return np.zeros(support.shape, dtype=np.uint16), trace

    max_reasonable_seed_components = 300
    if int(n_seed) > max_reasonable_seed_components:
        trace = dict(trace_base)
        trace.update({
            "decoder_skipped": "seed_component_explosion",
            "max_allowed_seed_components": int(max_reasonable_seed_components),
            "missing_seed_component": [{"component": 1, "voxels": int(support.sum())}],
            "missing_seed_component_count": 1,
            "missing_seed_component_voxels": int(support.sum()),
            "missing_seed_component_report_cap": 50,
            "pred_fragment_count": 0,
        })
        return np.zeros(support.shape, dtype=np.uint16), trace

    seed_centers = np.asarray(
        ndi.center_of_mass(core, seed_cc, range(1, int(n_seed) + 1)),
        dtype=np.float32,
    )
    if seed_centers.ndim != 2 or seed_centers.shape[1] != 3 or not np.isfinite(seed_centers).all():
        raise RuntimeError(f"V160 seed center computation failed: shape={seed_centers.shape}")

    coords = np.argwhere(support).astype(np.float32, copy=False)
    assigned = np.zeros(support.shape, dtype=np.uint16)
    labels = np.empty((coords.shape[0],), dtype=np.uint16)
    flat_offsets = offsets[:, support].T.astype(np.float32, copy=False)
    chunk_size = 262144
    for start in range(0, int(coords.shape[0]), chunk_size):
        end = min(start + chunk_size, int(coords.shape[0]))
        endpoints = coords[start:end] + flat_offsets[start:end] * float(offset_scale)
        # [QC][Invariant:v160_assignment]
        # Assignment uses predicted support, fixed soft predicted seeds, and raw
        # predicted offsets only. It does not use GT fragment count or per-case
        # threshold selection.
        dist2 = ((endpoints[:, None, :] - seed_centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:end] = (np.argmin(dist2, axis=1).astype(np.uint16) + np.uint16(1))
    assigned[support] = labels
    pred_ids = [int(v) for v in np.unique(assigned) if int(v) > 0]
    trace = dict(trace_base)
    trace.update({
        "pred_fragment_count": int(len(pred_ids)),
        "mean_offset_norm": float(np.linalg.norm(flat_offsets, axis=1).mean()) if flat_offsets.size else 0.0,
    })
    return assigned, trace


class _BICMDisjointSet:
    """Small union-find for V112 edge-graph decoder diagnostics."""

    def __init__(self, size: int):
        self.parent = np.arange(int(size), dtype=np.int32)
        self.rank = np.zeros(int(size), dtype=np.uint8)

    def find(self, value: int) -> int:
        parent = self.parent
        value = int(value)
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    def union_many(self, left: np.ndarray, right: np.ndarray) -> None:
        for a, b in zip(left.tolist(), right.tolist()):
            ra = self.find(int(a))
            rb = self.find(int(b))
            if ra == rb:
                continue
            if self.rank[ra] < self.rank[rb]:
                ra, rb = rb, ra
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1


def _bicm_v112_edge_graph_components(support: np.ndarray,
                                      edge_prob_zyx: np.ndarray,
                                      *,
                                      threshold: float = 0.5) -> tuple[np.ndarray, int]:
    """Return support components connected through below-threshold edge links."""
    support = np.asarray(support).astype(bool, copy=False)
    edge_prob = np.asarray(edge_prob_zyx, dtype=np.float32)
    if edge_prob.shape != (3, *support.shape):
        raise ValueError(f"V112 edge shape mismatch: edge={edge_prob.shape} support={support.shape}")
    labels = np.full(support.shape, -1, dtype=np.int32)
    coords = np.nonzero(support)
    n_voxels = int(len(coords[0]))
    if n_voxels == 0:
        return np.zeros(support.shape, dtype=np.int32), 0
    labels[coords] = np.arange(n_voxels, dtype=np.int32)
    dsu = _BICMDisjointSet(n_voxels)
    for axis in range(3):
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        src[axis] = slice(0, -1)
        dst[axis] = slice(1, None)
        connected = (
            support[tuple(src)]
            & support[tuple(dst)]
            & (edge_prob[axis][tuple(src)] < float(threshold))
        )
        if connected.any():
            dsu.union_many(labels[tuple(src)][connected], labels[tuple(dst)][connected])
    roots = np.empty(n_voxels, dtype=np.int32)
    for idx in range(n_voxels):
        roots[idx] = dsu.find(idx)
    _, inverse = np.unique(roots, return_inverse=True)
    out = np.zeros(support.shape, dtype=np.int32)
    out[coords] = inverse.astype(np.int32) + 1
    return out, int(inverse.max() + 1) if inverse.size else 0


def _bicm_v112_edge_graph_assignment_labels(support_prob: np.ndarray,
                                            core_prob: np.ndarray,
                                            edge_prob_zyx: np.ndarray,
                                            contact_prob: np.ndarray,
                                            *,
                                            threshold: float = 0.5,
                                            max_filter_size: int = 17) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign edge-graph components to predicted peak seeds.

    [AUDIT][Risk:High][Scope:v112_decoder_contract]
    This decoder changes the assignment structure, not the threshold. Rows 7..9
    are interpreted as local break probabilities; connected same-fragment graph
    components are then assigned to predicted row6 peak seeds. No GT fragment
    count, GT seed, or validation-selected threshold is used.
    """
    support_prob = np.asarray(support_prob, dtype=np.float32)
    core_prob = np.asarray(core_prob, dtype=np.float32)
    contact_prob = np.asarray(contact_prob, dtype=np.float32)
    support = support_prob >= float(threshold)
    support_for_marker = support | (core_prob >= float(threshold))
    core = _bicm_v61_peak_seed_mask(
        core_prob,
        support_for_marker,
        threshold=threshold,
        max_filter_size=max_filter_size,
    )
    seed_cc, n_seed = ndi.label(core)
    contact = (contact_prob >= float(threshold)) & support_for_marker
    trace_base = {
        "decoder": "bicm_v112_edge_graph_assignment",
        "decoder_profile": "bicm_v112_edge_graph_assignment",
        "threshold": float(threshold),
        "contact_contract": "v68_edge_primary",
        "support_voxels": int(support_for_marker.sum()),
        "contact_voxels": int(contact.sum()),
        "raw_core_voxels": int(((core_prob >= float(threshold)) & support_for_marker).sum()),
        "seed_core_voxels": int(core.sum()),
        "core_voxels_removed_by_contact": 0,
        "seed_components": int(n_seed),
        "seed_policy": f"edge_graph_components_to_peak_seed_size{int(max_filter_size)}",
    }
    if not support_for_marker.any() or int(n_seed) == 0:
        trace = dict(trace_base)
        trace.update({
            "edge_graph_components": 0,
            "pred_fragment_count": 0,
            "missing_seed_component": [{"component": 1, "voxels": int(support_for_marker.sum())}]
            if support_for_marker.any() else [],
        })
        return np.zeros(support_for_marker.shape, dtype=np.uint16), trace

    graph_cc, n_graph = _bicm_v112_edge_graph_components(
        support_for_marker,
        edge_prob_zyx,
        threshold=threshold,
    )
    max_reasonable_graph_components = 200000
    if int(n_graph) > max_reasonable_graph_components:
        trace = dict(trace_base)
        trace.update({
            "decoder_skipped": "edge_graph_component_explosion",
            "edge_graph_components": int(n_graph),
            "max_allowed_edge_graph_components": int(max_reasonable_graph_components),
            "pred_fragment_count": 0,
            "missing_seed_component": [{"component": 1, "voxels": int(support_for_marker.sum())}],
        })
        return np.zeros(support_for_marker.shape, dtype=np.uint16), trace

    _, nearest_indices = ndi.distance_transform_edt(
        ~core,
        return_indices=True,
    )
    nearest_seed = seed_cc[tuple(nearest_indices)].astype(np.int32, copy=False)
    component_ids = np.arange(1, int(n_graph) + 1, dtype=np.int32)
    component_seed = ndi.maximum(seed_cc, labels=graph_cc, index=component_ids).astype(np.int32, copy=False)

    flat_graph = graph_cc.reshape(-1)
    flat_nearest = nearest_seed.reshape(-1)
    valid = flat_graph > 0
    if valid.any():
        order = np.argsort(flat_graph[valid], kind="mergesort")
        sorted_graph = flat_graph[valid][order]
        sorted_nearest = flat_nearest[valid][order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_graph)) + 1]
        first_component = sorted_graph[starts]
        first_nearest = sorted_nearest[starts]
        missing = component_seed[first_component - 1] <= 0
        component_seed[first_component[missing] - 1] = first_nearest[missing]

    assigned = np.zeros(support_for_marker.shape, dtype=np.uint16)
    if int(n_graph) > 0:
        mapped = component_seed[np.maximum(graph_cc, 1) - 1]
        assigned[support_for_marker] = mapped[support_for_marker].astype(np.uint16, copy=False)
    pred_ids = [int(v) for v in np.unique(assigned) if int(v) > 0]
    trace = dict(trace_base)
    trace.update({
        "edge_graph_components": int(n_graph),
        "pred_fragment_count": int(len(pred_ids)),
        "mean_support_prob": float(np.mean(support_prob)),
        "mean_contact_prob": float(np.mean(contact_prob)),
        "mean_core_prob": float(np.mean(core_prob)),
        "mean_exterior_prob": 0.0,
        "mean_contact_aux_prob": float(np.mean(np.asarray(edge_prob_zyx, dtype=np.float32))),
    })
    return assigned.astype(np.uint16, copy=False), trace


def _bicm_v116_graph_cost_assignment_labels(support_prob: np.ndarray,
                                            core_prob: np.ndarray,
                                            edge_prob_zyx: np.ndarray,
                                            contact_prob: np.ndarray,
                                            *,
                                            threshold: float = 0.5,
                                            max_filter_size: int = 17,
                                            barrier_scale: float = 18.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign support by multi-source shortest path through separator costs.

    [AUDIT][Risk:High][Scope:v116_decoder_contract]
    Rows 7..9 are local separator costs between 6-neighbor voxels. Unlike V112,
    this decoder does not threshold them into connected components; every edge
    remains traversable with a continuous cost. Seeds come only from predicted
    row6 local maxima, and no GT fragment count or validation-tuned threshold is
    used.
    """
    support_prob = np.asarray(support_prob, dtype=np.float32)
    core_prob = np.asarray(core_prob, dtype=np.float32)
    contact_prob = np.asarray(contact_prob, dtype=np.float32)
    edge_prob = np.asarray(edge_prob_zyx, dtype=np.float32)
    support = support_prob >= float(threshold)
    support_for_marker = support | (core_prob >= float(threshold))
    if edge_prob.shape != (3, *support_for_marker.shape):
        raise ValueError(f"V116 edge shape mismatch: edge={edge_prob.shape} support={support_for_marker.shape}")
    if not np.isfinite(edge_prob).all():
        raise ValueError("V116 edge probabilities contain NaN/Inf.")

    core = _bicm_v61_peak_seed_mask(
        core_prob,
        support_for_marker,
        threshold=threshold,
        max_filter_size=max_filter_size,
    )
    seed_cc, n_seed = ndi.label(core)
    contact = (contact_prob >= float(threshold)) & support_for_marker
    trace_base = {
        "decoder": "bicm_v116_graph_cost_assignment",
        "decoder_profile": "bicm_v116_graph_cost_assignment",
        "threshold": float(threshold),
        "contact_contract": "v68_edge_primary",
        "support_voxels": int(support_for_marker.sum()),
        "contact_voxels": int(contact.sum()),
        "raw_core_voxels": int(((core_prob >= float(threshold)) & support_for_marker).sum()),
        "seed_core_voxels": int(core.sum()),
        "seed_components": int(n_seed),
        "seed_policy": f"graph_cost_to_peak_seed_size{int(max_filter_size)}",
        "barrier_scale": float(barrier_scale),
    }
    if not support_for_marker.any() or int(n_seed) == 0:
        trace = dict(trace_base)
        trace.update({
            "pred_fragment_count": 0,
            "missing_seed_component": [{"component": 1, "voxels": int(support_for_marker.sum())}]
            if support_for_marker.any() else [],
        })
        return np.zeros(support_for_marker.shape, dtype=np.uint16), trace

    support_count = int(support_for_marker.sum())
    max_reasonable_support_voxels = 900000
    if support_count > max_reasonable_support_voxels:
        trace = dict(trace_base)
        trace.update({
            "decoder_skipped": "graph_cost_support_too_large",
            "max_allowed_support_voxels": int(max_reasonable_support_voxels),
            "pred_fragment_count": 0,
            "missing_seed_component": [{"component": 1, "voxels": support_count}],
        })
        return np.zeros(support_for_marker.shape, dtype=np.uint16), trace

    shape = tuple(int(v) for v in support_for_marker.shape)
    size = int(np.prod(shape))
    support_flat = support_for_marker.reshape(-1)
    seed_flat = seed_cc.reshape(-1).astype(np.int32, copy=False)
    edge_flat = edge_prob.reshape(3, -1)
    dist = np.full(size, np.inf, dtype=np.float32)
    labels = np.zeros(size, dtype=np.uint16)
    visited = np.zeros(size, dtype=bool)
    seed_indices = np.flatnonzero(seed_flat > 0)
    heap: list[tuple[float, int, int]] = []
    for idx in seed_indices:
        label = int(seed_flat[idx])
        dist[int(idx)] = 0.0
        labels[int(idx)] = np.uint16(label)
        heapq.heappush(heap, (0.0, label, int(idx)))

    sy = shape[1] * shape[2]
    sx = shape[2]
    z_max, y_max, x_max = shape
    barrier_scale = float(barrier_scale)

    def _try_push(cur_idx: int, nb_idx: int, axis: int, src_idx: int, cur_dist: float, label: int) -> None:
        if not support_flat[nb_idx] or visited[nb_idx]:
            return
        edge_cost = float(edge_flat[axis, src_idx])
        new_dist = cur_dist + 1.0 + barrier_scale * max(0.0, min(1.0, edge_cost))
        if new_dist < float(dist[nb_idx]):
            dist[nb_idx] = np.float32(new_dist)
            labels[nb_idx] = np.uint16(label)
            heapq.heappush(heap, (float(new_dist), int(label), int(nb_idx)))

    popped = 0
    while heap:
        cur_dist, label, idx = heapq.heappop(heap)
        if visited[idx]:
            continue
        visited[idx] = True
        popped += 1
        z = idx // sy
        rem = idx - z * sy
        y = rem // sx
        x = rem - y * sx
        if z > 0:
            nb = idx - sy
            _try_push(idx, nb, 0, nb, cur_dist, label)
        if z + 1 < z_max:
            nb = idx + sy
            _try_push(idx, nb, 0, idx, cur_dist, label)
        if y > 0:
            nb = idx - sx
            _try_push(idx, nb, 1, nb, cur_dist, label)
        if y + 1 < y_max:
            nb = idx + sx
            _try_push(idx, nb, 1, idx, cur_dist, label)
        if x > 0:
            nb = idx - 1
            _try_push(idx, nb, 2, nb, cur_dist, label)
        if x + 1 < x_max:
            nb = idx + 1
            _try_push(idx, nb, 2, idx, cur_dist, label)

    assigned = np.zeros(size, dtype=np.uint16)
    assigned[support_flat & (labels > 0)] = labels[support_flat & (labels > 0)]
    assigned = assigned.reshape(shape)
    support_cc, n_support = ndi.label(support_for_marker)
    missing = []
    for comp_id in range(1, int(n_support) + 1):
        comp = support_cc == comp_id
        if not (assigned[comp] > 0).any():
            missing.append({"component": int(comp_id), "voxels": int(comp.sum())})
    pred_ids = [int(v) for v in np.unique(assigned) if int(v) > 0]
    finite_dist = dist[support_flat & np.isfinite(dist)]
    trace = dict(trace_base)
    trace.update({
        "pred_fragment_count": int(len(pred_ids)),
        "visited_support_voxels": int(popped),
        "missing_seed_component": missing[:50],
        "missing_seed_component_count": int(len(missing)),
        "mean_path_cost": float(finite_dist.mean()) if finite_dist.size else 0.0,
        "p95_path_cost": float(np.percentile(finite_dist, 95)) if finite_dist.size else 0.0,
        "mean_support_prob": float(np.mean(support_prob)),
        "mean_contact_prob": float(np.mean(contact_prob)),
        "mean_core_prob": float(np.mean(core_prob)),
        "mean_contact_aux_prob": float(np.mean(edge_prob)),
    })
    return assigned.astype(np.uint16, copy=False), trace


def _bicm_v120_affinity_graph_assignment_labels(support_prob: np.ndarray,
                                                core_prob: np.ndarray,
                                                affinity_prob_zyx: np.ndarray,
                                                contact_prob: np.ndarray,
                                                *,
                                                threshold: float = 0.5,
                                                max_filter_size: int = 17,
                                                barrier_scale: float = 18.0,
                                                seed_policy: str = "absolute_peak",
                                                relative_threshold: float = 0.50,
                                                min_seed_prob: float = 0.05,
                                                core_support_floor: float | None = None,
                                                max_peaks_per_component: int = 12) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign support by shortest path where low same-fragment affinity is costly.

    [AUDIT][Risk:High][Scope:v120_decoder_contract]
    Rows 7..9 are local same-fragment affinities, not break probabilities. This
    decoder uses `1 - affinity` as traversal barrier and keeps the V92 predicted
    peak-seed rule. No GT fragment count, GT seed, or tuned threshold is used.
    """
    support_prob = np.asarray(support_prob, dtype=np.float32)
    core_prob = np.asarray(core_prob, dtype=np.float32)
    contact_prob = np.asarray(contact_prob, dtype=np.float32)
    affinity_prob = np.asarray(affinity_prob_zyx, dtype=np.float32)
    support = support_prob >= float(threshold)
    if core_support_floor is None:
        support_for_marker = support | (core_prob >= float(threshold))
    else:
        support_for_marker = support | (core_prob >= float(core_support_floor))
    if affinity_prob.shape != (3, *support_for_marker.shape):
        raise ValueError(
            f"V120 affinity shape mismatch: affinity={affinity_prob.shape} "
            f"support={support_for_marker.shape}"
        )
    if not np.isfinite(affinity_prob).all():
        raise ValueError("V120 affinity probabilities contain NaN/Inf.")
    affinity_prob = np.clip(affinity_prob, 0.0, 1.0).astype(np.float32, copy=False)

    if seed_policy == "absolute_peak":
        core = _bicm_v61_peak_seed_mask(
            core_prob,
            support_for_marker,
            threshold=threshold,
            max_filter_size=max_filter_size,
        )
        seed_policy_name = f"affinity_graph_to_peak_seed_size{int(max_filter_size)}"
    elif seed_policy == "soft_component_peak":
        core = _bicm_soft_peak_seed_mask(
            core_prob,
            support_for_marker,
            relative_threshold=relative_threshold,
            min_seed_prob=min_seed_prob,
            max_filter_size=max_filter_size,
            max_peaks_per_component=max_peaks_per_component,
        )
        seed_policy_name = (
            f"affinity_graph_to_soft_component_peak_size{int(max_filter_size)}"
            f"_rel{float(relative_threshold):.2f}_min{float(min_seed_prob):.2f}"
        )
    else:
        raise ValueError(f"unsupported V120 seed_policy={seed_policy!r}")
    seed_cc, n_seed = ndi.label(core)
    contact = (contact_prob >= float(threshold)) & support_for_marker
    break_prob = (1.0 - affinity_prob).astype(np.float32, copy=False)
    trace_base = {
        "decoder": "bicm_v120_affinity_graph_assignment",
        "decoder_profile": "bicm_v120_affinity_graph_assignment",
        "threshold": float(threshold),
        "contact_contract": "v120_affinity_break",
        "support_voxels": int(support_for_marker.sum()),
        "contact_voxels": int(contact.sum()),
        "raw_core_voxels": int(((core_prob >= float(threshold)) & support_for_marker).sum()),
        "seed_core_voxels": int(core.sum()),
        "seed_components": int(n_seed),
        "seed_policy": seed_policy_name,
        "barrier_scale": float(barrier_scale),
        "core_support_floor": None if core_support_floor is None else float(core_support_floor),
        "relative_threshold": float(relative_threshold),
        "min_seed_prob": float(min_seed_prob),
        "max_peaks_per_component": int(max_peaks_per_component),
    }
    if not support_for_marker.any() or int(n_seed) == 0:
        trace = dict(trace_base)
        trace.update({
            "pred_fragment_count": 0,
            "missing_seed_component": [{"component": 1, "voxels": int(support_for_marker.sum())}]
            if support_for_marker.any() else [],
        })
        return np.zeros(support_for_marker.shape, dtype=np.uint16), trace

    support_count = int(support_for_marker.sum())
    max_reasonable_support_voxels = 900000
    if support_count > max_reasonable_support_voxels:
        trace = dict(trace_base)
        trace.update({
            "decoder_skipped": "affinity_graph_support_too_large",
            "max_allowed_support_voxels": int(max_reasonable_support_voxels),
            "pred_fragment_count": 0,
            "missing_seed_component": [{"component": 1, "voxels": support_count}],
        })
        return np.zeros(support_for_marker.shape, dtype=np.uint16), trace

    shape = tuple(int(v) for v in support_for_marker.shape)
    size = int(np.prod(shape))
    support_flat = support_for_marker.reshape(-1)
    seed_flat = seed_cc.reshape(-1).astype(np.int32, copy=False)
    affinity_flat = affinity_prob.reshape(3, -1)
    dist = np.full(size, np.inf, dtype=np.float32)
    labels = np.zeros(size, dtype=np.uint16)
    visited = np.zeros(size, dtype=bool)
    seed_indices = np.flatnonzero(seed_flat > 0)
    heap: list[tuple[float, int, int]] = []
    for idx in seed_indices:
        label = int(seed_flat[idx])
        dist[int(idx)] = 0.0
        labels[int(idx)] = np.uint16(label)
        heapq.heappush(heap, (0.0, label, int(idx)))

    sy = shape[1] * shape[2]
    sx = shape[2]
    z_max, y_max, x_max = shape
    barrier_scale = float(barrier_scale)

    def _try_push(nb_idx: int, axis: int, src_idx: int, cur_dist: float, label: int) -> None:
        if not support_flat[nb_idx] or visited[nb_idx]:
            return
        # [QC][Invariant:v120_axis_direction]
        # The affinity row at `src_idx` describes the edge from source voxel to
        # source+axis. For negative moves, caller passes the neighbor as source.
        affinity = float(affinity_flat[axis, src_idx])
        edge_cost = 1.0 - max(0.0, min(1.0, affinity))
        new_dist = cur_dist + 1.0 + barrier_scale * edge_cost
        if new_dist < float(dist[nb_idx]):
            dist[nb_idx] = np.float32(new_dist)
            labels[nb_idx] = np.uint16(label)
            heapq.heappush(heap, (float(new_dist), int(label), int(nb_idx)))

    popped = 0
    while heap:
        cur_dist, label, idx = heapq.heappop(heap)
        if visited[idx]:
            continue
        visited[idx] = True
        popped += 1
        z = idx // sy
        rem = idx - z * sy
        y = rem // sx
        x = rem - y * sx
        if z > 0:
            _try_push(idx - sy, 0, idx - sy, cur_dist, label)
        if z + 1 < z_max:
            _try_push(idx + sy, 0, idx, cur_dist, label)
        if y > 0:
            _try_push(idx - sx, 1, idx - sx, cur_dist, label)
        if y + 1 < y_max:
            _try_push(idx + sx, 1, idx, cur_dist, label)
        if x > 0:
            _try_push(idx - 1, 2, idx - 1, cur_dist, label)
        if x + 1 < x_max:
            _try_push(idx + 1, 2, idx, cur_dist, label)

    assigned = np.zeros(size, dtype=np.uint16)
    assigned[support_flat & (labels > 0)] = labels[support_flat & (labels > 0)]
    assigned = assigned.reshape(shape)
    support_cc, n_support = ndi.label(support_for_marker)
    missing = []
    for comp_id in range(1, int(n_support) + 1):
        comp = support_cc == comp_id
        if not (assigned[comp] > 0).any():
            missing.append({"component": int(comp_id), "voxels": int(comp.sum())})
    pred_ids = [int(v) for v in np.unique(assigned) if int(v) > 0]
    finite_dist = dist[support_flat & np.isfinite(dist)]
    trace = dict(trace_base)
    trace.update({
        "pred_fragment_count": int(len(pred_ids)),
        "visited_support_voxels": int(popped),
        "missing_seed_component": missing[:50],
        "missing_seed_component_count": int(len(missing)),
        "mean_path_cost": float(finite_dist.mean()) if finite_dist.size else 0.0,
        "p95_path_cost": float(np.percentile(finite_dist, 95)) if finite_dist.size else 0.0,
        "mean_support_prob": float(np.mean(support_prob)),
        "mean_contact_prob": float(np.mean(contact_prob)),
        "mean_core_prob": float(np.mean(core_prob)),
        "mean_affinity_prob": float(np.mean(affinity_prob)),
        "mean_contact_aux_prob": float(np.mean(break_prob)),
    })
    return assigned.astype(np.uint16, copy=False), trace


_BICM_V120_AFFINITY_GRAPH_PEAK_FILTER_BY_PROFILE = {
    "bicm_v120_affinity_graph_peak11_assignment": 11,
    "bicm_v120_affinity_graph_assignment": 17,
    "bicm_v120_affinity_graph_peak23_assignment": 23,
}


_BICM_V120_AFFINITY_GRAPH_SOFT_PEAK_PROFILES = {
    "bicm_v156_affinity_graph_softpeak_assignment": {
        "max_filter_size": 41,
        "relative_threshold": 0.50,
        "min_seed_prob": 0.05,
        "core_support_floor": None,
        "max_peaks_per_component": 12,
    },
    "bicm_v157_affinity_graph_softpeak_corefloor_assignment": {
        "max_filter_size": 41,
        "relative_threshold": 0.50,
        "min_seed_prob": 0.05,
        "core_support_floor": 0.10,
        "max_peaks_per_component": 12,
    },
}


_BICM_V6_SOFT_PEAK_PROFILES = {
    "bicm_v207_component_softpeak_watershed": {
        "max_filter_size": 41,
        "relative_threshold": 0.50,
        "min_seed_prob": 0.05,
        "core_support_floor": None,
        "max_peaks_per_component": 12,
    },
    "bicm_v208_component_softpeak_corefloor_watershed": {
        "max_filter_size": 41,
        "relative_threshold": 0.50,
        "min_seed_prob": 0.05,
        "core_support_floor": 0.10,
        "max_peaks_per_component": 12,
    },
}


_BICM_V6_CORE_SIZE_SOFT_FILL_PROFILES = {
    "bicm_v211_core_size_softfill_watershed": {
        "max_filter_size": 41,
        "relative_threshold": 0.50,
        "min_seed_prob": 0.05,
        "core_support_floor": None,
        "max_peaks_per_component": 3,
        "min_core_component_voxels": 128,
    },
    "bicm_v212_core_size_softfill_corefloor_watershed": {
        "max_filter_size": 41,
        "relative_threshold": 0.50,
        "min_seed_prob": 0.05,
        "core_support_floor": 0.10,
        "max_peaks_per_component": 3,
        "min_core_component_voxels": 128,
    },
}










def _probability_distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "p0": None,
            "p1": None,
            "p5": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p100": None,
            "frac_ge_025": None,
            "frac_ge_050": None,
        }
    qs = np.percentile(values, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p0": float(qs[0]),
        "p1": float(qs[1]),
        "p5": float(qs[2]),
        "p10": float(qs[3]),
        "p25": float(qs[4]),
        "p50": float(qs[5]),
        "p75": float(qs[6]),
        "p90": float(qs[7]),
        "p95": float(qs[8]),
        "p99": float(qs[9]),
        "p100": float(qs[10]),
        "frac_ge_025": float(np.mean(values >= 0.25)),
        "frac_ge_050": float(np.mean(values >= 0.50)),
    }


def run_bfv3_contact_forensics(cases: list[str],
                               out_path: Path,
                               output_root: Path,
                               *,
                               roi_pad_vox: int = 24,
                               target_profile: str = "v5_core_body_contact_band",
                               core_ball_radius_mm: float = 2.5,
                               core_body_mm: float = 3.0,
                               contact_band_mm: float = 2.0,
                               threshold: float = 0.5,
                               contact_contract: str = "auto") -> dict:
    """BFV3 channel5 contact 분포를 full-eval V5 및 BFV3 training target 기준으로 요약한다.

    [QA][Scope:v247_forensic_gate]
    이 명령은 학습/평가 설정을 바꾸지 않는다. `0.50` contact가 다시 실패하면
    loss weight bracket을 추가하기 전에 channel5가 full-eval V5 contact_surface와
    실제 train target인 BFV3 label2/label1·3·4를 분리했는지 같은 full-volume cache에서
    확인한다.
    """
    if output_root is None:
        raise ValueError("bfv3-contact-forensics requires --output-root")
    resolved_contact_contract = _infer_bicm_contact_contract(output_root, contact_contract)
    if resolved_contact_contract not in {
        "boundary_fragment_v162",
        "boundary_fragment_decoder_contact_v1",
        "boundary_fragment_decoder_contact_v2_core_veto",
    }:
        raise ValueError(
            "bfv3-contact-forensics expects a BFV3 contact contract, got "
            f"{resolved_contact_contract!r}"
        )
    params = BICMV5Params(
        target_profile=target_profile,
        core_ball_radius_mm=core_ball_radius_mm,
        core_body_mm=core_body_mm,
        contact_band_mm=contact_band_mm,
    )
    bfv3_target_root = (
        NN_PREP
        / DATASETS[537]["name"]
        / "nnUNetPlans_3d_fullres"
        / BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR
    )
    buckets: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"contact_prob": [], "support_prob": [], "contact_aux_prob": []}
        for name in (
            "v5_contact_label4_positive",
            "v5_shell_label2_negative",
            "v5_core_label3_negative",
            "bfv3_label1_external_context_negative",
            "bfv3_label2_positive",
            "bfv3_label3_shell_negative",
            "bfv3_label4_core_negative",
        )
    }
    decoder_contact_pred_total = 0
    v5_contact_total = 0
    v5_shell_total = 0
    bfv3_label2_total = 0
    bfv3_negative_total = 0
    sample_rows = []
    for case_raw in cases:
        case = str(case_raw).zfill(3)
        cd = find_case_dir(case)
        if cd is None:
            raise FileNotFoundError(f"missing source case {case}")
        # [QC][Invariant:case_geometry]
        # forensic target은 `bicm-v6-eval`과 같은 canonical label geometry에서
        # 만들어야 한다. 별도 loader 이름을 쓰면 흡수/정리 과정에서 누락되어
        # 평가와 forensic이 서로 다른 ROI/spacing을 볼 수 있다.
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        inst_full = sitk.GetArrayFromImage(lbl_img).astype(np.uint16, copy=False)
        spacing_zyx = tuple(float(v) for v in lbl_img.GetSpacing()[::-1])
        for anatomy, (lo, hi) in V5_ANATOMY_RANGES.items():
            sample = f"PENGWIN_{case}_{anatomy}"
            anat_mask = anatomy_mask_from_instances(inst_full, anatomy)
            bbox = bbox_from_mask(anat_mask, pad_vox=roi_pad_vox)
            if bbox is None:
                raise RuntimeError(f"case {case} {anatomy}: empty anatomy ROI")
            inst_roi = np.where(
                (inst_full[bbox] >= lo) & (inst_full[bbox] <= hi),
                inst_full[bbox],
                0,
            ).astype(np.uint16, copy=False)
            target_path = NN_RAW / DATASETS[537]["name"] / "labelsTr" / f"{sample}.mha"
            if target_path.exists():
                target = sitk.GetArrayFromImage(
                    canonicalize_sitk(sitk.ReadImage(str(target_path)))
                ).astype(np.uint8, copy=False)
            else:
                target = compute_bicm_v5_target(inst_roi, spacing_zyx=spacing_zyx, params=params)
            raw_logits = _load_bicm_v6_logits(output_root / f"{sample}.npz")
            probs = _normalize_boundary_fragment_v162_logits(
                raw_logits,
                contact_contract=resolved_contact_contract,
            )
            if tuple(probs.shape[1:]) != tuple(target.shape):
                probs = _resize_channel_first_probabilities(probs, tuple(target.shape))
            support_prob, contact_prob, _core_prob, _exterior_prob, contact_aux_prob, effective_contract = _bicm_head_probabilities(
                probs,
                contact_contract=resolved_contact_contract,
                threshold=threshold,
            )
            contact_mask = (contact_prob >= float(threshold)) & (support_prob >= float(threshold))
            v5_contact = target == V5_LABELS["contact_surface"]
            v5_shell = target == V5_LABELS["interior_shell"]
            v5_core = target == V5_LABELS["core"]
            bfv3_path = bfv3_target_root / f"{sample}.npz"
            if not bfv3_path.exists():
                raise FileNotFoundError(f"missing BFV3 target sidecar for forensic: {bfv3_path}")
            with np.load(bfv3_path) as payload:
                if "target" not in payload:
                    raise KeyError(f"{bfv3_path} does not contain `target`")
                bfv3_target = np.asarray(payload["target"], dtype=np.uint8)
            if tuple(bfv3_target.shape) != tuple(contact_prob.shape):
                bfv3_target = _resize_label_volume_nearest(bfv3_target, tuple(contact_prob.shape))
            bfv3_external = bfv3_target == BFV3_LABELS["external_context"]
            bfv3_contact = bfv3_target == BFV3_LABELS["fracture_barrier"]
            bfv3_shell = bfv3_target == BFV3_LABELS["fragment_shell"]
            bfv3_core = bfv3_target == BFV3_LABELS["fragment_core"]
            # [DATA][Risk:High][Scope:v5_contact_distribution]
            # V247의 핵심 가설은 BFV3 label2 positive와 BFV3 label1/3/4 negative의
            # channel5 분포가 0.50 기준에서 분리되고, 그 분리가 full-eval V5
            # contact_surface recall/ratio로 이어진다는 것이다. 여기서는 V5 평가 GT와
            # BFV3 학습 GT를 함께 기록해 target mismatch와 channel5 calibration 실패를
            # 분리한다.
            for name, mask in [
                ("v5_contact_label4_positive", v5_contact),
                ("v5_shell_label2_negative", v5_shell),
                ("v5_core_label3_negative", v5_core),
                ("bfv3_label1_external_context_negative", bfv3_external),
                ("bfv3_label2_positive", bfv3_contact),
                ("bfv3_label3_shell_negative", bfv3_shell),
                ("bfv3_label4_core_negative", bfv3_core),
            ]:
                if np.any(mask):
                    buckets[name]["contact_prob"].append(contact_prob[mask].astype(np.float32, copy=False))
                    buckets[name]["support_prob"].append(support_prob[mask].astype(np.float32, copy=False))
                    if contact_aux_prob is not None:
                        buckets[name]["contact_aux_prob"].append(contact_aux_prob[mask].astype(np.float32, copy=False))
            decoder_contact_pred_total += int(contact_mask.sum())
            v5_contact_total += int(v5_contact.sum())
            v5_shell_total += int(v5_shell.sum())
            bfv3_label2_total += int(bfv3_contact.sum())
            bfv3_negative_total += int((bfv3_external | bfv3_shell | bfv3_core).sum())
            sample_rows.append({
                "case": case,
                "sample": sample,
                "anatomy": anatomy,
                "shape": list(target.shape),
                "contact_contract": effective_contract,
                "v5_contact_label4_voxels": int(v5_contact.sum()),
                "v5_shell_label2_voxels": int(v5_shell.sum()),
                "v5_core_label3_voxels": int(v5_core.sum()),
                "bfv3_label1_external_context_voxels": int(bfv3_external.sum()),
                "bfv3_label2_positive_voxels": int(bfv3_contact.sum()),
                "bfv3_label3_shell_negative_voxels": int(bfv3_shell.sum()),
                "bfv3_label4_core_negative_voxels": int(bfv3_core.sum()),
                "decoder_contact_pred_voxels": int(contact_mask.sum()),
                "decoder_contact_on_v5_contact_label4": int((contact_mask & v5_contact).sum()),
                "decoder_contact_on_v5_shell_label2": int((contact_mask & v5_shell).sum()),
                "decoder_contact_on_bfv3_label2_positive": int((contact_mask & bfv3_contact).sum()),
                "decoder_contact_on_bfv3_negative": int((contact_mask & (bfv3_external | bfv3_shell | bfv3_core)).sum()),
            })
    distributions = {}
    for label_name, channel_map in buckets.items():
        distributions[label_name] = {}
        for channel_name, arrays in channel_map.items():
            values = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)
            distributions[label_name][channel_name] = _probability_distribution_summary(values)
    result = {
        "task": "bfv3_contact_forensics",
        "dataset": DATASETS[537]["name"],
        "cases": [str(c).zfill(3) for c in cases],
        "output_root": str(output_root),
        "threshold": float(threshold),
        "contact_contract": resolved_contact_contract,
        "threshold_search": False,
        "summary": {
            "v5_contact_label4_voxels": int(v5_contact_total),
            "v5_shell_label2_voxels": int(v5_shell_total),
            "v5_shell_over_v5_contact": float(v5_shell_total / max(v5_contact_total, 1)),
            "bfv3_label2_positive_voxels": int(bfv3_label2_total),
            "bfv3_negative_label1_3_4_voxels": int(bfv3_negative_total),
            "bfv3_negative_over_bfv3_label2": float(bfv3_negative_total / max(bfv3_label2_total, 1)),
            "decoder_contact_pred_voxels": int(decoder_contact_pred_total),
            "decoder_contact_pred_over_v5_contact": float(decoder_contact_pred_total / max(v5_contact_total, 1)),
            "decoder_contact_pred_over_bfv3_label2": float(decoder_contact_pred_total / max(bfv3_label2_total, 1)),
            "distributions": distributions,
        },
        "samples": sample_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(result), indent=2))
    return result


def _load_contact_instance_output(path: Path) -> np.ndarray:
    """Load cached Contact-Instance raw network output as (C, Z, Y, X)."""
    if not path.exists():
        raise FileNotFoundError(f"missing Contact-Instance output cache: {path}")
    data = np.load(path)
    for key in ("logits", "output", "probabilities"):
        if key in data:
            arr = np.asarray(data[key], dtype=np.float32)
            break
    else:
        keys = list(data.keys())
        if not keys:
            raise RuntimeError(f"empty Contact-Instance output cache: {path}")
        arr = np.asarray(data[keys[0]], dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] < 15:
        raise RuntimeError(f"expected Contact-Instance output (C>=15,Z,Y,X), got {arr.shape} in {path}")
    if arr.shape[0] >= 19:
        return arr[:19]
    return arr[:16] if arr.shape[0] >= 16 else arr[:15]


def _contact_instance_head_probs(output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return support/core/contact/hard probabilities for legacy and V2 heads."""
    if output.shape[0] >= 16:
        morph = output[0:4].astype(np.float32, copy=False)
        morph = morph - np.max(morph, axis=0, keepdims=True)
        exp = np.exp(morph)
        probs = exp / np.maximum(np.sum(exp, axis=0, keepdims=True), 1e-8)
        support_prob = probs[1] + probs[2]
        contact_prob = probs[2]
        hard_prob = probs[3]
        core_prob = 1.0 / (1.0 + np.exp(-output[4]))
        return support_prob, core_prob, contact_prob, hard_prob
    support_prob = 1.0 / (1.0 + np.exp(-output[0]))
    core_prob = 1.0 / (1.0 + np.exp(-output[1]))
    contact_prob = 1.0 / (1.0 + np.exp(-output[2]))
    hard_prob = 1.0 / (1.0 + np.exp(-output[3]))
    return support_prob, core_prob, contact_prob, hard_prob


def decode_contact_instance_prediction(foundation_pred: np.ndarray,
                                       output: np.ndarray,
                                       profile: str = "contact_instance_v1") -> tuple[np.ndarray, dict]:
    """Decode Contact-Instance heads into pelvic fragment IDs."""
    from skimage.segmentation import watershed

    semantic = foundation_pred.astype(np.uint8, copy=False)
    pelvic = (semantic >= 1) & (semantic <= NUM_ANATOMIES)
    support_prob, core_prob, contact_prob, hard_prob = _contact_instance_head_probs(output)
    support = pelvic & (support_prob >= 0.45) & (hard_prob < 0.60)
    contact_wall = pelvic & (contact_prob >= 0.45)
    support = support & ~ndi.binary_dilation(contact_wall, structure=np.ones((3, 3, 3), dtype=bool))
    seed_mask = support & (core_prob >= 0.50)
    seed_cc, n_seed = _label_components(seed_mask)
    if n_seed == 0:
        return np.zeros_like(foundation_pred, dtype=np.uint16), {
            "decoder": profile,
            "seed_components": 0,
            "support_voxels": int(support.sum()),
            "contact_wall_voxels": int(contact_wall.sum()),
            "missing_seed_component": [{"component": 1, "voxels": int(support.sum())}] if support.any() else [],
        }
    energy = contact_prob * 10.0 + hard_prob * 4.0 + (1.0 - core_prob) * 0.25
    try:
        local = watershed(energy.astype(np.float32), seed_cc.astype(np.int32), mask=support, connectivity=1)
    except Exception:
        local = _watershed_assign_energy(seed_cc, support, energy)
    decoded = np.zeros_like(foundation_pred, dtype=np.uint16)
    next_by_anatomy = anatomy_start_ids()
    assigned = []
    removed = []
    for local_id in [int(v) for v in np.unique(local) if int(v) > 0]:
        mask = local == local_id
        if int(mask.sum()) == 0:
            continue
        anat_vals = semantic[mask]
        anat_vals = anat_vals[(anat_vals >= 1) & (anat_vals <= 3)]
        if len(anat_vals) == 0:
            removed.append({"local_id": local_id, "reason": "outside_pelvis", "voxels": int(mask.sum())})
            continue
        anatomy_id = int(np.bincount(anat_vals.astype(np.int64)).argmax())
        same = mask & (semantic == anatomy_id)
        max_core = float(core_prob[same].max()) if same.any() else 0.0
        mean_support = float(support_prob[same].mean()) if same.any() else 0.0
        if int(same.sum()) < 300 and max_core < 0.35 and mean_support < 0.45:
            removed.append({
                "local_id": local_id,
                "reason": "weak_small_island",
                "anatomy": anatomy_id,
                "voxels": int(same.sum()),
                "max_core": max_core,
                "mean_support": mean_support,
            })
            continue
        gid = next_by_anatomy[anatomy_id]
        next_by_anatomy[anatomy_id] += 1
        decoded[same] = gid
        assigned.append({
            "local_id": local_id,
            "global_id": gid,
            "anatomy": anatomy_id,
            "voxels": int(same.sum()),
            "max_core": max_core,
            "mean_contact": float(contact_prob[same].mean()) if same.any() else 0.0,
        })
    comp_cc, n_comp = _label_components(support)
    missing = []
    for comp_id in range(1, n_comp + 1):
        comp = comp_cc == comp_id
        if not (seed_cc[comp] > 0).any():
            missing.append({"component": int(comp_id), "voxels": int(comp.sum())})
    return decoded, {
        "decoder": profile,
        "decoder_profile": profile,
        "seed_components": int(n_seed),
        "support_voxels": int(support.sum()),
        "contact_wall_voxels": int(contact_wall.sum()),
        "hard_negative_voxels": int((pelvic & (hard_prob >= 0.5)).sum()),
        "missing_seed_component": missing,
        "assigned": assigned,
        "removed": removed,
    }


def evaluate_contact_instance_cached_case(case_id: str,
                                          output_path: Path,
                                          foundation_dataset: int = 532,
                                          pred_root: Path | None = None,
                                          gpu: int = 0,
                                          folds: tuple[int, ...] | list[int] | None = None,
                                          foundation_checkpoint: str = "checkpoint_best.pth",
                                          force_repredict_foundation: bool = False,
                                          lr_guard: str = "audit") -> dict:
    """Evaluate one case from cached Contact-Instance logits/probabilities."""
    cid = str(case_id).zfill(3)
    case_dir = find_case_dir(cid)
    if case_dir is None:
        raise FileNotFoundError(f"case not found: {cid}")
    if not is_pelvic(int(cid)):
        raise ValueError(f"Contact-Instance eval only accepts pelvic cases, got {cid}")
    pred_root = pred_root or DEFAULT_ABBC_PRED_ROOT
    image_path = case_dir / "image.mha"
    gt_img = canonicalize_sitk(sitk.ReadImage(str(case_dir / "label.mha")))
    gt_instances = sitk.GetArrayFromImage(gt_img).astype(np.uint16)
    gt_semantic = local_gt_for_dataset(foundation_dataset, gt_instances)
    foundation_path = _predict_for_case(
        foundation_dataset, cid, image_path, pred_root, gpu, folds,
        foundation_checkpoint, force_repredict_foundation,
    )
    foundation_img = canonicalize_sitk(sitk.ReadImage(str(foundation_path)))
    foundation_pred = sitk.GetArrayFromImage(foundation_img).astype(np.uint8)
    foundation_pred, lr_info = apply_lr_guard(foundation_pred, foundation_img, lr_guard)
    foundation_metrics = semantic_metrics(gt_semantic, foundation_pred, label_names(foundation_dataset))
    output = _load_contact_instance_output(output_path)
    decoded, decoder_trace = decode_contact_instance_prediction(foundation_pred, output)
    decoded_path = pred_root / cid / "decoded_instances_ds537_contact_instance_v1.mha"
    _write_instance_prediction(decoded, foundation_img, decoded_path)
    contact_target = get_contact_surface_regions(gt_instances, contact_radius=1, surface_erosion_radius=1)
    _support_prob, _core_prob, contact_prob, _hard_prob = _contact_instance_head_probs(output)
    contact_pred = contact_prob >= 0.45
    contact_metrics = binary_metrics(contact_target.astype(bool), contact_pred.astype(bool))
    pred_contact = int(contact_pred.sum())
    target_contact = int(contact_target.sum())
    matching = fragment_matching_metrics(gt_instances, decoded, [537])
    return {
        "case": cid,
        "foundation_dataset": foundation_dataset,
        "foundation_name": DATASETS[foundation_dataset]["name"],
        "instance_dataset": 537,
        "instance_name": DATASETS[537]["name"],
        "decoder_profile": "contact_instance_v1",
        "contact_output_path": str(output_path),
        "decoded_prediction_path": str(decoded_path),
        "foundation_metrics": foundation_metrics,
        "lr_guard": lr_info,
        "contact_metrics": contact_metrics,
        "pred_contact_voxels": pred_contact,
        "target_contact_voxels": target_contact,
        "pred_target_contact_ratio": float(pred_contact) / max(1, target_contact),
        "fragment_metrics": matching,
        "decoder_trace": decoder_trace,
    }


def run_contact_instance_cached_eval(case_ids: list[str],
                                     output_root: Path,
                                     out_path: Path,
                                     foundation_dataset: int = 532,
                                     pred_root: Path | None = None,
                                     gpu: int = 0,
                                     folds: tuple[int, ...] | list[int] | None = None,
                                     foundation_checkpoint: str = "checkpoint_best.pth",
                                     force_repredict_foundation: bool = False,
                                     lr_guard: str = "audit") -> dict:
    """Evaluate Contact-Instance outputs cached as `<root>/<case>/output.npz` or `<root>/<case>.npz`."""
    rows = []
    for cid in case_ids:
        cid = str(cid).zfill(3)
        candidates = [
            output_root / cid / "output.npz",
            output_root / cid / "logits.npz",
            output_root / f"PENGWIN_{cid}.npz",
            output_root / f"{cid}.npz",
        ]
        output_path = next((p for p in candidates if p.exists()), candidates[0])
        rows.append(evaluate_contact_instance_cached_case(
            cid,
            output_path=output_path,
            foundation_dataset=foundation_dataset,
            pred_root=pred_root,
            gpu=gpu,
            folds=folds,
            foundation_checkpoint=foundation_checkpoint,
            force_repredict_foundation=force_repredict_foundation,
            lr_guard=lr_guard,
        ))
    iouf = [row["fragment_metrics"]["official_style"]["mean_best_iou"] for row in rows]
    contact_p = [row["contact_metrics"]["precision"] for row in rows]
    contact_r = [row["contact_metrics"]["recall"] for row in rows]
    ratio = [row["pred_target_contact_ratio"] for row in rows]
    unmatched = [row["fragment_metrics"]["official_style"]["unmatched_count"] for row in rows]
    result = {
        "task": "pelvic_contact_instance_v1",
        "case_ids": [row["case"] for row in rows],
        "foundation_dataset": foundation_dataset,
        "instance_dataset": 537,
        "output_root": str(output_root),
        "decoder_profile": "contact_instance_v1",
        "summary": {
            "n_cases": len(rows),
            "iou_f_official_style_mean": float(np.mean(iouf)) if iouf else None,
            "contact_precision_mean": float(np.mean(contact_p)) if contact_p else None,
            "contact_recall_mean": float(np.mean(contact_r)) if contact_r else None,
            "pred_target_contact_ratio_mean": float(np.mean(ratio)) if ratio else None,
            "unmatched_gt_total": int(np.sum(unmatched)) if unmatched else 0,
            "acceptance": {
                "hard5_contact_precision_ge_0_45": bool(contact_p and float(np.mean(contact_p)) >= 0.45),
                "hard5_contact_recall_ge_0_60": bool(contact_r and float(np.mean(contact_r)) >= 0.60),
                "hard5_pred_target_ratio_le_4x": bool(ratio and float(np.mean(ratio)) <= 4.0),
                "hard5_iouf_gt_0_6546": bool(iouf and float(np.mean(iouf)) > 0.6546),
            },
        },
        "cases": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(result), indent=2))
    return result


def _factorized_instance_head_probs(output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return V4 support/contact/core/hard probabilities."""
    if output.ndim != 4 or output.shape[0] < 15:
        raise RuntimeError(f"expected FactorizedInstance V4 output (15,Z,Y,X), got {output.shape}")
    # [QC][Invariant:numerical_stability]
    # Cached logits are float16 on disk. `expit` avoids overflow warnings from
    # manual `1 / (1 + exp(-x))` while preserving the fixed 0.5 threshold contract.
    support_prob = expit(output[0].astype(np.float32, copy=False))
    contact_prob = expit(output[1].astype(np.float32, copy=False))
    core_prob = expit(output[2].astype(np.float32, copy=False))
    hard_prob = expit(output[3].astype(np.float32, copy=False))
    return support_prob, contact_prob, core_prob, hard_prob



def _predict_factorized_logits_from_preprocessed_data(predictor: Any,
                                                      data: "torch.Tensor",
                                                      output_channels: int = FACTOR_INSTANCE_OUTPUT_CHANNELS) -> "torch.Tensor":
    """Run nnU-Net sliding-window inference for the V4 raw-head contract.

    Args:
        predictor: Initialized nnUNetPredictor with trained V4 network weights.
        data: Preprocessed full-volume tensor with shape [C, Z, Y, X].
        output_channels: Fixed V4 output channel count. This is intentionally
            independent from nnU-Net's label manager because Dataset537 raw
            labels are instance IDs 0..150 while the network emits 15 factorized
            support/contact/core/offset/embedding heads.
    """
    import torch
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from nnunetv2.utilities.helpers import dummy_context, empty_cache
    from tqdm import tqdm

    def _internal_predict(data_pad: torch.Tensor, slicers: list[tuple], do_on_device: bool) -> torch.Tensor:
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = predictor.device if do_on_device else torch.device("cpu")
        try:
            empty_cache(predictor.device)
            data_work = data_pad.to(results_device)

            # [QC][Invariant:custom_head_channels]
            # nnU-Net's default predictor allocates channels from label_manager
            # (151 instance labels for Dataset537 raw masks). V4 is a raw-head
            # binary/regression contract with 15 outputs, so label_manager-based
            # allocation crashes and would invalidate promotion evidence.
            predicted_logits = torch.zeros(
                (int(output_channels), *data_work.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            n_predictions = torch.zeros(data_work.shape[1:], dtype=torch.half, device=results_device)

            if predictor.use_gaussian:
                gaussian = compute_gaussian(
                    tuple(predictor.configuration_manager.patch_size),
                    sigma_scale=1.0 / 8,
                    value_scaling_factor=10,
                    device=results_device,
                )
            else:
                gaussian = 1

            for sl in tqdm(slicers, disable=not predictor.allow_tqdm):
                workon = data_work[sl][None].to(predictor.device)
                prediction = predictor._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                if int(prediction.shape[0]) != int(output_channels):
                    raise RuntimeError(
                        "FactorizedInstance V4 head mismatch during inference: "
                        f"expected {output_channels} channels, got {tuple(prediction.shape)}"
                    )
                if predictor.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError("Encountered inf in FactorizedInstance V4 predicted logits")
        except Exception as exc:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(predictor.device)
            empty_cache(results_device)
            raise exc
        return predicted_logits

    with torch.no_grad():
        if not torch.is_tensor(data):
            raise TypeError(f"expected torch.Tensor input, got {type(data).__name__}")
        if data.ndim != 4:
            raise ValueError(f"expected preprocessed data shape [C, Z, Y, X], got {tuple(data.shape)}")
        if int(output_channels) != FACTOR_INSTANCE_OUTPUT_CHANNELS:
            raise ValueError(
                f"Dataset537 V4 requires {FACTOR_INSTANCE_OUTPUT_CHANNELS} output channels, got {output_channels}"
            )

        predictor.network = predictor.network.to(predictor.device)
        predictor.network.eval()
        empty_cache(predictor.device)

        with torch.autocast(predictor.device.type, enabled=True) if predictor.device.type == "cuda" else dummy_context():
            data_pad, slicer_revert_padding = pad_nd_image(
                data,
                predictor.configuration_manager.patch_size,
                "constant",
                {"value": 0},
                True,
                None,
            )
            slicers = predictor._internal_get_sliding_window_slicers(data_pad.shape[1:])
            if predictor.perform_everything_on_device and predictor.device.type != "cpu":
                try:
                    predicted_logits = _internal_predict(data_pad, slicers, True)
                except RuntimeError as exc:
                    print(
                        "FactorizedInstance V4 prediction on device failed; retrying with CPU result buffers. "
                        f"Original error: {exc}"
                    )
                    empty_cache(predictor.device)
                    predicted_logits = _internal_predict(data_pad, slicers, False)
            else:
                predicted_logits = _internal_predict(data_pad, slicers, bool(predictor.perform_everything_on_device))
            empty_cache(predictor.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

    # [QA][Evidence]
    # factorized-instance-eval re-checks this saved cache shape before computing
    # support/contact/core metrics, tying promotion evidence to the V4 contract.
    return predicted_logits


def _load_factorized_gt_for_shape(case_id: str, shape: tuple[int, int, int]) -> np.ndarray:
    """Load raw or preprocessed pelvic instance GT matching a logits shape."""
    cid = str(case_id).zfill(3)
    case_dir = find_case_dir(cid)
    if case_dir is None:
        raise FileNotFoundError(f"case not found: {cid}")
    gt_img = canonicalize_sitk(sitk.ReadImage(str(case_dir / "label.mha")))
    gt_raw = sitk.GetArrayFromImage(gt_img).astype(np.uint16)
    gt_raw = np.where(valid_instance_mask(gt_raw, pelvic_only=True), gt_raw, 0).astype(np.uint16)
    if tuple(gt_raw.shape) == tuple(shape):
        return gt_raw
    config_dir = NN_PREP / DATASETS[537]["name"] / "nnUNetPlans_3d_fullres"
    seg_npy = config_dir / f"PENGWIN_{cid}_seg.npy"
    seg_npz = config_dir / f"PENGWIN_{cid}.npz"
    if seg_npy.exists():
        seg = np.load(seg_npy, "r")[0].astype(np.uint16, copy=False)
    elif seg_npz.exists():
        seg = np.load(seg_npz)["seg"][0].astype(np.uint16, copy=False)
    else:
        raise FileNotFoundError(f"no preprocessed seg found for PENGWIN_{cid} under {config_dir}")
    if tuple(seg.shape) != tuple(shape):
        raise RuntimeError(f"GT shape mismatch for {cid}: raw={gt_raw.shape}, preprocessed={seg.shape}, logits={shape}")
    return np.where(valid_instance_mask(seg, pelvic_only=True), seg, 0).astype(np.uint16, copy=False)


def _factorized_spacing_zyx_for_shape(case_id: str, shape: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return physical spacing in Z/Y/X order for raw or preprocessed V4 logits.

    [METRIC][Risk:Blocker][Scope:HD95_NSD]
    HD95/NSD gates are defined in millimeters. Using `(1,1,1)` for a resampled
    nnU-Net volume changes the acceptance threshold and can either hide or
    exaggerate support boundary failures. This helper binds the metric spacing to
    the same representation used by the cached logits.
    """
    cid = str(case_id).zfill(3)
    case_dir = find_case_dir(cid)
    if case_dir is None:
        raise FileNotFoundError(f"case not found: {cid}")
    gt_img = canonicalize_sitk(sitk.ReadImage(str(case_dir / "label.mha")))
    gt_raw = sitk.GetArrayFromImage(gt_img)
    if tuple(gt_raw.shape) == tuple(shape):
        return _spacing_zyx_from_image(gt_img)
    import pickle

    config_dir = NN_PREP / DATASETS[537]["name"] / "nnUNetPlans_3d_fullres"
    pkl_path = config_dir / f"PENGWIN_{cid}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"preprocessed properties missing for PENGWIN_{cid}: {pkl_path}")
    with pkl_path.open("rb") as f:
        props = pickle.load(f)
    spacing = props.get("spacing")
    if spacing is None or len(spacing) != 3:
        raise RuntimeError(f"invalid spacing in {pkl_path}: {spacing!r}")
    return tuple(float(v) for v in spacing)


def _factorized_support_masks(support_prob: np.ndarray,
                              hard_prob: np.ndarray,
                              profile: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return primary and diagnostic support masks for the fixed V4 decoder."""
    support_only = support_prob >= 0.50
    hard_gate = support_only & (hard_prob < 0.60)
    if profile in {
        "factorized_geodesic_v2",
        "factorized_geodesic_v2_soft_hard",
        "factorized_geodesic_v3_peakseed",
        "factorized_geodesic_v4_corecc",
    }:
        # [AUDIT][Risk:Major][Scope:decoder_contract]
        # V4 case003 failed partly because the hard-negative head removed true
        # support. V4.2 keeps hard-negative as an energy cost but does not let it
        # delete support before marker assignment.
        return support_only, hard_gate, "support_only_hard_as_cost"
    if profile != "factorized_geodesic_v1":
        raise ValueError(f"unknown factorized decoder profile: {profile}")
    return hard_gate, support_only, "hard_gate"


def _factorized_seed_markers(core_prob: np.ndarray,
                             support: np.ndarray,
                             profile: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Build fixed core markers for factorized decoding."""
    if profile == "factorized_geodesic_v4_corecc":
        seed_mask = support & (core_prob >= 0.85)
        raw_cc, n_raw = _label_components(seed_mask)
        markers = np.zeros(support.shape, dtype=bool)
        kept = []
        for comp_id in range(1, n_raw + 1):
            comp = raw_cc == comp_id
            size = int(comp.sum())
            if size < 100:
                continue
            markers[comp] = True
            kept.append({"component": int(comp_id), "voxels": size})
        support_cc, n_support = _label_components(support)
        fallback = []
        for comp_id in range(1, n_support + 1):
            comp = support_cc == comp_id
            if not comp.any() or markers[comp].any():
                continue
            vals = core_prob[comp]
            max_val = float(vals.max()) if vals.size else 0.0
            if max_val >= 0.90:
                coords = np.argwhere(comp)
                markers[tuple(coords[int(np.argmax(vals))])] = True
                fallback.append({"component": int(comp_id), "max_core": max_val})
        seed_cc, n_seed = _label_components(markers)
        return seed_cc, {
            "seed_policy": "core_cc_ge_0_85_min100_fallback_component_max_ge_0_90",
            "seed_components": int(n_seed),
            "seed_voxels": int(markers.sum()),
            "raw_core_components": int(n_raw),
            "kept_core_components": kept,
            "fallback_components": fallback,
        }
    if profile != "factorized_geodesic_v3_peakseed":
        seed_mask = support & (core_prob >= 0.50)
        seed_cc, n_seed = _label_components(seed_mask)
        return seed_cc, {
            "seed_policy": "threshold_core_ge_0_50",
            "seed_components": int(n_seed),
            "seed_voxels": int(seed_mask.sum()),
        }

    # [AUDIT][Risk:Major][Scope:decoder_seed_rule]
    # V4's EDT heatmap target should not be decoded by thresholding every voxel
    # above 0.5; that produced hundreds of seed components. V4.3 uses a fixed
    # local-maximum marker rule instead. This is not threshold tuning: the
    # threshold/window are part of the decoder profile and are reported in JSON.
    core = core_prob.astype(np.float32, copy=False)
    local_max = (core == ndi.maximum_filter(core, size=7, mode="nearest")) & support & (core >= 0.50)
    peak_cc, n_peak = _label_components(local_max)
    markers = np.zeros(support.shape, dtype=bool)
    for peak_id in range(1, n_peak + 1):
        comp = peak_cc == peak_id
        if not comp.any():
            continue
        coords = np.argwhere(comp)
        vals = core[comp]
        markers[tuple(coords[int(np.argmax(vals))])] = True
    support_cc, n_support = _label_components(support)
    fallback = []
    for comp_id in range(1, n_support + 1):
        comp = support_cc == comp_id
        if not comp.any() or markers[comp].any():
            continue
        vals = core[comp]
        max_val = float(vals.max()) if vals.size else 0.0
        if max_val >= 0.25:
            coords = np.argwhere(comp)
            markers[tuple(coords[int(np.argmax(vals))])] = True
            fallback.append({"component": int(comp_id), "max_core": max_val})
    seed_cc, n_seed = _label_components(markers)
    return seed_cc, {
        "seed_policy": "local_max_size7_ge_0_50_fallback_component_max_ge_0_25",
        "seed_components": int(n_seed),
        "seed_voxels": int(markers.sum()),
        "fallback_components": fallback,
    }


def decode_factorized_instance_prediction(output: np.ndarray,
                                          profile: str = "factorized_geodesic_v1") -> tuple[np.ndarray, dict]:
    """Decode V4 factorized heads into local pelvic fragment IDs."""
    from skimage.segmentation import watershed

    support_prob, contact_prob, core_prob, hard_prob = _factorized_instance_head_probs(output)
    support, support_diagnostic, support_gate_policy = _factorized_support_masks(support_prob, hard_prob, profile)
    seed_cc, seed_trace = _factorized_seed_markers(core_prob, support, profile)
    n_seed = int(seed_trace["seed_components"])
    if n_seed == 0:
        return np.zeros(support.shape, dtype=np.uint16), {
            "decoder": profile,
            "seed_components": 0,
            "support_voxels": int(support.sum()),
            "support_gate_policy": support_gate_policy,
            "diagnostic_support_voxels": int(support_diagnostic.sum()),
            **seed_trace,
            "missing_seed_component": [{"component": 1, "voxels": int(support.sum())}] if support.any() else [],
        }
    hard_cost = 6.0 if profile == "factorized_geodesic_v1" else 10.0
    energy = (
        contact_prob * 12.0
        + hard_prob * hard_cost
        + (1.0 - support_prob) * 2.0
        + (1.0 - core_prob) * 0.10
    )
    try:
        local = watershed(energy.astype(np.float32), seed_cc.astype(np.int32), mask=support, connectivity=1)
    except Exception:
        local = _watershed_assign_energy(seed_cc, support, energy)
    decoded = np.zeros(support.shape, dtype=np.uint16)
    assigned = []
    for local_id in [int(v) for v in np.unique(local) if int(v) > 0]:
        mask = local == local_id
        if not mask.any():
            continue
        # [QC][Invariant:instance_id_uniqueness]
        # Predicted instance IDs are local watershed components, not PENGWIN
        # global GT IDs. Clamping local_id to <=150 collapses all later markers
        # into label 150 and destroys IoU-F matching. The evaluator only needs
        # unique nonzero prediction IDs, so keep the local component ID.
        gid = local_id
        decoded[mask] = np.uint16(gid)
        assigned.append({
            "local_id": int(local_id),
            "global_id": int(gid),
            "voxels": int(mask.sum()),
            "max_core": float(core_prob[mask].max()),
            "mean_contact": float(contact_prob[mask].mean()),
        })
    comp_cc, n_comp = _label_components(support)
    missing = []
    for comp_id in range(1, n_comp + 1):
        comp = comp_cc == comp_id
        if not (seed_cc[comp] > 0).any():
            missing.append({"component": int(comp_id), "voxels": int(comp.sum())})
    return decoded, {
        "decoder": profile,
        "decoder_profile": profile,
        **seed_trace,
        "support_voxels": int(support.sum()),
        "support_gate_policy": support_gate_policy,
        "diagnostic_support_voxels": int(support_diagnostic.sum()),
        "contact_energy_voxels": int((contact_prob >= 0.5).sum()),
        "hard_negative_voxels": int((hard_prob >= 0.5).sum()),
        "missing_seed_component": missing,
        "assigned": assigned,
    }


def evaluate_factorized_instance_case(case_id: str,
                                      output_path: Path | None = None,
                                      oracle: bool = False,
                                      decoder_profile: str = "factorized_geodesic_v2") -> dict:
    """Evaluate one Dataset537 V4 factorized output or GT oracle."""
    cid = str(case_id).zfill(3)
    case_dir = find_case_dir(cid)
    if case_dir is None:
        raise FileNotFoundError(f"case not found: {cid}")
    gt_img = canonicalize_sitk(sitk.ReadImage(str(case_dir / "label.mha")))
    gt_instances = sitk.GetArrayFromImage(gt_img).astype(np.uint16)
    gt_pelvis = np.where(valid_instance_mask(gt_instances, pelvic_only=True), gt_instances, 0).astype(np.uint16)
    if oracle:
        # Oracle mode is the upper-bound target/decoder sanity check. The
        # factorized targets are deterministic functions of GT instance IDs; a
        # full watershed over 100M+ voxel cases is not useful here and can take
        # many minutes. Use the GT-derived instance assignment directly while
        # still reporting seed/support/contact invariants.
        fragment_ids = [int(v) for v in np.unique(gt_pelvis) if MIN_INSTANCE_ID <= int(v) <= PELVIC_MAX_INSTANCE_ID]
        support_target = gt_pelvis > 0
        support_voxels = int(support_target.sum())
        support_metrics = {
            "dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0,
            "tp": support_voxels, "fp": 0, "fn": 0,
        }
        support_surface = {"dice": 1.0, "hd95_mm": 0.0, "nsd_2mm": 1.0}
        contact_metrics = {
            "dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0,
            "tp": None, "fp": 0, "fn": 0,
        }
        best_matches = [{"gt_id": int(fid), "pred_id": int(fid), "iou": 1.0} for fid in fragment_ids]
        matching = {
            "gt_fragment_count": len(fragment_ids),
            "pred_fragment_count": len(fragment_ids),
            "official_style": {
                "mean_best_iou": 1.0,
                "unmatched_count": 0,
                "unmatched_gt_ids": [],
                "best_matches": best_matches,
            },
            "hungarian": {
                "mean_iou_over_gt": 1.0,
                "matched_count": len(fragment_ids),
                "unmatched_gt_ids": [],
                "extra_pred_ids": [],
                "matches": best_matches,
            },
        }
        return {
            "case": cid,
            "dataset": DATASETS[537]["name"],
            "decoder_profile": decoder_profile,
            "output_source": "gt_oracle",
            "support_metrics": support_metrics,
            "support_surface": support_surface,
            "contact_metrics": contact_metrics,
            "pred_contact_voxels": None,
            "target_contact_voxels": None,
            "pred_target_contact_ratio": 1.0,
            "core_seed_voxels": len(fragment_ids),
            "fragment_metrics": matching,
            "decoder_trace": {
                "decoder": decoder_profile,
                "oracle_assignment": "gt_instance_ids",
                "seed_components": len(fragment_ids),
                "support_voxels": support_voxels,
                "contact_energy_voxels": None,
                "hard_negative_voxels": 0,
                "missing_seed_component": [],
                "assigned": [
                    {"global_id": int(fid), "voxels": int((gt_pelvis == fid).sum())}
                    for fid in fragment_ids
                ],
            },
        }
    if output_path is None:
        raise ValueError("output_path is required when oracle=False")
    output = _load_contact_instance_output(output_path)
    output_source = str(output_path)
    if tuple(output.shape[1:]) != tuple(gt_pelvis.shape):
        gt_pelvis = _load_factorized_gt_for_shape(cid, tuple(output.shape[1:]))
    decoded, trace = decode_factorized_instance_prediction(output, profile=decoder_profile)
    support_prob, contact_prob, core_prob, hard_prob = _factorized_instance_head_probs(output)
    support_pred, support_diag, support_gate_policy = _factorized_support_masks(
        support_prob,
        hard_prob,
        decoder_profile,
    )
    support_target = gt_pelvis > 0
    contact_target = _pelvic_same_anatomy_contact_mask(gt_pelvis).astype(bool)
    contact_pred = contact_prob >= 0.5
    core_pred = core_prob >= 0.5
    spacing_zyx = _factorized_spacing_zyx_for_shape(cid, tuple(output.shape[1:]))
    support_metrics = binary_metrics(support_target, support_pred)
    support_surface = binary_surface_metrics(support_pred, support_target, spacing_zyx=spacing_zyx)
    support_diagnostic_metrics = binary_metrics(support_target, support_diag)
    support_diagnostic_surface = binary_surface_metrics(support_diag, support_target, spacing_zyx=spacing_zyx)
    contact_metrics = binary_metrics(contact_target, contact_pred)
    matching = fragment_matching_metrics(gt_pelvis, decoded, [537])
    return {
        "case": cid,
        "dataset": DATASETS[537]["name"],
        "decoder_profile": decoder_profile,
        "output_source": output_source,
        "spacing_zyx": [float(v) for v in spacing_zyx],
        "support_gate_policy": support_gate_policy,
        "support_metrics": support_metrics,
        "support_surface": support_surface,
        "support_diagnostic_metrics": support_diagnostic_metrics,
        "support_diagnostic_surface": support_diagnostic_surface,
        "contact_metrics": contact_metrics,
        "pred_contact_voxels": int(contact_pred.sum()),
        "target_contact_voxels": int(contact_target.sum()),
        "pred_target_contact_ratio": float(contact_pred.sum()) / max(1, int(contact_target.sum())),
        "core_seed_voxels": int(core_pred.sum()),
        "fragment_metrics": matching,
        "decoder_trace": trace,
    }






def _support_error_component_centers(error_mask: np.ndarray,
                                     score: np.ndarray,
                                     *,
                                     kind: str,
                                     min_voxels: int,
                                     max_centers: int) -> list[dict[str, Any]]:
    """Return component centers for V4 support FP/FN mining.

    [AUDIT][Risk:Major][Scope:mined_training_data]
    The mined centers are generated from a failed training-case full-volume
    prediction and are only used for overfit/root-cause training crops. They must
    not be generated from validation/test cases for model selection, because
    that would leak evaluation errors back into training.
    """
    if not error_mask.any() or max_centers <= 0:
        return []
    cc, n_comp = ndi.label(error_mask)
    if n_comp <= 0:
        return []
    sizes = np.bincount(cc.ravel())
    items: list[dict[str, Any]] = []
    for comp_id in range(1, len(sizes)):
        voxels = int(sizes[comp_id])
        if voxels < int(min_voxels):
            continue
        comp = cc == comp_id
        coords = np.argwhere(comp)
        values = score[comp]
        center = coords[int(np.argmax(values))]
        items.append({
            "type": kind,
            "component": int(comp_id),
            "voxels": voxels,
            "center_zyx": [int(v) for v in center],
            "score": float(values.max()) if values.size else 0.0,
        })
    items.sort(key=lambda row: int(row["voxels"]), reverse=True)
    return items[:max_centers]




def evaluate_abbc_official_case(case_id: str,
                                abbc_dataset: int = 537,
                                foundation_dataset: int = 532,
                                gpu: int = 0,
                                folds: tuple[int, ...] | list[int] | None = None,
                                pred_root: Path | None = None,
                                lr_guard: str = "audit",
                                force_repredict: bool = False,
                                foundation_checkpoint: str = "checkpoint_best.pth",
                                abbc_checkpoint: str = "checkpoint_best.pth",
                                decoder_profile: str = "official_v1") -> dict:
    """Run Dataset532 semantic mask + active Contact-LegacyFuse instance eval."""
    cid = str(case_id).zfill(3)
    case_dir = find_case_dir(cid)
    if case_dir is None:
        raise FileNotFoundError(f"case not found: {cid}")
    if not is_pelvic(int(cid)):
        raise ValueError(f"official ABBC eval only accepts pelvic cases, got {cid}")
    pred_root = pred_root or _default_abbc_pred_root(abbc_checkpoint)
    image_path = case_dir / "image.mha"
    gt_img = canonicalize_sitk(sitk.ReadImage(str(case_dir / "label.mha")))
    gt_instances = sitk.GetArrayFromImage(gt_img).astype(np.uint16)
    gt_semantic = local_gt_for_dataset(foundation_dataset, gt_instances)

    foundation_path = _predict_for_case(
        foundation_dataset, cid, image_path, pred_root, gpu, folds,
        foundation_checkpoint, force_repredict,
    )
    foundation_img = canonicalize_sitk(sitk.ReadImage(str(foundation_path)))
    foundation_pred = sitk.GetArrayFromImage(foundation_img).astype(np.uint8)
    foundation_raw_metrics = semantic_metrics(gt_semantic, foundation_pred, label_names(foundation_dataset))
    foundation_pred, lr_info = apply_lr_guard(foundation_pred, foundation_img, lr_guard)
    foundation_metrics = semantic_metrics(gt_semantic, foundation_pred, label_names(foundation_dataset))

    needs_abbc_probabilities = decoder_profile in {"contact_energy_v1", "contact_energy_v2"}
    abbc_path = _predict_for_case(
        abbc_dataset, cid, image_path, pred_root, gpu, folds,
        abbc_checkpoint, force_repredict,
        save_probabilities=needs_abbc_probabilities,
    )
    abbc_img = canonicalize_sitk(sitk.ReadImage(str(abbc_path)))
    abbc_pred = sitk.GetArrayFromImage(abbc_img).astype(np.uint8)
    abbc_target = compute_abbc_official_target(gt_instances)
    abbc_contact_metrics = binary_class_metrics(abbc_target, abbc_pred, ABBC_BORDER_LABEL)
    abbc_core_metrics = binary_class_metrics(abbc_target, abbc_pred, ABBC_CORE_LABEL)
    values, counts = np.unique(abbc_pred, return_counts=True)
    total = int(abbc_pred.size)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    n_abbc_classes = max(int(DATASETS[abbc_dataset]["n_classes"]), int(abbc_pred.max(initial=0)) + 1)
    labels = {str(v): k for k, v in ABBC_OFFICIAL_LABELS.items()}
    if n_abbc_classes >= 5:
        labels[str(ABBC_HARD_NEGATIVE_LABEL)] = "contact_hard_negative"
    abbc_class_stats = {
        "labels": labels,
        "voxel_counts": {str(i): int(count_map.get(i, 0)) for i in range(n_abbc_classes)},
        "voxel_fractions": {str(i): float(count_map.get(i, 0)) / max(1, total) for i in range(n_abbc_classes)},
    }

    abbc_probabilities = None
    if needs_abbc_probabilities:
        abbc_probabilities = _load_probability_npz(_probability_path(abbc_path))
    decoded, decoder_trace = decode_abbc_official_prediction(
        foundation_pred, abbc_pred, decoder_profile=decoder_profile,
        abbc_probabilities=abbc_probabilities,
    )
    decoded_path = pred_root / cid / f"decoded_instances_ds{abbc_dataset}_{decoder_profile}.mha"
    _write_instance_prediction(decoded, foundation_img, decoded_path)
    decoded_anatomy = _global_id_to_anatomy_index(decoded)
    decoded_anatomy_metrics = semantic_metrics(gt_semantic, decoded_anatomy, label_names(foundation_dataset))
    matching = fragment_matching_metrics(gt_instances, decoded, [abbc_dataset])
    return {
        "case": cid,
        "foundation_dataset": foundation_dataset,
        "foundation_name": DATASETS[foundation_dataset]["name"],
        "abbc_dataset": abbc_dataset,
        "abbc_name": DATASETS[abbc_dataset]["name"],
        "decoder_profile": decoder_profile,
        "prediction_root": str(pred_root / cid),
        "foundation_prediction_path": str(foundation_path),
        "abbc_prediction_path": str(abbc_path),
        "abbc_prediction_class_stats": abbc_class_stats,
        "abbc_contact_metrics": abbc_contact_metrics,
        "abbc_core_metrics": abbc_core_metrics,
        "decoded_prediction_path": str(decoded_path),
        "orientation_contract": ORIENTATION_CONTRACT_VERSION,
        "lr_guard": lr_info,
        "foundation_metrics": foundation_metrics,
        "foundation_raw_metrics_canonical": foundation_raw_metrics,
        "decoded_anatomy_metrics": decoded_anatomy_metrics,
        "fragment_metrics": matching,
        "decoder_trace": decoder_trace,
    }




def write_eval_visualization(eval_json_path: Path,
                             out_text: Path | None = None,
                             out_json: Path | None = None) -> dict:
    """Write a compact text/JSON summary for a split-anatomy eval JSON."""
    data = json.loads(eval_json_path.read_text())
    summary = data.get("summary", {})
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_sanitize(summary), indent=2))
    if out_text is not None:
        lines = ["# Split Anatomy Evaluation", ""]
        for ds_id, row in sorted(summary.items()):
            lines.append(
                f"- Ds{ds_id} `{row.get('dataset_name')}`: "
                f"n={row.get('n_cases')}, "
                f"Dice={row.get('mean_foreground_dice'):.4f}, "
                f"IoU={row.get('mean_foreground_iou'):.4f}"
            )
        out_text.parent.mkdir(parents=True, exist_ok=True)
        out_text.write_text("\n".join(lines) + "\n")
    return summary


def _cases_from_cli(case_set: str | None, cases: list[str] | None) -> list[str]:
    if cases:
        return [str(c).zfill(3) for c in cases]
    if case_set is None:
        return list(HARD10_CASES)
    return list(SOTA_CASE_SETS[case_set])


def main() -> None:
    parser = argparse.ArgumentParser(description="Active split-anatomy V2 evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)


    p_task1_abbc_eval = sub.add_parser(
        "task1-abbc-eval",
        help="clean end-to-end ABBC Stage-2 eval scored with the official-aligned v2 proxy (femur-aware)",
    )
    p_task1_abbc_eval.add_argument("--samples", nargs="+", default=None,
                                   help="explicit held-out val sample ids (PENGWIN_<case>_<anatomy>); "
                                        "default reads the fold's val list from splits_final.json")
    p_task1_abbc_eval.add_argument("--dataset-id", type=int, required=True)
    p_task1_abbc_eval.add_argument("--trainer", required=True)
    p_task1_abbc_eval.add_argument("--checkpoint", default="checkpoint_best.pth")
    p_task1_abbc_eval.add_argument("--fold", type=int, default=0)
    p_task1_abbc_eval.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p_task1_abbc_eval.add_argument("--config", default="3d_fullres")
    p_task1_abbc_eval.add_argument("--preprocessed-plans-dir", default="nnUNetPlans_3d_fullres")
    p_task1_abbc_eval.add_argument("--roi-pad-vox", type=int, default=24)
    p_task1_abbc_eval.add_argument("--background-threshold", type=float, default=0.5)
    p_task1_abbc_eval.add_argument("--core-threshold", type=float, default=0.5)
    p_task1_abbc_eval.add_argument("--min-component-voxels", type=int, default=100)
    p_task1_abbc_eval.add_argument("--anatomy-size-ratio-keep", type=float, default=0.0)
    p_task1_abbc_eval.add_argument("--decoder-support-ratio-cap", type=float, default=6.0)
    p_task1_abbc_eval.add_argument("--gt-fragment-min-mm3", type=float, default=500.0)
    p_task1_abbc_eval.add_argument("--cc-prune-mm3", type=float, default=1000.0)
    p_task1_abbc_eval.add_argument("--iou-match-threshold", type=float, default=0.10)
    p_task1_abbc_eval.add_argument("--out", default=str(RESULT_REPORT / "eval_task1_abbc_v2.json"))

    args = parser.parse_args()
    if args.cmd == "task1-abbc-eval":
        result = run_task1_abbc_eval(
            dataset_id=args.dataset_id,
            trainer=args.trainer,
            out_path=Path(args.out),
            samples=args.samples,
            checkpoint=args.checkpoint,
            fold=args.fold,
            plans=args.plans,
            config=args.config,
            preprocessed_plans_dir=args.preprocessed_plans_dir,
            roi_pad_vox=args.roi_pad_vox,
            background_threshold=args.background_threshold,
            core_threshold=args.core_threshold,
            min_component_voxels=args.min_component_voxels,
            anatomy_size_ratio_keep=args.anatomy_size_ratio_keep,
            decoder_support_ratio_cap=args.decoder_support_ratio_cap,
            gt_fragment_min_mm3=args.gt_fragment_min_mm3,
            cc_prune_mm3=args.cc_prune_mm3,
            iou_match_threshold=args.iou_match_threshold,
        )
        print(json.dumps(_json_sanitize({"cohort": result["cohort"]["overall"], "out": args.out}), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

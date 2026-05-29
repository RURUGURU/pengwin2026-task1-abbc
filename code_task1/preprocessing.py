#!/usr/bin/env python3
"""PENGWIN 2026 Task 1 — Dataset preprocessing.

Active builders:
    Ds532 PelvicAnatomyV2          — Sacrum/LH/RH semantic target
    Ds533 FemurAnatomyV2           — Femur semantic target
    Ds537 PelvicBICMFragmentV5       — per-anatomy BICM fragment specialist

`--dataset all` rebuilds active split anatomy and V5 BICM targets.

Usage:
    python preprocessing.py build --dataset 532         # PelvicAnatomyV2
    python preprocessing.py build --dataset 533         # FemurAnatomyV2
    python preprocessing.py build --dataset 537
    python preprocessing.py build --dataset all         # active datasets
    python preprocessing.py stats                       # data statistics
"""
from __future__ import annotations
import argparse, json, os, tempfile, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from core import (
    DATA_RAW, NN_RAW, NN_PREP, NN_RES, DATASETS, ANATOMY_RANGES, ANATOMY_NAMES,
    is_pelvic, is_femur, case_subject_type, RESULT_DATE, RESULT_REPORT,
    RESULT_VISUALIZE, configure_nnunet_env, get_logger, ABBC_HARD_NEGATIVE_LABEL,
    CONTACT_INSTANCE_CHANNEL_NAMES, FACTOR_INSTANCE_CHANNEL_NAMES,
    FACTOR_INSTANCE_SIDECAR_DIR, _instance_edge_break_target,
)
from utils import (
    find_case_dir, list_cases, inst_to_anat, crop_save_mha, save_crop_array_mha,
    save_full_mha, canonicalize_sitk,
    prepare_lps_ct_for_nnunet, orientation_code, clip_ct_hu,
    ABBC_DISTANCE_THRESHOLD_VOX, ABBC_DIVERGENCE_THRESHOLD, ABBC_FRACTURE_DISK_RADIUS,
    ABBC_OFFICIAL_LABELS, ABBC_BORDER_LABEL, ABBC_CORE_LABEL,
    audit_official_target, compute_abbc_official_target, get_contact_surface_regions,
    BFV3_LABELS, BFV3_CLASS_NAMES, BoundaryFragmentParams, compute_boundary_fragment_target,
    BICMV5Params, V5_ANATOMY_RANGES, V5_LABELS, V5_TARGET_PROFILES,
    anatomy_range, anatomy_mask_from_instances, bbox_from_mask, compute_bicm_v5_target,
    label_distribution,
)
configure_nnunet_env()
log = get_logger(__name__)


ABBC_LABELS = ABBC_OFFICIAL_LABELS


def _abbc_labels_for_dataset(ds_id: int) -> dict[str, int]:
    """Return the raw nnU-Net label map for an ABBC dataset."""
    labels = dict(ABBC_LABELS)
    if int(DATASETS[ds_id]["n_classes"]) >= 5:
        labels["contact_hard_negative"] = ABBC_HARD_NEGATIVE_LABEL
    return labels


def _abbc_allowed_values(ds_id: int) -> set[int]:
    return set(range(int(DATASETS[ds_id]["n_classes"])))


def _instance_labels_for_dataset(ds_id: int) -> dict[str, int]:
    """Return Dataset537 raw label contract: background + pelvic fragment IDs."""
    cfg = DATASETS[ds_id]
    lo, hi = cfg["global_label_range"] or (1, 150)
    labels = {"background": 0}
    labels.update({f"fragment_{idx:03d}": idx for idx in range(int(lo), int(hi) + 1)})
    return labels


def _pelvic_instance_target(inst: np.ndarray) -> np.ndarray:
    """Keep pelvic PENGWIN fragment IDs and drop femur/other labels."""
    out = np.where((inst >= 1) & (inst <= 150), inst, 0)
    return out.astype(np.uint16, copy=False)


V3_INPUT_VARIANTS = ("ct_hu", "ct_lut", "ct_lut_ifs", "ct_lut_anat_gate", "ct_anatprob")
V4_INPUT_VARIANTS = ("ct_lut", "ct_hu", "ct_lut_ifs")
V5_INPUT_VARIANTS = ("ct_lut", "ct_lut_anat_sdf")
V5_DATASET532_PROB_CHANNEL = {"Sacrum": 1, "LeftHip": 2, "RightHip": 3}


def _boundary_fragment_labels_for_dataset() -> dict[str, int]:
    """Return the fixed Dataset537 BoundaryFragment V3 label map."""
    return dict(BFV3_LABELS)


def _bone_lut_normalize(arr: np.ndarray) -> np.ndarray:
    """Deterministic bone-window LUT for the CT-LUT ablation."""
    x = arr.astype(np.float32, copy=False)
    xp = np.asarray([-1000.0, -200.0, 150.0, 700.0, 1500.0, 2000.0], dtype=np.float32)
    fp = np.asarray([0.0, 0.05, 0.35, 0.70, 0.92, 1.0], dtype=np.float32)
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp).astype(np.float32)


def _bone_lut_ifs_triplet(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IFS-style deterministic uncertainty triplet for the CT-LUT ablation.

    This follows the useful part of the IFS_U-Net idea for our setting without
    changing the model family: represent the same CT window as membership,
    nonmembership, and hesitation. The hesitation channel is high around
    ambiguous mid-membership voxels, which is where cortical/fracture boundary
    uncertainty tends to live.
    """

    membership = _bone_lut_normalize(arr)
    nonmembership = (1.0 - membership).astype(np.float32, copy=False)
    hesitation = (1.0 - np.abs(2.0 * membership - 1.0)).astype(np.float32, copy=False)
    return membership, nonmembership, hesitation


def _select_case_subset(cases: list[Path], case_subset: list[str] | None) -> list[Path]:
    """Filter source case directories by zero-padded case IDs."""
    if not case_subset:
        return cases
    wanted = {str(c).zfill(3) for c in case_subset}
    selected = [cd for cd in cases if cd.name.zfill(3) in wanted]
    missing = sorted(wanted - {cd.name.zfill(3) for cd in selected})
    if missing:
        raise FileNotFoundError(f"case subset missing source cases: {missing}")
    return selected


# =============================================================================
# Bone-CT preprocessing helpers
# =============================================================================
def bone_window_clip(img_arr: np.ndarray,
                     hu_low: float = -1000.0,
                     hu_high: float = 2000.0) -> np.ndarray:
    """Clip CT HU to a bone-relevant range BEFORE any further normalization.

    v8 best practice (PENGWIN'26 dataset 실측 기반):
        2026 audit: Pelvic bone HU p1=-69, p99=1313, max=1963 (rare metal artifact
        gets up to 19242). Femur bone p1=-71, p99=1464, max=1597.
        → Clipping at [-1000, 2000] preserves 99.99% of bone signal while
        truncating extreme metal artifacts (causes z-score to inflate).
        nnU-Net's default CT-percentile normalization handles z-score AFTER this.

    Why we clip BEFORE nnU-Net normalization:
        Default nnU-Net uses percentile-based clip (0.5/99.5%) computed PER-CASE,
        which works but inflates the z-score variance when a few metal artifacts
        push the 99.5% to >5000. Pre-clipping at -1000/2000 produces tighter,
        more uniform z-scores across cases.

    Args:
        img_arr: input CT volume (z, y, x), HU values (int or float).
        hu_low, hu_high: clip range. Defaults [-1000, 2000].

    Returns:
        np.ndarray, same shape, same dtype. In-range values unchanged; outliers clipped.
    """
    return clip_ct_hu(img_arr, (hu_low, hu_high))


def canonicalize_and_clip_image(img: sitk.Image) -> tuple[sitk.Image, np.ndarray]:
    """Return an LPS-oriented CT image and its bone-window-clipped array.

    Audit note:
        The previous raw nnU-Net datasets were generated while source cases
        still had mixed LPS/RAS directions. That silently poisons left/right
        anatomy learning and makes old checkpoints unsuitable as final
        evidence. Every builder must pass CT through this helper before writing
        `imagesTr`, so rebuild audits can assert all images are LPS.
    """
    img_lps = canonicalize_sitk(img)
    arr = bone_window_clip(sitk.GetArrayFromImage(img_lps))
    return img_lps, arr


def assert_lps_image(img: sitk.Image, context: str) -> None:
    """Fail fast if a generated image is not in the canonical LPS orientation."""
    code = orientation_code(img)
    if code != "LPS":
        raise RuntimeError(f"{context}: expected LPS orientation, got {code}")


def assert_matching_geometry(img: sitk.Image, lbl: sitk.Image, context: str) -> None:
    """Fail fast if CT/label geometry diverges after canonicalization."""
    if img.GetSize() != lbl.GetSize():
        raise RuntimeError(f"{context}: image/label size mismatch {img.GetSize()} != {lbl.GetSize()}")
    if not np.allclose(img.GetSpacing(), lbl.GetSpacing(), rtol=0, atol=1e-5):
        raise RuntimeError(f"{context}: image/label spacing mismatch {img.GetSpacing()} != {lbl.GetSpacing()}")
    if not np.allclose(img.GetOrigin(), lbl.GetOrigin(), rtol=0, atol=1e-4):
        raise RuntimeError(f"{context}: image/label origin mismatch {img.GetOrigin()} != {lbl.GetOrigin()}")
    if not np.allclose(img.GetDirection(), lbl.GetDirection(), rtol=0, atol=1e-5):
        raise RuntimeError(f"{context}: image/label direction mismatch")


def _spacing_zyx(spacing_xyz: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    """Return spacing in numpy array order.

    SimpleITK exposes spacing as x/y/z, while `GetArrayFromImage` returns
    z/y/x. ABBC core generation uses Euclidean distance in millimetres; mixing
    the order would silently bias seed placement in anisotropic scans.
    """
    sx, sy, sz = [float(v) for v in spacing_xyz]
    return (sz, sy, sx)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep one connected component from a binary mask using 26-connectivity."""
    from scipy import ndimage as ndi

    cc, n_cc = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_cc <= 1:
        return mask.astype(bool, copy=False)
    sizes = ndi.sum(mask, cc, index=np.arange(1, n_cc + 1))
    keep = int(np.argmax(sizes)) + 1
    return cc == keep


def _connected_component_count(mask: np.ndarray) -> int:
    """Return 26-connected component count for a binary mask."""
    from scipy import ndimage as ndi

    _, n_cc = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return int(n_cc)


def _connected_component_count_bbox(mask: np.ndarray) -> int:
    """Count components after cropping to foreground bbox.

    ABBC target audits call this once per GT fragment. Running connected
    components over the full CT volume for every tiny fragment is needlessly
    expensive; cropping preserves the exact component count while turning a
    minutes-long audit into a practical QA gate.
    """
    bbox = _mask_bbox(mask, pad=1)
    if bbox is None:
        return 0
    return _connected_component_count(mask[bbox])


def _mask_bbox(mask: np.ndarray, pad: int = 2) -> tuple[slice, slice, slice] | None:
    """Return a small padded bbox around a binary mask in z/y/x order."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    mins = np.maximum(coords.min(axis=0) - int(pad), 0)
    maxs = np.minimum(coords.max(axis=0) + int(pad) + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))


def to_abbc_official_pelvis(label: np.ndarray) -> np.ndarray:
    """Convert original PENGWIN instance GT into the official ABBC target.

    Official ABBC class order is fixed as `1=boundary`, `2=core`,
    `3=border/fracture`. This replaces the retired per-anatomy target where
    class 2 meant border and class 3 meant core.
    """
    return compute_abbc_official_target(
        label,
        divergence_threshold=ABBC_DIVERGENCE_THRESHOLD,
        distance_threshold=ABBC_DISTANCE_THRESHOLD_VOX,
        fracture_disk_radius=ABBC_FRACTURE_DISK_RADIUS,
        enforce_one_core_cc=True,
    )


def _load_probability_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    for key in ("probabilities", "softmax"):
        if key in data:
            arr = data[key]
            break
    else:
        keys = list(data.keys())
        if not keys:
            raise RuntimeError(f"empty probability npz: {path}")
        arr = data[keys[0]]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise RuntimeError(f"expected probability array (C,Z,Y,X) in {path}, got {arr.shape}")
    return arr


def _pelvic_sdf_from_probs(probs: np.ndarray,
                           spacing_zyx: tuple[float, float, float],
                           clip_mm: float = 40.0) -> np.ndarray:
    """Signed distance to Dataset532 pelvic probability foreground.

    Positive values are inside the predicted pelvis. The channel is clipped and
    scaled to [-1, 1] so nnU-Net can consume it as `nonorm` context.
    """
    pelvic_prob = np.clip(probs[1:4].sum(axis=0), 0.0, 1.0)
    pelvic = pelvic_prob >= 0.5
    if not pelvic.any():
        return np.zeros_like(pelvic_prob, dtype=np.float32)
    inside = ndi_distance_transform(pelvic, spacing_zyx)
    outside = ndi_distance_transform(~pelvic, spacing_zyx)
    sdf = np.clip(inside - outside, -float(clip_mm), float(clip_mm)) / float(clip_mm)
    return sdf.astype(np.float32, copy=False)


def _selected_anatomy_sdf_from_prob(prob: np.ndarray,
                                    spacing_zyx: tuple[float, float, float],
                                    clip_mm: float = 40.0) -> np.ndarray:
    """Signed distance for one Dataset532 anatomy probability channel.

    [AUDIT][Risk:Major][Scope:v7_input_context]
    This channel is deterministic inference context, not a target-derived
    feature. It may come from a trained Dataset532 checkpoint, so the build
    report records the checkpoint and input profile. For production validation,
    this must be regenerated out-of-fold; six-case root-cause overfits use it
    only to test whether anatomy support context fixes CT-only support leakage.

    [QC][Invariant:range]
    Values are clipped/scaled to [-1, 1] and saved as nnU-Net `nonorm`
    channels. Keeping the range bounded prevents the SDF from dominating the
    CT-LUT channel solely by numeric scale.
    """
    arr = np.asarray(prob, dtype=np.float32)
    mask = arr >= 0.5
    if not mask.any():
        return np.zeros_like(arr, dtype=np.float32)
    inside = ndi_distance_transform(mask, spacing_zyx)
    outside = ndi_distance_transform(~mask, spacing_zyx)
    sdf = np.clip(inside - outside, -float(clip_mm), float(clip_mm)) / float(clip_mm)
    return sdf.astype(np.float32, copy=False)


def _bone_like_hard_negative_prior(img_arr: np.ndarray,
                                   probs: np.ndarray,
                                   hu_threshold: float = 250.0,
                                   low_pelvis_prob: float = 0.35) -> np.ndarray:
    """Inference-computable prior for bone-like voxels likely outside pelvis.

    This is an input channel, so it must not depend on GT. It marks high-HU
    bone/metal-like voxels where Dataset532 assigns little pelvic probability.
    The class-4 target is stricter and GT-derived, but this channel is available
    at both training and inference time.
    """
    pelvic_prob = np.clip(probs[1:4].sum(axis=0), 0.0, 1.0)
    prior = (img_arr.astype(np.float32, copy=False) > float(hu_threshold)) & (pelvic_prob < float(low_pelvis_prob))
    return prior.astype(np.float32, copy=False)


def _contact_hard_negative_target(img_arr: np.ndarray,
                                  inst: np.ndarray,
                                  abbc: np.ndarray,
                                  probs: np.ndarray) -> np.ndarray:
    """Build Dataset537 class-4 supervision without touching true contacts/core.

    Class 4 is a contact false-positive suppressor. It is not an instance body:
    decoder profiles exclude it from support and treat its probability as a
    crossing/forbidden-region penalty. We therefore label only hard negatives
    that are far enough from GT contact and never overwrite core voxels.
    """
    from scipy import ndimage as ndi

    pelvis_gt = (inst >= 1) & (inst <= 150)
    contact = abbc == ABBC_BORDER_LABEL
    core = abbc == ABBC_CORE_LABEL
    contact_guard = ndi.binary_dilation(contact, structure=np.ones((3, 3, 3), dtype=bool), iterations=2)
    pelvic_prob = np.clip(probs[1:4].sum(axis=0), 0.0, 1.0)
    near_gt_pelvis = ndi.binary_dilation(pelvis_gt, structure=np.ones((3, 3, 3), dtype=bool), iterations=8)

    bone_like = img_arr.astype(np.float32, copy=False) > 250.0
    outside_bone = bone_like & ~pelvis_gt & (near_gt_pelvis | (pelvic_prob >= 0.15))
    foundation_leakage = (pelvic_prob >= 0.50) & ~pelvis_gt & bone_like

    eroded = ndi.binary_erosion(pelvis_gt, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    external_cortical_surface = pelvis_gt & ~eroded & bone_like
    external_cortical_surface &= ~contact_guard & ~core

    hard_negative = (outside_bone | foundation_leakage | external_cortical_surface)
    hard_negative &= ~contact_guard & ~core
    return hard_negative


def ndi_distance_transform(mask: np.ndarray,
                           spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.distance_transform_edt(mask, sampling=spacing_zyx).astype(np.float32, copy=False)


def _anatomy_context_root() -> Path:
    return Path(os.environ.get(
        "PENGWIN_ABBC_ANATOMY_CONTEXT_ROOT",
        str(RESULT_VISUALIZE / "anatomy_context_ds532_checkpoint_best"),
    ))


def _ensure_ds532_anatomy_probability_context(cases: list[Path],
                                              force: bool = False,
                                              gpu: int = 0,
                                              checkpoint: str = "checkpoint_best.pth") -> Path:
    """Generate Dataset532 soft anatomy context for ABBC input channels."""
    out_root = _anatomy_context_root()
    missing = []
    for cd in cases:
        cid = cd.name.zfill(3)
        if force or not (out_root / cid / "image.npz").exists():
            missing.append(cd)
    if not missing:
        print(f"  anatomy context cache ready: {out_root}")
        return out_root

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch

    model_root = (
        NN_RES / DATASETS[532]["name"]
        / f"{DATASETS[532]['trainer']}__nnUNetResEncUNetLPlans__3d_fullres"
    )
    if not (model_root / "fold_0" / checkpoint).exists():
        raise FileNotFoundError(f"Dataset532 checkpoint missing: {model_root / 'fold_0' / checkpoint}")
    device = torch.device("cuda", int(gpu)) if torch.cuda.is_available() else torch.device("cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_root), use_folds=(0,), checkpoint_name=checkpoint,
    )
    print(f"  generating Dataset532 probability context for {len(missing)} cases -> {out_root}")
    source_lists = []
    output_truncated = []
    meta_rows = []
    ds532_raw = NN_RAW / DATASETS[532]["name"] / "imagesTr"
    for cd in missing:
        cid = cd.name.zfill(3)
        out_dir = out_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_input = ds532_raw / f"PENGWIN_{cid}_0000.mha"
        if not raw_input.exists():
            # [DATA][Scope:anatomy_context][Risk:Major]
            # Aggressive cleanup may remove nnU-Net raw Dataset532 while keeping
            # the trained checkpoint. Rebuilding the full anatomy raw dataset
            # just to run a six-case V7 diagnostic wastes disk. Materialize only
            # the canonicalized CT required by this context cache; labels are not
            # needed for inference and no Dataset537 target is derived from this
            # temporary input.
            img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
            raw_input.parent.mkdir(parents=True, exist_ok=True)
            save_full_mha(arr_img, img, raw_input, dtype=arr_img.dtype)
        source_lists.append([str(raw_input)])
        output_truncated.append(str(out_dir / "image"))
        meta_rows.append((cid, cd, raw_input, out_dir))
    predictor.predict_from_files(
        source_lists,
        output_truncated,
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=max(1, min(8, int(os.environ.get("PENGWIN_ABBC_CONTEXT_PREPROC", "4")))),
        num_processes_segmentation_export=max(1, min(4, int(os.environ.get("PENGWIN_ABBC_CONTEXT_EXPORT", "2")))),
    )
    for cid, cd, raw_input, out_dir in meta_rows:
        (out_dir / "context_meta.json").write_text(json.dumps({
            "case": cid,
            "source_image": str(cd / "image.mha"),
            "prepared_input": str(raw_input),
            "dataset": 532,
            "checkpoint": checkpoint,
            "model_root": str(model_root),
            "channels": ["Sacrum", "LeftHip", "RightHip"],
        }, indent=2))
    return out_root


def _build_abbc_official_case_worker(args: tuple[int, str, bool, str]) -> dict:
    """Build one official ABBC raw case.

    ABBC target creation is the slow part of the official rebuild. Keeping the
    worker at module scope lets the dataset builder parallelize by case without
    changing the nnU-Net raw dataset contract or output paths.
    """
    ds_id, case_dir_str, force, context_root_str = args
    cfg = DATASETS[ds_id]
    cd = Path(case_dir_str)
    dst = NN_RAW / cfg["name"]
    cid = int(cd.name)
    out_img = dst / "imagesTr" / f"PENGWIN_{cid:03d}_0000.mha"
    out_lbl = dst / "labelsTr" / f"PENGWIN_{cid:03d}.mha"
    n_context_channels = 5 if int(cfg["n_classes"]) >= 5 else 4
    out_context = [dst / "imagesTr" / f"PENGWIN_{cid:03d}_{ch:04d}.mha" for ch in range(1, 1 + n_context_channels)]
    if out_img.exists() and out_lbl.exists() and all(p.exists() for p in out_context) and not force:
        return {"case": f"{cid:03d}", "skipped": True}

    img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
    lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
    assert_lps_image(img, f"Ds{ds_id} case {cid:03d} CT")
    assert_lps_image(lbl_img, f"Ds{ds_id} case {cid:03d} label")
    assert_matching_geometry(img, lbl_img, f"Ds{ds_id} case {cid:03d}")

    inst = sitk.GetArrayFromImage(lbl_img)
    abbc = to_abbc_official_pelvis(inst)

    context_root = Path(context_root_str)
    prob_path = context_root / f"{cid:03d}" / "image.npz"
    probs = _load_probability_npz(prob_path)
    if probs.shape[0] < 4:
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: expected Dataset532 4-class probs, got {probs.shape}")
    if probs.shape[1:] != arr_img.shape:
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: anatomy prob shape {probs.shape[1:]} != CT {arr_img.shape}")
    spacing_zyx = _spacing_zyx(img.GetSpacing())
    sdf = _pelvic_sdf_from_probs(probs, spacing_zyx=spacing_zyx)
    hard_negative_prior = _bone_like_hard_negative_prior(arr_img, probs)
    if int(cfg["n_classes"]) >= 5:
        hard_negative_target = _contact_hard_negative_target(arr_img, inst, abbc, probs)
        abbc = abbc.copy()
        abbc[hard_negative_target] = ABBC_HARD_NEGATIVE_LABEL

    values = set(int(v) for v in np.unique(abbc))
    allowed = _abbc_allowed_values(ds_id)
    if not values.issubset(allowed):
        raise RuntimeError(
            f"Ds{ds_id} case {cid:03d}: ABBC labels {sorted(values)} outside {sorted(allowed)}"
        )
    if not (abbc == ABBC_CORE_LABEL).any():
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: no ABBC core voxels")

    save_full_mha(arr_img, img, out_img, dtype=arr_img.dtype)
    save_full_mha(probs[1], img, out_context[0], dtype=np.float32)
    save_full_mha(probs[2], img, out_context[1], dtype=np.float32)
    save_full_mha(probs[3], img, out_context[2], dtype=np.float32)
    save_full_mha(sdf, img, out_context[3], dtype=np.float32)
    if int(cfg["n_classes"]) >= 5:
        save_full_mha(hard_negative_prior, img, out_context[4], dtype=np.float32)
    save_full_mha(abbc, lbl_img, out_lbl, dtype=np.uint8)
    return {
        "case": f"{cid:03d}",
        "skipped": False,
        "core_voxels": int((abbc == ABBC_CORE_LABEL).sum()),
        "contact_voxels": int((abbc == ABBC_BORDER_LABEL).sum()),
        "hard_negative_voxels": int((abbc == ABBC_HARD_NEGATIVE_LABEL).sum()),
        "anatomy_context": str(prob_path),
    }


def _build_contact_instance_case_worker(args: tuple[int, str, bool, str]) -> dict:
    """Build one Dataset537 raw case with instance-ID labels.

    Raw labels are intentionally not ABBC classes. nnU-Net preprocessing will
    resample this instance map into `_seg.npy`; the custom trainer derives
    support/core/contact/offset/embedding targets from that resampled instance
    map so training and validation use the same geometry.
    """
    ds_id, case_dir_str, force, context_root_str = args
    cfg = DATASETS[ds_id]
    cd = Path(case_dir_str)
    dst = NN_RAW / cfg["name"]
    cid = int(cd.name)
    out_img = dst / "imagesTr" / f"PENGWIN_{cid:03d}_0000.mha"
    out_lbl = dst / "labelsTr" / f"PENGWIN_{cid:03d}.mha"
    out_context = [dst / "imagesTr" / f"PENGWIN_{cid:03d}_{ch:04d}.mha" for ch in range(1, 6)]
    if out_img.exists() and out_lbl.exists() and all(p.exists() for p in out_context) and not force:
        return {"case": f"{cid:03d}", "skipped": True}

    img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
    lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
    assert_lps_image(img, f"Ds{ds_id} case {cid:03d} CT")
    assert_lps_image(lbl_img, f"Ds{ds_id} case {cid:03d} label")
    assert_matching_geometry(img, lbl_img, f"Ds{ds_id} case {cid:03d}")

    inst = _pelvic_instance_target(sitk.GetArrayFromImage(lbl_img))
    if not (inst > 0).any():
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: empty pelvic instance target")

    context_root = Path(context_root_str)
    prob_path = context_root / f"{cid:03d}" / "image.npz"
    probs = _load_probability_npz(prob_path)
    if probs.shape[0] < 4:
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: expected Dataset532 4-class probs, got {probs.shape}")
    if probs.shape[1:] != arr_img.shape:
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: anatomy prob shape {probs.shape[1:]} != CT {arr_img.shape}")
    spacing_zyx = _spacing_zyx(img.GetSpacing())
    sdf = _pelvic_sdf_from_probs(probs, spacing_zyx=spacing_zyx)
    hard_negative_prior = _bone_like_hard_negative_prior(arr_img, probs)

    values = set(int(v) for v in np.unique(inst))
    if any(v < 0 or v > 150 for v in values):
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: instance IDs outside 0..150: {sorted(values)}")

    save_full_mha(arr_img, img, out_img, dtype=arr_img.dtype)
    save_full_mha(probs[1], img, out_context[0], dtype=np.float32)
    save_full_mha(probs[2], img, out_context[1], dtype=np.float32)
    save_full_mha(probs[3], img, out_context[2], dtype=np.float32)
    save_full_mha(sdf, img, out_context[3], dtype=np.float32)
    save_full_mha(hard_negative_prior, img, out_context[4], dtype=np.float32)
    save_full_mha(inst, lbl_img, out_lbl, dtype=np.uint16)
    fragment_ids = [int(v) for v in np.unique(inst) if int(v) > 0]
    return {
        "case": f"{cid:03d}",
        "skipped": False,
        "fragment_count": len(fragment_ids),
        "fragment_ids": fragment_ids,
        "target_voxels": int((inst > 0).sum()),
        "anatomy_context": str(prob_path),
    }


def compute_body_mask(img_arr: np.ndarray,
                      hu_threshold: float = -500.0,
                      morph_close_iter: int = 3,
                      min_volume: int = 100_000) -> np.ndarray:
    """Pre-compute body silhouette mask (boolean) for FP suppression.

    v8 best practice:
        Sano 5위 PeFreCT (PENGWIN'24) reported 19.23 mean unmatched fragments
        per case from CT-table / hand-rest contamination. Pre-computing body
        silhouette and storing once per case ensures consistent FP filtering.

    Algorithm:
        1. Threshold > hu_threshold (-500): excludes air, table, gantry.
        2. Morphological closing (iter=3) to fill hand/clothing gaps.
        3. Largest 3D connected component = body silhouette.
        4. Sanity: if largest CC < min_volume, return all-True (degenerate input).

    Args:
        img_arr: CT volume (z, y, x), HU values.
        hu_threshold: HU cutoff. -500 captures both bone+soft-tissue but rejects
        CT table (typically -800 HU) and air pockets.
        morph_close_iter: morphological closing iterations to seal gaps.
        min_volume: largest CC sanity threshold; below this, return None.

    Returns:
        boolean numpy array same shape as img_arr (True = inside body).
    """
    from scipy import ndimage as ndi
    mask = img_arr > hu_threshold
    if not mask.any():
        return np.ones_like(img_arr, dtype=bool)
    # Closing to fill holes inside body silhouette
    if morph_close_iter > 0:
        mask = ndi.binary_closing(mask, iterations=morph_close_iter)
    cc, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=int))
    if n == 0:
        return np.ones_like(img_arr, dtype=bool)
    sizes = ndi.sum(mask, cc, index=np.arange(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    if sizes[largest - 1] < min_volume:
        # degenerate input; defer filtering
        return np.ones_like(img_arr, dtype=bool)
    return (cc == largest)


def build_anatomy_semantic_dataset(ds_id: int, force: bool = False) -> int:
    """Build a trusted CT → semantic anatomy nnU-Net raw dataset.

    Contract:
        - Use only original GT labels from `/workspace/data/task1_2/extracted`.
        - Collapse fragment IDs into the dataset's local anatomy IDs without
          inventing supervision.
        - Keep background as one class; air/soft tissue/table/other-bone labels
          are not verified in Task1/2.
        - Canonicalize both CT and label to LPS so LH/RH semantics are stable.
    """
    cfg = DATASETS[ds_id]
    dst = NN_RAW / cfg["name"]
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = list_cases(case_filter=None if cfg["filter"] == "all" else cfg["filter"])
    local_label_map = {
        anat: local_idx for local_idx, anat in enumerate(cfg["anatomies"], start=1)
    }
    global_label_map = {
        anat: ANATOMY_NAMES.index(anat) + 1 for anat in cfg["anatomies"]
    }
    print(
        f"[anatomy] Ds{ds_id} {cfg['name']} — {len(cases)} trusted labeled CT cases "
        f"anatomies={cfg['anatomies']}"
    )

    for i, cd in enumerate(cases):
        cid = int(cd.name)
        out_img = dst / "imagesTr" / f"PENGWIN_{cid:03d}_0000.mha"
        out_lbl = dst / "labelsTr" / f"PENGWIN_{cid:03d}.mha"
        if out_img.exists() and out_lbl.exists() and not force:
            continue

        img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        assert_lps_image(img, f"Ds{ds_id} case {cid:03d} CT")
        assert_lps_image(lbl_img, f"Ds{ds_id} case {cid:03d} label")
        assert_matching_geometry(img, lbl_img, f"Ds{ds_id} case {cid:03d}")

        inst = sitk.GetArrayFromImage(lbl_img)
        global_anat = inst_to_anat(inst)
        anat = np.zeros_like(global_anat, dtype=np.uint8)
        for name in cfg["anatomies"]:
            anat[global_anat == global_label_map[name]] = local_label_map[name]
        values = set(int(v) for v in np.unique(anat))
        allowed = set(range(int(cfg["n_classes"])))
        if not values.issubset(allowed):
            raise RuntimeError(
                f"Ds{ds_id} case {cid:03d}: labels {sorted(values)} outside {sorted(allowed)}"
            )
        if not (anat > 0).any():
            raise RuntimeError(f"Ds{ds_id} case {cid:03d}: empty foreground after anatomy remap")

        save_full_mha(arr_img, img, out_img, dtype=arr_img.dtype)
        save_full_mha(anat, lbl_img, out_lbl, dtype=np.uint8)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(cases)}]")

    labels = {"background": 0}
    labels.update({name: idx for name, idx in local_label_map.items()})
    ds_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": len(cases),
        "file_ending": ".mha",
        "description": (
            f"PENGWIN 2026 — {cfg['name']} split semantic anatomy target. "
            f"Original fragment IDs are collapsed to background/{'/'.join(cfg['anatomies'])}; "
            "no TotalSegmentator, Task3, pseudo-label, or unlabeled source is included."
        ),
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    print(f"  ✅ {cfg['name']} ready")
    return len(cases)


def build_abbc_official_dataset(ds_id: int,
                                force: bool = False) -> int:
    """Build the official-aligned pelvic ABBC dataset.

    Contract:
        - input channels are CT + Dataset532 soft anatomy + pelvis signed
          distance context, plus a hard-negative prior for Dataset537;
        - labels are ABBC Contact classes 0/1/2/3 and optional class 4;
        - class order is background/boundary-shell/core/contact-surface/(hard-negative);
        - all pelvic fragments enter one instance representation.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "abbc_official":
        raise ValueError(f"Dataset{ds_id} is not an official ABBC dataset")
    dst = NN_RAW / cfg["name"]
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = list_cases(case_filter=cfg["filter"])
    print(
        f"[abbc-official] Ds{ds_id} {cfg['name']} — {len(cases)} trusted pelvic CT cases "
        f"channels=CT+P(Sa/LH/RH)+pelvisSDF"
        f"{'+hardNegativePrior' if int(cfg['n_classes']) >= 5 else ''} "
        f"labels=background/boundary/core/contact"
        f"{'/hard-negative' if int(cfg['n_classes']) >= 5 else ''} "
        f"div={ABBC_DIVERGENCE_THRESHOLD} dist={ABBC_DISTANCE_THRESHOLD_VOX} "
        f"fracture_radius={ABBC_FRACTURE_DISK_RADIUS}"
    )
    context_root = _ensure_ds532_anatomy_probability_context(
        cases,
        force=bool(int(os.environ.get("PENGWIN_ABBC_FORCE_CONTEXT", "0"))),
        gpu=int(os.environ.get("PENGWIN_ABBC_CONTEXT_GPU", "0")),
        checkpoint=os.environ.get("PENGWIN_ABBC_CONTEXT_CHECKPOINT", "checkpoint_best.pth"),
    )

    workers = max(1, int(os.environ.get("PENGWIN_ABBC_BUILD_WORKERS", "4")))
    workers = min(workers, len(cases))
    print(f"  workers={workers} (override with PENGWIN_ABBC_BUILD_WORKERS)")
    if workers == 1:
        for i, cd in enumerate(cases, start=1):
            _build_abbc_official_case_worker((ds_id, str(cd), force, str(context_root)))
            if i % 10 == 0 or i == len(cases):
                print(f"  [{i}/{len(cases)}]")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_build_abbc_official_case_worker, (ds_id, str(cd), force, str(context_root)))
                for cd in cases
            ]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 10 == 0 or done == len(cases):
                    print(f"  [{done}/{len(cases)}]")

    channel_names = {
        "0": "CT",
        "1": "nonorm",
        "2": "nonorm",
        "3": "nonorm",
        "4": "nonorm",
    }
    input_contract = [
        "CT",
        "Dataset532_Sacrum_probability",
        "Dataset532_LeftHip_probability",
        "Dataset532_RightHip_probability",
        "pelvis_signed_distance",
    ]
    label_order = ["background", "boundary_shell", "core", "contact_surface"]
    if int(cfg["n_classes"]) >= 5:
        channel_names["5"] = "nonorm"
        input_contract.append("bone_like_hard_negative_prior")
        label_order.append("contact_hard_negative")

    ds_json = {
        "channel_names": channel_names,
        "labels": _abbc_labels_for_dataset(ds_id),
        "numTraining": len(cases),
        "file_ending": ".mha",
        "abbc_official": {
            "label_order": label_order,
            "input_contract": input_contract,
            "anatomy_context_checkpoint": os.environ.get("PENGWIN_ABBC_CONTEXT_CHECKPOINT", "checkpoint_best.pth"),
            "anatomy_context_root": str(context_root),
            "divergence_threshold": float(ABBC_DIVERGENCE_THRESHOLD),
            "distance_threshold_vox": float(ABBC_DISTANCE_THRESHOLD_VOX),
            "contact_radius_vox": max(1, int(ABBC_FRACTURE_DISK_RADIUS) // 2),
            "core_clearance_radius_delta": 2,
            "enforce_one_core_cc": True,
            "hard_negative_class": ABBC_HARD_NEGATIVE_LABEL if int(cfg["n_classes"]) >= 5 else None,
        },
        "description": (
            f"PENGWIN 2026 — {cfg['name']} Contact-Energy V2 ABBC target. "
            "Inputs are CT plus Dataset532 soft anatomy context, pelvis SDF, "
            "and an optional hard-negative prior. Local classes are "
            "background/boundary-shell/core/contact-surface"
            f"{'/contact-hard-negative' if int(cfg['n_classes']) >= 5 else ''}."
        ),
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    print(f"  ✅ {cfg['name']} ready")
    return len(cases)


def _build_boundary_fragment_v3_case_worker(args: tuple[int, str, bool, str, str | None]) -> dict:
    """Build one Dataset537 BoundaryFragment V3 raw case."""
    ds_id, case_dir_str, force, v3_input, context_root_str = args
    cfg = DATASETS[ds_id]
    cd = Path(case_dir_str)
    dst = NN_RAW / cfg["name"]
    cid = int(cd.name)
    out_lbl = dst / "labelsTr" / f"PENGWIN_{cid:03d}.mha"
    if v3_input == "ct_anatprob":
        n_channels = 4
    elif v3_input == "ct_lut_ifs":
        n_channels = 3
    else:
        n_channels = 1
    out_images = [dst / "imagesTr" / f"PENGWIN_{cid:03d}_{ch:04d}.mha" for ch in range(n_channels)]
    if out_lbl.exists() and all(p.exists() for p in out_images) and not force:
        return {"case": f"{cid:03d}", "skipped": True}

    img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
    lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
    assert_lps_image(img, f"Ds{ds_id} case {cid:03d} CT")
    assert_lps_image(lbl_img, f"Ds{ds_id} case {cid:03d} label")
    assert_matching_geometry(img, lbl_img, f"Ds{ds_id} case {cid:03d}")

    inst = _pelvic_instance_target(sitk.GetArrayFromImage(lbl_img))
    if not (inst > 0).any():
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: empty pelvic instance source")
    spacing_zyx = _spacing_zyx(img.GetSpacing())
    target, target_audit = compute_boundary_fragment_target(
        inst,
        spacing_zyx=spacing_zyx,
        params=BoundaryFragmentParams(),
    )
    invalid = set(int(v) for v in np.unique(target)) - set(BFV3_LABELS.values())
    if invalid:
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: invalid V3 labels {sorted(invalid)}")
    if not (target == BFV3_LABELS["fragment_core"]).any():
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: no class-4 core voxels")

    if v3_input == "ct_hu":
        save_full_mha(arr_img, img, out_images[0], dtype=arr_img.dtype)
    elif v3_input in {"ct_lut", "ct_lut_anat_gate"}:
        save_full_mha(_bone_lut_normalize(arr_img), img, out_images[0], dtype=np.float32)
    elif v3_input == "ct_lut_ifs":
        membership, nonmembership, hesitation = _bone_lut_ifs_triplet(arr_img)
        save_full_mha(membership, img, out_images[0], dtype=np.float32)
        save_full_mha(nonmembership, img, out_images[1], dtype=np.float32)
        save_full_mha(hesitation, img, out_images[2], dtype=np.float32)
    elif v3_input == "ct_anatprob":
        if not context_root_str:
            raise RuntimeError("ct_anatprob requires Dataset532 anatomy probability context")
        context_root = Path(context_root_str)
        prob_path = context_root / f"{cid:03d}" / "image.npz"
        probs = _load_probability_npz(prob_path)
        if probs.shape[0] < 4:
            raise RuntimeError(f"Ds{ds_id} case {cid:03d}: expected 4 Dataset532 prob channels, got {probs.shape}")
        if probs.shape[1:] != arr_img.shape:
            raise RuntimeError(f"Ds{ds_id} case {cid:03d}: anatomy prob shape {probs.shape[1:]} != CT {arr_img.shape}")
        save_full_mha(_bone_lut_normalize(arr_img), img, out_images[0], dtype=np.float32)
        save_full_mha(probs[1], img, out_images[1], dtype=np.float32)
        save_full_mha(probs[2], img, out_images[2], dtype=np.float32)
        save_full_mha(probs[3], img, out_images[3], dtype=np.float32)
    else:
        raise ValueError(f"unknown V3 input variant: {v3_input}")
    save_full_mha(target, lbl_img, out_lbl, dtype=np.uint8)
    return {
        "case": f"{cid:03d}",
        "skipped": False,
        "input_variant": v3_input,
        "target_audit": target_audit,
    }


def build_boundary_fragment_v3_dataset(ds_id: int,
                                       force: bool = False,
                                       v3_input: str = "ct_hu",
                                       case_subset: list[str] | None = None) -> int:
    """Build Dataset537_PelvicBoundaryFragmentV3 raw data.

    V3 deliberately starts with one CT channel. AnatomyV2 can be used later as a
    final decoder/eval gate (`ct_lut_anat_gate`) or as a diagnostic input
    ablation (`ct_anatprob`), but it is no longer mandatory model context.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "boundary_fragment":
        raise ValueError(f"Dataset{ds_id} is not a BoundaryFragment V3 dataset")
    if v3_input not in V3_INPUT_VARIANTS:
        raise ValueError(f"--v3-input must be one of {V3_INPUT_VARIANTS}, got {v3_input!r}")
    dst = NN_RAW / cfg["name"]
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = _select_case_subset(list_cases(case_filter=cfg["filter"]), case_subset)
    print(
        f"[boundary-fragment-v3] Ds{ds_id} {cfg['name']} — {len(cases)} pelvic cases "
        f"input={v3_input} labels=0:bg,1:external,2:thin_ridge,3:shell,4:core"
    )

    context_root = None
    if v3_input == "ct_anatprob":
        context_root = _ensure_ds532_anatomy_probability_context(
            cases,
            force=bool(int(os.environ.get("PENGWIN_V3_FORCE_CONTEXT", "0"))),
            gpu=int(os.environ.get("PENGWIN_V3_CONTEXT_GPU", "0")),
            checkpoint=os.environ.get("PENGWIN_V3_CONTEXT_CHECKPOINT", "checkpoint_best.pth"),
        )

    workers = max(1, int(os.environ.get("PENGWIN_V3_BUILD_WORKERS", "4")))
    workers = min(workers, max(1, len(cases)))
    items = [(ds_id, str(cd), force, v3_input, str(context_root) if context_root else None) for cd in cases]
    rows = []
    if workers == 1:
        for i, item in enumerate(items, start=1):
            rows.append(_build_boundary_fragment_v3_case_worker(item))
            if i % 10 == 0 or i == len(items):
                print(f"  [{i}/{len(items)}]")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_build_boundary_fragment_v3_case_worker, item) for item in items]
            for future in as_completed(futures):
                rows.append(future.result())
                done += 1
                if done % 10 == 0 or done == len(items):
                    print(f"  [{done}/{len(items)}]")
    rows.sort(key=lambda row: row["case"])

    if v3_input == "ct_anatprob":
        channel_names = {"0": "CT", "1": "nonorm", "2": "nonorm", "3": "nonorm"}
        input_contract = ["ct_lut", "Dataset532_Sacrum_probability", "Dataset532_LeftHip_probability", "Dataset532_RightHip_probability"]
    elif v3_input == "ct_lut_ifs":
        channel_names = {"0": "nonorm", "1": "nonorm", "2": "nonorm"}
        input_contract = ["membership_bone_lut", "nonmembership_1_minus_membership", "hesitation_1_minus_abs_2m_minus_1"]
    else:
        channel_names = {"0": "CT"}
        input_contract = [v3_input]
    ds_json = {
        "channel_names": channel_names,
        "labels": _boundary_fragment_labels_for_dataset(),
        "numTraining": len(cases),
        "file_ending": ".mha",
        "boundary_fragment_v3": {
            "input_variant": v3_input,
            "input_contract": input_contract,
            "ifs_input_contract": (
                {
                    "membership": "bone-window LUT normalized to [0, 1]",
                    "nonmembership": "1 - membership",
                    "hesitation": "1 - abs(2 * membership - 1)",
                    "normalization": "all ct_lut_ifs channels use nonorm",
                }
                if v3_input == "ct_lut_ifs" else None
            ),
            "label_order": BFV3_CLASS_NAMES,
            "support_labels": [2, 3, 4],
            "split_barrier_label": 2,
            "seed_label": 4,
            "class_1": "external cortical/context hard-negative band, outside fragment support",
            "class_2": "any pelvic different-fragment thin contact ridge; default search 3 mm, ridge 1 mm",
            "class_3": "fragment peripheral shell, 0-8 mm from fragment boundary, support-preserving around class 2",
            "class_4": "fragment deep core >8 mm, with one connected central fallback marker",
            "distance_units": "physical mm",
            "target_profile": "v3_thin_ridge",
            "contact_search_mm": 3.0,
            "contact_ridge_mm": 1.0,
            "contact_same_anatomy_only": False,
            "legacy_contact_band_mm": 3.0,
            "external_band_mm": 3.0,
            "shell_mm": 8.0,
            "case_subset": [cd.name.zfill(3) for cd in cases] if case_subset else None,
            "anatomy_gate_policy": (
                "decoder/eval final safety gate only" if v3_input == "ct_lut_anat_gate" else "not used"
            ),
        },
        "description": (
            f"PENGWIN 2026 — {cfg['name']} BoundaryFragment V3. "
            "One semantic specialist predicts external context, thin fracture ridge, "
            "fragment shell, and fragment core. Instance IDs are recovered by a fixed decoder."
        ),
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    audit_path = RESULT_REPORT / f"audit_boundary_fragment_targets_{v3_input}_{RESULT_DATE}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "dataset_id": ds_id,
        "name": cfg["name"],
        "input_variant": v3_input,
        "cases": rows,
    }, indent=2))
    print(f"  target audit: {audit_path}")
    print(f"  ✅ {cfg['name']} ready")
    return len(cases)


def build_contact_instance_dataset(ds_id: int,
                                   force: bool = False) -> int:
    """Build Dataset537 raw data for Contact-Instance V1.

    Inputs remain inference-safe 6-channel volumes:
    CT, Dataset532 Sacrum/LeftHip/RightHip probabilities, pelvis SDF, and a
    bone-like prior. Labels are global pelvic instance IDs 0..150, not ABBC
    semantic classes. Dense offset/contact/core maps are generated after
    nnU-Net preprocessing from the resampled instance map to avoid storing
    another >100 GB target volume.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "contact_instance":
        raise ValueError(f"Dataset{ds_id} is not a contact-instance dataset")
    dst = NN_RAW / cfg["name"]
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = list_cases(case_filter=cfg["filter"])
    print(
        f"[contact-instance] Ds{ds_id} {cfg['name']} — {len(cases)} trusted pelvic CT cases "
        "channels=CT+P(Sa/LH/RH)+pelvisSDF+bonePrior labels=instance IDs 0..150"
    )
    context_root = _ensure_ds532_anatomy_probability_context(
        cases,
        force=bool(int(os.environ.get("PENGWIN_ABBC_FORCE_CONTEXT", "0"))),
        gpu=int(os.environ.get("PENGWIN_ABBC_CONTEXT_GPU", "0")),
        checkpoint=os.environ.get("PENGWIN_ABBC_CONTEXT_CHECKPOINT", "checkpoint_best.pth"),
    )

    workers = max(1, int(os.environ.get("PENGWIN_INSTANCE_BUILD_WORKERS", "4")))
    workers = min(workers, len(cases))
    print(f"  workers={workers} (override with PENGWIN_INSTANCE_BUILD_WORKERS)")
    if workers == 1:
        for i, cd in enumerate(cases, start=1):
            _build_contact_instance_case_worker((ds_id, str(cd), force, str(context_root)))
            if i % 10 == 0 or i == len(cases):
                print(f"  [{i}/{len(cases)}]")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_build_contact_instance_case_worker, (ds_id, str(cd), force, str(context_root)))
                for cd in cases
            ]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 10 == 0 or done == len(cases):
                    print(f"  [{done}/{len(cases)}]")

    ds_json = {
        "channel_names": {
            "0": "CT",
            "1": "nonorm",
            "2": "nonorm",
            "3": "nonorm",
            "4": "nonorm",
            "5": "nonorm",
        },
        "labels": _instance_labels_for_dataset(ds_id),
        "numTraining": len(cases),
        "file_ending": ".mha",
        "contact_instance_v1": {
            "input_contract": [
                "CT",
                "Dataset532_Sacrum_probability",
                "Dataset532_LeftHip_probability",
                "Dataset532_RightHip_probability",
                "pelvis_signed_distance",
                "bone_like_hard_negative_prior",
            ],
            "raw_label_contract": "0 background, 1..150 pelvic global fragment IDs",
            "output_channels": CONTACT_INSTANCE_CHANNEL_NAMES,
            "sidecar_policy": (
                "preprocessed instance maps are the source of truth; compact "
                "sidecars store fragment centers and crop-sampling locations, "
                "while dense support/core/contact/offset targets are generated "
                "deterministically inside the native nnU-Net dataloader"
            ),
            "foundation_dataset": int(cfg["foundation_dataset"] or 532),
            "anatomy_context_checkpoint": os.environ.get("PENGWIN_ABBC_CONTEXT_CHECKPOINT", "checkpoint_best.pth"),
            "anatomy_context_root": str(context_root),
        },
        "description": (
            f"PENGWIN 2026 — {cfg['name']} Contact-Instance V1. "
            "Raw labels keep pelvic fragment IDs; the custom trainer predicts "
            "support/core/contact-energy/hard-negative/offset/embedding heads."
        ),
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    print(f"  ✅ {cfg['name']} ready")
    return len(cases)


def _build_factorized_instance_v4_case_worker(args: tuple[int, str, bool, str]) -> dict:
    """Build one Dataset537 V4 raw case with instance-ID labels.

    V4 stores only inference-safe image channels and the original pelvic
    instance ID map. Dense support/contact/core/offset targets are derived from
    nnU-Net's resampled `_seg.npy` after planning so training and full-volume
    evaluation share the same geometry.
    """
    ds_id, case_dir_str, force, v4_input = args
    cfg = DATASETS[ds_id]
    cd = Path(case_dir_str)
    dst = NN_RAW / cfg["name"]
    cid = int(cd.name)
    out_lbl = dst / "labelsTr" / f"PENGWIN_{cid:03d}.mha"
    n_channels = 3 if v4_input == "ct_lut_ifs" else 1
    out_images = [dst / "imagesTr" / f"PENGWIN_{cid:03d}_{ch:04d}.mha" for ch in range(n_channels)]
    if out_lbl.exists() and all(p.exists() for p in out_images) and not force:
        return {"case": f"{cid:03d}", "skipped": True}

    img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
    lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
    assert_lps_image(img, f"Ds{ds_id} case {cid:03d} CT")
    assert_lps_image(lbl_img, f"Ds{ds_id} case {cid:03d} label")
    assert_matching_geometry(img, lbl_img, f"Ds{ds_id} case {cid:03d}")
    inst = _pelvic_instance_target(sitk.GetArrayFromImage(lbl_img))
    if not (inst > 0).any():
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: empty pelvic instance target")
    values = set(int(v) for v in np.unique(inst))
    invalid = [v for v in values if v < 0 or v > 150]
    if invalid:
        raise RuntimeError(f"Ds{ds_id} case {cid:03d}: instance IDs outside 0..150: {invalid}")

    if v4_input == "ct_hu":
        save_full_mha(arr_img, img, out_images[0], dtype=arr_img.dtype)
    elif v4_input == "ct_lut":
        save_full_mha(_bone_lut_normalize(arr_img), img, out_images[0], dtype=np.float32)
    elif v4_input == "ct_lut_ifs":
        membership, nonmembership, hesitation = _bone_lut_ifs_triplet(arr_img)
        save_full_mha(membership, img, out_images[0], dtype=np.float32)
        save_full_mha(nonmembership, img, out_images[1], dtype=np.float32)
        save_full_mha(hesitation, img, out_images[2], dtype=np.float32)
    else:
        raise ValueError(f"unknown V4 input variant: {v4_input}")
    save_full_mha(inst, lbl_img, out_lbl, dtype=np.uint16)
    fragment_ids = [int(v) for v in np.unique(inst) if int(v) > 0]
    return {
        "case": f"{cid:03d}",
        "skipped": False,
        "input_variant": v4_input,
        "fragment_count": len(fragment_ids),
        "target_voxels": int((inst > 0).sum()),
    }


def build_factorized_instance_v4_dataset(ds_id: int,
                                         force: bool = False,
                                         v4_input: str = "ct_lut",
                                         case_subset: list[str] | None = None) -> int:
    """Build Dataset537_PelvicFactorizedInstanceV4 raw data."""
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "factorized_instance":
        raise ValueError(f"Dataset{ds_id} is not a FactorizedInstance V4 dataset")
    if v4_input not in V4_INPUT_VARIANTS:
        raise ValueError(f"--v4-input must be one of {V4_INPUT_VARIANTS}, got {v4_input!r}")
    dst = NN_RAW / cfg["name"]
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = _select_case_subset(list_cases(case_filter=cfg["filter"]), case_subset)
    print(
        f"[factorized-instance-v4] Ds{ds_id} {cfg['name']} — {len(cases)} pelvic cases "
        f"input={v4_input} labels=instance IDs 0..150"
    )
    workers = max(1, int(os.environ.get("PENGWIN_V4_BUILD_WORKERS", "4")))
    workers = min(workers, max(1, len(cases)))
    items = [(ds_id, str(cd), force, v4_input) for cd in cases]
    rows = []
    if workers == 1:
        for i, item in enumerate(items, start=1):
            rows.append(_build_factorized_instance_v4_case_worker(item))
            if i % 10 == 0 or i == len(items):
                print(f"  [{i}/{len(items)}]")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_build_factorized_instance_v4_case_worker, item) for item in items]
            for future in as_completed(futures):
                rows.append(future.result())
                done += 1
                if done % 10 == 0 or done == len(items):
                    print(f"  [{done}/{len(items)}]")
    rows.sort(key=lambda row: row["case"])
    if v4_input == "ct_lut_ifs":
        channel_names = {"0": "nonorm", "1": "nonorm", "2": "nonorm"}
        input_contract = [
            "membership_bone_lut",
            "nonmembership_1_minus_membership",
            "hesitation_1_minus_abs_2m_minus_1",
        ]
    elif v4_input == "ct_lut":
        # [DATA][Risk:Major][Scope:normalization_contract]
        # `ct_lut` is already a deterministic LUT/bone-window remap. Declaring it
        # as CT would ask nnU-Net to apply CTNormalization again and makes this
        # input variant incomparable to its documented contract.
        channel_names = {"0": "nonorm"}
        input_contract = [v4_input]
    else:
        channel_names = {"0": "CT"}
        input_contract = [v4_input]
    ds_json = {
        "channel_names": channel_names,
        "labels": _instance_labels_for_dataset(ds_id),
        "numTraining": len(cases),
        "file_ending": ".mha",
        "factorized_instance_v4": {
            "input_variant": v4_input,
            "input_contract": input_contract,
            "raw_label_contract": "0 background, 1..150 pelvic global fragment IDs",
            "output_channels": FACTOR_INSTANCE_CHANNEL_NAMES,
            "sidecar_dir": FACTOR_INSTANCE_SIDECAR_DIR,
            "support_target": "binary pelvic fragment support from resampled instance IDs",
            "contact_target": "same-anatomy adjacent cross-fragment contact energy ridge",
            "core_target": (
                "runtime crop target controlled by PENGWIN_FACTOR_CORE_TARGET; "
                "factorized_instance_v4 defaults to edt_heatmap, V4.1/V4.2 "
                "seedfit profiles force compact_marker with one connected marker"
            ),
            "hard_negative_target": "outside-support external cortex or bone-like context; never overlaps support/core/contact",
            "offset_target": "offset-to-fragment-center inside support",
            "embedding_target": "instance ID map for optional discriminative pull/push loss",
            "active_decoder": "factorized_geodesic_v1",
            "case_subset": [cd.name.zfill(3) for cd in cases] if case_subset else None,
            "anatomy_v2_policy": "not primary input; optional final safety gate ablation only",
        },
        "description": (
            f"PENGWIN 2026 — {cfg['name']} FactorizedInstance V4. "
            "Raw labels keep pelvic fragment IDs; native nnU-Net trainer emits "
            "support/contact/core/hard-negative/offset/embedding heads."
        ),
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    audit_path = RESULT_REPORT / f"audit_factorized_instance_v4_raw_{v4_input}_{RESULT_DATE}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "dataset_id": ds_id,
        "name": cfg["name"],
        "input_variant": v4_input,
        "cases": rows,
    }, indent=2))
    print(f"  raw audit: {audit_path}")
    print(f"  ✅ {cfg['name']} ready")
    return len(cases)


def build_bicm_v5_dataset(ds_id: int,
                          force: bool = False,
                          v5_input: str = "ct_lut",
                          v5_target_profile: str = "v5_tiny_marker",
                          v5_core_ball_radius_mm: float = 2.5,
                          v5_core_body_mm: float = 3.0,
                          v5_contact_band_mm: float = 2.0,
                          case_subset: list[str] | None = None) -> int:
    """Build Dataset537 per-anatomy BICM V5 raw data.

    [AUDIT][Risk:High][Scope:pipeline_reset]
    V5 stops the V4 factorized/global-pelvis path. Anatomy is used to crop one
    ROI per Sacrum/LeftHip/RightHip sample. The default model input is a single
    CT-LUT channel. The V7 diagnostic input profile may add selected Dataset532
    anatomy probability/SDF channels while leaving the target and trainer fixed.

    [DATA][Leakage]
    Labels come only from the original Task1 GT instance map. Dataset532
    predictions, when requested by `ct_lut_anat_sdf`, are saved as input
    context only and never alter label generation or case selection.
    """
    if v5_input not in V5_INPUT_VARIANTS:
        raise ValueError(f"--v5-input must be one of {V5_INPUT_VARIANTS}, got {v5_input!r}")
    if v5_target_profile not in V5_TARGET_PROFILES:
        raise ValueError(f"--v5-target-profile must be one of {V5_TARGET_PROFILES}, got {v5_target_profile!r}")
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "bicm_v5":
        raise ValueError(f"Dataset{ds_id} is not a BICM V5 dataset")
    dst = NN_RAW / cfg["name"]
    images_dir = dst / "imagesTr"
    labels_dir = dst / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    cases = _select_case_subset(list_cases("pelvic"), case_subset)
    params = BICMV5Params(
        target_profile=v5_target_profile,
        core_ball_radius_mm=float(v5_core_ball_radius_mm),
        core_body_mm=float(v5_core_body_mm),
        contact_band_mm=float(v5_contact_band_mm),
    )
    pad_vox = int(os.environ.get("PENGWIN_V5_ROI_PAD_VOX", "24"))
    uses_anatomy_context = v5_input == "ct_lut_anat_sdf"
    context_checkpoint = os.environ.get("PENGWIN_V7_CONTEXT_CHECKPOINT", "checkpoint_best.pth")
    context_root = None
    if uses_anatomy_context:
        # [AUDIT][Risk:High][Scope:input_ablation]
        # This is the only changed variable for the V7 root-cause run:
        # target, decoder, trainer, loss, and ROI samples stay fixed while the
        # model receives selected Dataset532 support context. The cache path and
        # checkpoint are recorded below so later fold0 work can replace this with
        # out-of-fold anatomy probabilities instead of silently reusing fold0.
        context_root = _ensure_ds532_anatomy_probability_context(
            cases,
            force=os.environ.get("PENGWIN_V7_FORCE_CONTEXT", "0") == "1",
            gpu=int(os.environ.get("PENGWIN_V7_CONTEXT_GPU", "0")),
            checkpoint=context_checkpoint,
        )
    if uses_anatomy_context:
        channel_names = {"0": "nonorm", "1": "nonorm", "2": "nonorm"}
        input_contract = [
            "ct_lut",
            "Dataset532_selected_anatomy_probability",
            "Dataset532_selected_anatomy_sdf",
        ]
    else:
        channel_names = {"0": "nonorm"}
        input_contract = ["ct_lut"]
    rows = []
    print(
        f"[bicm_v5] Ds{ds_id} {cfg['name']} — {len(cases)} pelvic CT cases "
        f"x {len(V5_ANATOMY_RANGES)} anatomy ROI samples input={v5_input} "
        f"target={v5_target_profile}"
    )
    for cd in cases:
        cid = int(cd.name)
        img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        assert_lps_image(img, f"Ds{ds_id} case {cid:03d} CT")
        assert_lps_image(lbl_img, f"Ds{ds_id} case {cid:03d} label")
        assert_matching_geometry(img, lbl_img, f"Ds{ds_id} case {cid:03d}")
        inst_full = sitk.GetArrayFromImage(lbl_img).astype(np.uint16)
        spacing_zyx = tuple(float(v) for v in lbl_img.GetSpacing()[::-1])
        image_channel = _bone_lut_normalize(arr_img)
        context_probs: dict[str, np.ndarray] = {}
        if uses_anatomy_context:
            assert context_root is not None
            prob_path = context_root / f"{cid:03d}" / "image.npz"
            probs = _load_probability_npz(prob_path)
            if tuple(probs.shape[1:]) != tuple(arr_img.shape):
                raise RuntimeError(
                    f"Dataset532 context shape mismatch for case {cid:03d}: "
                    f"prob={tuple(probs.shape[1:])}, ct={tuple(arr_img.shape)}"
                )
            if probs.shape[0] <= max(V5_DATASET532_PROB_CHANNEL.values()):
                raise RuntimeError(
                    f"Dataset532 context for case {cid:03d} has {probs.shape[0]} channels; "
                    f"need anatomy channels {V5_DATASET532_PROB_CHANNEL}"
                )
            for anatomy, prob_channel in V5_DATASET532_PROB_CHANNEL.items():
                prob = np.asarray(probs[int(prob_channel)], dtype=np.float32)
                context_probs[anatomy] = prob
        for anatomy in V5_ANATOMY_RANGES:
            sample_id = f"PENGWIN_{cid:03d}_{anatomy}"
            out_imgs = [images_dir / f"{sample_id}_{idx:04d}.mha" for idx in range(len(channel_names))]
            out_lbl = labels_dir / f"{sample_id}.mha"
            if all(p.exists() for p in out_imgs) and out_lbl.exists() and not force:
                continue
            anat_mask = anatomy_mask_from_instances(inst_full, anatomy)
            bbox = bbox_from_mask(anat_mask, pad_vox=pad_vox)
            if bbox is None:
                raise RuntimeError(f"Ds{ds_id} case {cid:03d} {anatomy}: empty anatomy ROI")
            lo, hi = V5_ANATOMY_RANGES[anatomy]
            inst_roi = inst_full[bbox]
            inst_roi = np.where((inst_roi >= lo) & (inst_roi <= hi), inst_roi, 0).astype(np.uint16, copy=False)
            target = compute_bicm_v5_target(inst_roi, spacing_zyx=spacing_zyx, params=params)
            values = set(int(v) for v in np.unique(target))
            allowed = set(V5_LABELS.values())
            if not values.issubset(allowed):
                raise RuntimeError(f"{sample_id}: V5 labels {sorted(values)} outside {sorted(allowed)}")
            if not (target == V5_LABELS["core"]).any():
                raise RuntimeError(f"{sample_id}: no V5 core voxels")
            crop_save_mha(image_channel, img, bbox, out_imgs[0], dtype=np.float32)
            row_extra = {}
            if uses_anatomy_context:
                prob = context_probs[anatomy]
                # [DATA][Leakage]
                # The selected anatomy channels are cropped by the same GT ROI
                # as the CT for this six-case overfit diagnostic. They are not
                # used to create targets; production fold0 must switch the cache
                # to out-of-fold anatomy predictions before this profile can be
                # considered promotion evidence.
                crop_save_mha(prob, img, bbox, out_imgs[1], dtype=np.float32)
                prob_roi = prob[bbox]
                # [QC][Perf:roi_sdf]
                # Full-volume EDT on six large CTs is several minutes and does
                # not change the experiment question. Compute SDF only inside
                # the materialized ROI because the model never sees voxels
                # outside this crop.
                sdf_roi = _selected_anatomy_sdf_from_prob(prob_roi, spacing_zyx=spacing_zyx)
                save_crop_array_mha(sdf_roi, img, bbox, out_imgs[2], dtype=np.float32)
                row_extra = {
                    "anatomy_prob_mean": float(np.mean(prob_roi)),
                    "anatomy_prob_fg_fraction": float(np.mean(prob_roi >= 0.5)),
                    "anatomy_sdf_mean": float(np.mean(sdf_roi)),
                }
            save_crop_array_mha(target, lbl_img, bbox, out_lbl, dtype=np.uint8)
            row = {
                "case": f"{cid:03d}",
                "sample": sample_id,
                "anatomy": anatomy,
                "bbox_zyx": [[int(s.start), int(s.stop)] for s in bbox],
                "labels": label_distribution(target),
                "input_channels": len(channel_names),
            }
            row.update(row_extra)
            rows.append(row)
    ds_json = {
        "channel_names": channel_names,
        "labels": dict(V5_LABELS),
        "numTraining": len(cases) * len(V5_ANATOMY_RANGES),
        "file_ending": ".mha",
        "v5_contract": {
            "input": v5_input,
            "input_contract": input_contract,
            "roi_source": (
                "GT anatomy IDs for raw target/oracle ROI; Dataset532 context, "
                "if present, is input-only and must be out-of-fold for promotion"
            ),
            "anatomy_context": {
                "enabled": uses_anatomy_context,
                "dataset": DATASETS[532]["name"] if uses_anatomy_context else None,
                "checkpoint": context_checkpoint if uses_anatomy_context else None,
                "cache_root": str(context_root) if uses_anatomy_context and context_root is not None else None,
                "channel_map": V5_DATASET532_PROB_CHANNEL if uses_anatomy_context else None,
                "sdf_clip_mm": 40.0 if uses_anatomy_context else None,
            },
            "anatomies": list(V5_ANATOMY_RANGES),
            "target_params": {
                "exterior_mm": params.exterior_mm,
                "core_mm": params.core_mm,
                "core_fallback_radius_vox": params.core_fallback_radius_vox,
                "target_profile": params.target_profile,
                "core_ball_radius_mm": params.core_ball_radius_mm,
                "core_body_mm": params.core_body_mm,
                "contact_band_mm": params.contact_band_mm,
                "roi_pad_vox": pad_vox,
            },
            "decoder": "core connected components + contact-surface watershed, no threshold sweep",
        },
        "description": (
            f"PENGWIN 2026 — {cfg['name']} V5 per-anatomy BICM target. "
            "One sample is one CT-LUT anatomy ROI. Labels are background, "
            "exterior context, interior shell, core, and contact surface."
        ),
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    audit_path = RESULT_REPORT / f"build_bicm_v5_dataset{ds_id}_{RESULT_DATE}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "dataset": cfg["name"],
        "input_variant": v5_input,
        "input_contract": input_contract,
        "anatomy_context_root": str(context_root) if context_root is not None else None,
        "anatomy_context_checkpoint": context_checkpoint if uses_anatomy_context else None,
        "target_profile": params.target_profile,
        "core_ball_radius_mm": params.core_ball_radius_mm,
        "core_body_mm": params.core_body_mm,
        "contact_band_mm": params.contact_band_mm,
        "samples": rows,
    }, indent=2))
    print(f"  raw audit: {audit_path}")
    print(f"  ✅ {cfg['name']} ready")
    return len(cases) * len(V5_ANATOMY_RANGES)



def build_dataset(ds_id: int,
                  force: bool = False,
                  abbc_official: bool = False,
                  v3_input: str = "ct_hu",
                  v4_input: str = "ct_lut",
                  v5_input: str = "ct_lut",
                  v5_target_profile: str = "v5_tiny_marker",
                  v5_core_ball_radius_mm: float = 2.5,
                  v5_core_body_mm: float = 3.0,
                  v5_contact_band_mm: float = 2.0,
                  case_subset: list[str] | None = None) -> int:
    cfg = DATASETS[ds_id]
    if cfg["kind"] == "anatomy_semantic":
        return build_anatomy_semantic_dataset(ds_id, force=force)
    if cfg["kind"] == "abbc_official":
        return build_abbc_official_dataset(ds_id, force=force)
    if cfg["kind"] == "boundary_fragment":
        return build_boundary_fragment_v3_dataset(
            ds_id,
            force=force,
            v3_input=v3_input,
            case_subset=case_subset,
        )
    if cfg["kind"] == "contact_instance":
        return build_contact_instance_dataset(ds_id, force=force)
    if cfg["kind"] == "factorized_instance":
        return build_factorized_instance_v4_dataset(
            ds_id,
            force=force,
            v4_input=v4_input,
            case_subset=case_subset,
        )
    if cfg["kind"] == "bicm_v5":
        return build_bicm_v5_dataset(
            ds_id,
            force=force,
            v5_input=v5_input,
            v5_target_profile=v5_target_profile,
            v5_core_ball_radius_mm=v5_core_ball_radius_mm,
            v5_core_body_mm=v5_core_body_mm,
            v5_contact_band_mm=v5_contact_band_mm,
            case_subset=case_subset,
        )
    raise ValueError(f"Unknown dataset kind: {cfg['kind']}")


def audit_nnunet_raw_datasets(ds_ids: list[int] | None = None,
                              out_json: Path | None = None,
                              fail_on_mixed_orientation: bool = True) -> dict:
    """Audit generated nnU-Net raw datasets after conversion.

    The rebuild gate is intentionally strict: every image channel must be LPS,
    every dataset must have `dataset.json`, and channel counts must match
    labels. This catches exactly the previous failure mode where code claimed
    canonicalization but generated raw files remained mixed LPS/RAS.
    """
    ds_ids = ds_ids or sorted(DATASETS)
    audit = {"datasets": {}, "ok": True}
    for ds_id in ds_ids:
        cfg = DATASETS[ds_id]
        dst = NN_RAW / cfg["name"]
        images = sorted((dst / "imagesTr").glob("*.mha"))
        labels = sorted((dst / "labelsTr").glob("*.mha"))
        case_to_channels: dict[str, list[str]] = defaultdict(list)
        orientation_counts = Counter()
        geometry_errors = []
        label_values_seen = set()
        invalid_label_cases = []
        subject_label_violations = []
        empty_foreground_cases = []
        source_type_counts = Counter()
        audited_label_cases = set()
        for img_path in images:
            stem = img_path.stem
            case_key = stem.rsplit("_", 1)[0] if "_" in stem else stem
            case_to_channels[case_key].append(img_path.name)
            img_info = _read_image_info(img_path)
            code = img_info["orientation"]
            orientation_counts[code] += 1
            if case_key in audited_label_cases:
                continue
            audited_label_cases.add(case_key)
            label_path = dst / "labelsTr" / f"{case_key}.mha"
            if not label_path.exists():
                continue
            lbl_info = _read_image_info(label_path)
            if not _geometry_tuples_match(img_info, lbl_info):
                geometry_errors.append({
                    "case": case_key,
                    "error": "image/label geometry mismatch",
                    "image": img_info,
                    "label": lbl_info,
                })
            lbl = sitk.ReadImage(str(label_path))
            arr = sitk.GetArrayFromImage(lbl)
            values = set(int(v) for v in np.unique(arr))
            label_values_seen.update(values)
            allowed = set(range(int(cfg["n_classes"])))
            invalid = sorted(values - allowed)
            if invalid:
                invalid_label_cases.append({"case": case_key, "invalid_values": invalid})
            if not (arr > 0).any():
                empty_foreground_cases.append(case_key)
            try:
                cid = int(case_key.replace("PENGWIN_", ""))
            except ValueError:
                cid = -1
            source_type = case_subject_type(cid)
            source_type_counts[source_type] += 1
            if cfg["kind"] == "anatomy_semantic":
                expected_subject = cfg["filter"].capitalize() if cfg["filter"] in {"pelvic", "femur"} else None
                if expected_subject and source_type != expected_subject:
                    subject_label_violations.append({
                        "case": case_key,
                        "subject_type": source_type,
                        "expected_subject_type": expected_subject,
                        "values": sorted(values),
                    })
        n_cases = len(case_to_channels)
        dataset_json_path = dst / "dataset.json"
        if dataset_json_path.exists():
            expected_channels = len(json.loads(dataset_json_path.read_text()).get("channel_names", {"0": "CT"}))
        else:
            expected_channels = 1
        bad_channel_cases = {
            k: sorted(v) for k, v in case_to_channels.items()
            if len(v) != expected_channels
        }
        if dataset_json_path.exists():
            expected_cases = int(json.loads(dataset_json_path.read_text()).get(
                "numTraining",
                len(list_cases(case_filter=None if cfg["filter"] == "all" else cfg["filter"])),
            ))
        else:
            expected_cases = len(list_cases(case_filter=None if cfg["filter"] == "all" else cfg["filter"]))
        entry = {
            "dataset_id": ds_id,
            "name": cfg["name"],
            "kind": cfg["kind"],
            "dataset_json": (dst / "dataset.json").exists(),
            "n_images": len(images),
            "n_labels": len(labels),
            "n_cases": n_cases,
            "expected_cases": expected_cases,
            "expected_channels": expected_channels,
            "bad_channel_cases": bad_channel_cases,
            "orientation_counts": dict(orientation_counts),
            "source_type_counts": dict(source_type_counts),
            "label_values": sorted(label_values_seen),
            "geometry_errors": geometry_errors,
            "invalid_label_cases": invalid_label_cases,
            "subject_label_violations": subject_label_violations,
            "empty_foreground_cases": empty_foreground_cases,
        }
        entry["ok"] = (
            entry["dataset_json"]
            and n_cases == expected_cases
            and len(labels) == n_cases
            and not bad_channel_cases
            and set(orientation_counts) == {"LPS"}
            and not geometry_errors
            and not invalid_label_cases
            and not subject_label_violations
            and not empty_foreground_cases
        )
        audit["datasets"][str(ds_id)] = entry
        audit["ok"] = bool(audit["ok"] and entry["ok"])

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(audit, indent=2))
    if fail_on_mixed_orientation and not audit["ok"]:
        failed = [k for k, v in audit["datasets"].items() if not v["ok"]]
        raise RuntimeError(f"nnUNet raw dataset audit failed: {failed}")
    return audit


def _audit_abbc_target_case_worker(args: tuple[int, str, str]) -> dict:
    """Audit one ABBC target case for multiprocessing."""
    ds_id, cid, target_path_str = args
    cfg = DATASETS[ds_id]
    case_dir = find_case_dir(cid)
    if case_dir is None:
        return {"case": cid, "source_case_missing": True}
    source = sitk.GetArrayFromImage(canonicalize_sitk(sitk.ReadImage(str(case_dir / "label.mha"))))
    target = sitk.GetArrayFromImage(sitk.ReadImage(str(target_path_str)))
    values = set(int(v) for v in np.unique(target))
    allowed = _abbc_allowed_values(ds_id)
    invalid = sorted(values - allowed)
    case_audit = audit_official_target(source, target)
    counts = np.bincount(target.ravel(), minlength=int(cfg["n_classes"]))
    return {
        "case": cid,
        "source_case_missing": False,
        "invalid_values": invalid,
        "case_audit": case_audit,
        "label_counts": {str(k): int(counts[int(k)]) for k in sorted(allowed)},
    }


def audit_abbc_targets(ds_ids: list[int] | None = None,
                       out_json: Path | None = None,
                       fail_on_error: bool = True) -> dict:
    """Audit Contact-Energy ABBC raw labels before training.

    Checks the generated target-side seed contract for active ABBC datasets:
        - label values are exactly within {0,1,2,3};
        - class order is `[boundary_shell, core, contact_surface]`;
        - every pelvic GT fragment has one target core connected component;
        - no core voxel leaks outside pelvic fragment labels 1..150;
        - core/contact overlap and tiny-fragment seed coverage are tracked.
    """
    selected_ids = [
        ds_id for ds_id in (ds_ids or sorted(DATASETS))
        if DATASETS[ds_id]["kind"] == "abbc_official"
    ]
    audit = {"datasets": {}, "ok": True}
    for ds_id in selected_ids:
        cfg = DATASETS[ds_id]
        raw_root = NN_RAW / cfg["name"]
        label_paths = {p.stem.replace("PENGWIN_", ""): p for p in sorted((raw_root / "labelsTr").glob("*.mha"))}
        per_case = {}
        invalid_label_cases = []
        zero_core = []
        multi_core = []
        core_leak = []
        label_voxels = Counter()
        core_cc_hist = Counter()
        fragment_count = 0
        workers = max(1, int(os.environ.get("PENGWIN_ABBC_AUDIT_WORKERS", "8")))
        workers = min(workers, max(1, len(label_paths)))
        items = [(ds_id, cid, str(target_path)) for cid, target_path in sorted(label_paths.items())]
        if workers == 1:
            rows = [_audit_abbc_target_case_worker(item) for item in items]
        else:
            rows = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_audit_abbc_target_case_worker, item) for item in items]
                for future in as_completed(futures):
                    rows.append(future.result())
            rows.sort(key=lambda row: row["case"])

        for row in rows:
            cid = row["case"]
            if row.get("source_case_missing"):
                zero_core.append({"case": cid, "error": "source_case_missing"})
                continue
            invalid = row["invalid_values"]
            if invalid:
                invalid_label_cases.append({"case": cid, "invalid_values": invalid})
            case_audit = row["case_audit"]
            per_case[cid] = case_audit
            fragment_count += int(case_audit["n_fragments"])
            for k, v in row["label_counts"].items():
                label_voxels[int(k)] += int(v)
            for k, v in case_audit["core_cc_hist"].items():
                core_cc_hist[int(k)] += int(v)
            for fragment_id in case_audit["zero_core"]:
                zero_core.append({"case": cid, "fragment_id": int(fragment_id)})
            for row in case_audit["multi_core"]:
                row = dict(row)
                row["case"] = cid
                multi_core.append(row)
            if case_audit["core_leak_voxels"]:
                core_leak.append({"case": cid, "voxels": int(case_audit["core_leak_voxels"])})

        expected_cases = len(list_cases(case_filter=cfg["filter"]))
        total_voxels = sum(label_voxels.values())
        labels = _abbc_labels_for_dataset(ds_id)
        entry = {
            "dataset_id": ds_id,
            "name": cfg["name"],
            "kind": cfg["kind"],
            "anatomies": cfg["anatomies"],
            "global_label_range": list(cfg["global_label_range"] or (1, 150)),
            "n_labels": len(label_paths),
            "expected_cases": expected_cases,
            "n_fragments": fragment_count,
            "label_values_allowed": sorted(labels.values()),
            "label_contract": {str(v): k for k, v in labels.items()},
            "abbc_official": json.loads((NN_RAW / cfg["name"] / "dataset.json").read_text()).get(
                "abbc_official", {}
            ) if (NN_RAW / cfg["name"] / "dataset.json").exists() else {},
            "target_label_voxels": {
                str(k): int(label_voxels.get(k, 0))
                for k in sorted(labels.values())
            },
            "target_label_fractions": {
                str(k): float(label_voxels.get(k, 0)) / max(1, total_voxels)
                for k in sorted(labels.values())
            },
            "target_qc_note": (
                "Contact-Energy ABBC pseudo dice excludes background and maps to "
                "[boundary_shell, core, contact_surface"
                f"{', contact_hard_negative' if int(cfg['n_classes']) >= 5 else ''}]. "
                "Class 3 is the separator target; class 4 is negative supervision only."
            ),
            "core_cc_hist": {str(k): int(v) for k, v in sorted(core_cc_hist.items())},
            "zero_core": zero_core,
            "multi_core": multi_core,
            "core_leak": core_leak,
            "invalid_label_cases": invalid_label_cases,
            "per_case": per_case,
        }
        entry["ok"] = (
            len(label_paths) == expected_cases
            and not zero_core
            and not multi_core
            and not core_leak
            and not invalid_label_cases
        )
        audit["datasets"][str(ds_id)] = entry
        audit["ok"] = bool(audit["ok"] and entry["ok"])

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(audit, indent=2))
    if fail_on_error and not audit["ok"]:
        failed = [k for k, v in audit["datasets"].items() if not v["ok"]]
        raise RuntimeError(f"ABBC target audit failed: {failed}")
    return audit


def audit_boundary_fragment_targets(ds_ids: list[int] | None = None,
                                    out_json: Path | None = None,
                                    fail_on_error: bool = True) -> dict:
    """Audit Dataset537 V3 raw target invariants."""
    selected_ids = [
        ds_id for ds_id in (ds_ids or sorted(DATASETS))
        if DATASETS[ds_id]["kind"] == "boundary_fragment"
    ]
    audit = {"datasets": {}, "ok": True}
    for ds_id in selected_ids:
        cfg = DATASETS[ds_id]
        raw_root = NN_RAW / cfg["name"]
        label_paths = sorted((raw_root / "labelsTr").glob("*.mha"))
        label_voxels = Counter()
        invalid_cases = []
        overlap_cases = []
        zero_core_cases = []
        external_inside_cases = []
        for path in label_paths:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
            values = set(int(v) for v in np.unique(arr))
            invalid = sorted(values - set(BFV3_LABELS.values()))
            if invalid:
                invalid_cases.append({"case": path.stem, "invalid_values": invalid})
            counts = np.bincount(arr.ravel(), minlength=5)
            for i in range(5):
                label_voxels[i] += int(counts[i])
            if int(counts[4]) == 0:
                zero_core_cases.append(path.stem)
            # A class label cannot literally overlap another class in an argmax
            # target, but these fields make the contract explicit in the audit.
            if ((arr == 1) & ((arr == 2) | (arr == 3) | (arr == 4))).any():
                external_inside_cases.append(path.stem)
            if ((arr == 1) & (arr == 2)).any():
                overlap_cases.append(path.stem)
        total = sum(label_voxels.values())
        entry = {
            "dataset_id": ds_id,
            "name": cfg["name"],
            "kind": cfg["kind"],
            "n_labels": len(label_paths),
            "label_contract": {str(v): k for k, v in BFV3_LABELS.items()},
            "label_voxels": {str(k): int(label_voxels.get(k, 0)) for k in range(5)},
            "label_fractions": {str(k): float(label_voxels.get(k, 0)) / max(1, total) for k in range(5)},
            "invalid_cases": invalid_cases,
            "class1_class2_overlap_cases": overlap_cases,
            "class1_inside_support_cases": external_inside_cases,
            "zero_core_cases": zero_core_cases,
            "target_qc_note": (
                "V3.1 uses class2 as a thin contact ridge and class4 as decoder seed. "
                "Every case must have class4; class1 must never be fragment support."
            ),
        }
        entry["ok"] = bool(
            label_paths
            and not invalid_cases
            and not overlap_cases
            and not external_inside_cases
            and not zero_core_cases
        )
        audit["datasets"][str(ds_id)] = entry
        audit["ok"] = bool(audit["ok"] and entry["ok"])
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(audit, indent=2))
    if fail_on_error and not audit["ok"]:
        failed = [k for k, v in audit["datasets"].items() if not v["ok"]]
        raise RuntimeError(f"BoundaryFragment target audit failed: {failed}")
    return audit


def _geometry_tuple(img: sitk.Image) -> tuple:
    """Compact geometry tuple for exact source->raw orientation contract checks."""
    return (
        tuple(int(v) for v in img.GetSize()),
        tuple(round(float(v), 6) for v in img.GetSpacing()),
        tuple(round(float(v), 5) for v in img.GetOrigin()),
        tuple(round(float(v), 6) for v in img.GetDirection()),
    )


def _read_image_info(path: Path) -> dict:
    """Read MHA geometry metadata without loading the voxel array."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    direction = tuple(float(v) for v in reader.GetDirection())
    return {
        "path": str(path),
        "orientation": sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(direction),
        "size": tuple(int(v) for v in reader.GetSize()),
        "spacing": tuple(float(v) for v in reader.GetSpacing()),
        "origin": tuple(float(v) for v in reader.GetOrigin()),
        "direction": direction,
    }


def _geometry_tuple_from_info(info: dict) -> tuple:
    """Compact geometry tuple from `_read_image_info` output."""
    return (
        tuple(int(v) for v in info["size"]),
        tuple(round(float(v), 6) for v in info["spacing"]),
        tuple(round(float(v), 5) for v in info["origin"]),
        tuple(round(float(v), 6) for v in info["direction"]),
    )


def _geometry_tuples_match(a: dict, b: dict) -> bool:
    return (
        tuple(a["size"]) == tuple(b["size"])
        and np.allclose(a["spacing"], b["spacing"], rtol=0, atol=1e-5)
        and np.allclose(a["origin"], b["origin"], rtol=0, atol=1e-4)
        and np.allclose(a["direction"], b["direction"], rtol=0, atol=1e-5)
    )


def _geometry_tuple_matches_info(geom: tuple, info: dict) -> bool:
    return (
        tuple(geom[0]) == tuple(info["size"])
        and np.allclose(geom[1], info["spacing"], rtol=0, atol=1e-5)
        and np.allclose(geom[2], info["origin"], rtol=0, atol=1e-4)
        and np.allclose(geom[3], info["direction"], rtol=0, atol=1e-5)
    )


def audit_orientation_contract(ds_ids: list[int] | None = None,
                               out_json: Path | None = None,
                               fail_on_error: bool = True) -> dict:
    """Audit the training/inference orientation contract for active datasets.

    The critical invariant is stronger than "raw files are LPS": for every
    source case, `canonicalize_sitk(source image, LPS)` must have the same
    geometry as the generated nnU-Net raw image. This catches the exact failure
    mode where training consumed LPS raw files but standalone inference copied
    original RAS source bytes into nnU-Net.
    """
    ds_ids = ds_ids or sorted(DATASETS)
    audit = {
        "contract_version": "lps_ct_v1",
        "canonical_orientation": "LPS",
        "source_orientation_counts": {},
        "non_lps_source_cases": [],
        "datasets": {},
        "ok": True,
    }
    global_source_counts = Counter()
    global_non_lps: dict[str, dict] = {}

    for ds_id in ds_ids:
        cfg = DATASETS[ds_id]
        dst = NN_RAW / cfg["name"]
        cases = list_cases(case_filter=None if cfg["filter"] == "all" else cfg["filter"])
        source_counts = Counter()
        raw_image_counts = Counter()
        raw_label_counts = Counter()
        raw_geometry_errors = []
        canonical_geometry_errors = []
        missing_raw_cases = []
        image_label_geometry_errors = []

        for cd in cases:
            cid = cd.name.zfill(3)
            source_img_info = _read_image_info(cd / "image.mha")
            source_lbl_info = _read_image_info(cd / "label.mha")
            source_code = source_img_info["orientation"]
            source_counts[source_code] += 1
            global_source_counts[source_code] += 1
            if source_code != "LPS":
                global_non_lps[cid] = {
                    "case": cid,
                    "source_orientation": source_code,
                    "subject_type": case_subject_type(int(cid)),
                    "source_path": str(cd / "image.mha"),
                }

            raw_img_path = dst / "imagesTr" / f"PENGWIN_{cid}_0000.mha"
            raw_lbl_path = dst / "labelsTr" / f"PENGWIN_{cid}.mha"
            if not raw_img_path.exists() or not raw_lbl_path.exists():
                missing_raw_cases.append({
                    "case": cid,
                    "missing_image": not raw_img_path.exists(),
                    "missing_label": not raw_lbl_path.exists(),
                })
                continue

            raw_img_info = _read_image_info(raw_img_path)
            raw_lbl_info = _read_image_info(raw_lbl_path)
            raw_image_counts[raw_img_info["orientation"]] += 1
            raw_label_counts[raw_lbl_info["orientation"]] += 1
            if not _geometry_tuples_match(raw_img_info, raw_lbl_info):
                image_label_geometry_errors.append({
                    "case": cid,
                    "error": "raw image/label geometry mismatch",
                    "raw_image": raw_img_info,
                    "raw_label": raw_lbl_info,
                })

            if source_code == "LPS":
                canonical_source_img_tuple = _geometry_tuple_from_info(source_img_info)
            else:
                canonical_source_img_tuple = _geometry_tuple(canonicalize_sitk(
                    sitk.ReadImage(str(cd / "image.mha"))
                ))
            if source_lbl_info["orientation"] == "LPS":
                canonical_source_lbl_tuple = _geometry_tuple_from_info(source_lbl_info)
            else:
                canonical_source_lbl_tuple = _geometry_tuple(canonicalize_sitk(
                    sitk.ReadImage(str(cd / "label.mha"))
                ))

            if not _geometry_tuple_matches_info(canonical_source_img_tuple, raw_img_info):
                canonical_geometry_errors.append({
                    "case": cid,
                    "kind": "image",
                    "source_orientation": source_code,
                    "raw_orientation": raw_img_info["orientation"],
                    "error": "canonical source image geometry does not match raw imagesTr",
                })
            if not _geometry_tuple_matches_info(canonical_source_lbl_tuple, raw_lbl_info):
                canonical_geometry_errors.append({
                    "case": cid,
                    "kind": "label",
                    "source_orientation": source_lbl_info["orientation"],
                    "raw_orientation": raw_lbl_info["orientation"],
                    "error": "canonical source label geometry does not match raw labelsTr",
                })
            if raw_img_info["orientation"] != "LPS" or raw_lbl_info["orientation"] != "LPS":
                raw_geometry_errors.append({
                    "case": cid,
                    "raw_image_orientation": raw_img_info["orientation"],
                    "raw_label_orientation": raw_lbl_info["orientation"],
                })

        entry = {
            "dataset_id": ds_id,
            "name": cfg["name"],
            "n_cases": len(cases),
            "source_orientation_counts": dict(source_counts),
            "raw_image_orientation_counts": dict(raw_image_counts),
            "raw_label_orientation_counts": dict(raw_label_counts),
            "non_lps_source_cases": [
                case for case, meta in sorted(global_non_lps.items())
                if case in {cd.name.zfill(3) for cd in cases}
            ],
            "missing_raw_cases": missing_raw_cases,
            "raw_non_lps_cases": raw_geometry_errors,
            "image_label_geometry_errors": image_label_geometry_errors,
            "source_to_raw_geometry_errors": canonical_geometry_errors,
        }
        entry["ok"] = (
            not missing_raw_cases
            and not raw_geometry_errors
            and not image_label_geometry_errors
            and not canonical_geometry_errors
            and set(raw_image_counts) == {"LPS"}
            and set(raw_label_counts) == {"LPS"}
        )
        audit["datasets"][str(ds_id)] = entry
        audit["ok"] = bool(audit["ok"] and entry["ok"])

    audit["source_orientation_counts"] = dict(global_source_counts)
    audit["non_lps_source_cases"] = list(sorted(global_non_lps.values(), key=lambda x: x["case"]))
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(audit, indent=2))
    if fail_on_error and not audit["ok"]:
        failed = [k for k, v in audit["datasets"].items() if not v["ok"]]
        raise RuntimeError(f"orientation contract audit failed: {failed}")
    return audit


def _training_identifiers_from_raw(ds_id: int) -> list[str]:
    """Return nnU-Net case identifiers from raw labelsTr.

    nnU-Net stores split identifiers without channel suffixes and without file
    endings, for example `PENGWIN_003`. Using labelsTr as the source keeps
    labelsTr is the authoritative source for one split identifier per case.
    """
    cfg = DATASETS[ds_id]
    labels = sorted((NN_RAW / cfg["name"] / "labelsTr").glob("*.mha"))
    return [p.stem for p in labels]


def _case_int_from_identifier(identifier: str) -> int:
    """Convert nnU-Net identifier to the source integer case ID.

    [DATA][Risk:High][Scope:split_leakage]
    V5 uses per-anatomy identifiers such as `PENGWIN_003_LeftHip`. Splits must
    still be grouped by source case `003`, otherwise Sacrum can be in train
    while LeftHip from the same CT is in validation.
    """
    stem = identifier.replace("PENGWIN_", "")
    return int(stem.split("_", 1)[0])


def _identifier_sort_key(identifier: str) -> tuple[int, str]:
    return _case_int_from_identifier(identifier), identifier


def _source_fragment_count(identifier: str) -> int:
    """Return original GT fragment count for stratified split balancing."""
    cid = f"{_case_int_from_identifier(identifier):03d}"
    case_dir = find_case_dir(cid)
    if case_dir is None:
        raise FileNotFoundError(f"source case missing for split generation: {identifier}")
    lbl = canonicalize_sitk(sitk.ReadImage(str(case_dir / "label.mha")))
    arr = sitk.GetArrayFromImage(lbl)
    return int(len([v for v in np.unique(arr) if int(v) > 0]))


def _round_robin_by_fragment_count(identifiers: list[str],
                                   n_splits: int) -> list[list[str]]:
    """Distribute high-fragment source cases across folds.

    [DATA][Leakage:patient_case]
    Multiple nnU-Net identifiers can share one source CT in Dataset537 V5
    (`PENGWIN_003_Sacrum`, `PENGWIN_003_LeftHip`, ...). Keep those identifiers
    in the same fold so validation remains source-case isolated.
    """
    by_case: dict[int, list[str]] = defaultdict(list)
    for identifier in identifiers:
        by_case[_case_int_from_identifier(identifier)].append(identifier)
    rows = [
        (_source_fragment_count(group[0]), cid, sorted(group, key=_identifier_sort_key))
        for cid, group in by_case.items()
    ]
    rows.sort(key=lambda x: (-x[0], x[1]))
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for i, (_n_frag, _cid, group) in enumerate(rows):
        folds[i % n_splits].extend(group)
    return [sorted(fold, key=_identifier_sort_key) for fold in folds]


def _generate_split_anatomy_splits(train_identifiers: list[str],
                                   n_splits: int = 5) -> list[dict[str, list[str]]]:
    """Generate fragment-count-balanced folds for split anatomy datasets.

    Within each subject group we sort by GT fragment count and round-robin
    distribute, which keeps high-fragment hard cases spread out.
    """
    pelvic = [i for i in train_identifiers if is_pelvic(_case_int_from_identifier(i))]
    femur = [i for i in train_identifiers if is_femur(_case_int_from_identifier(i))]
    if pelvic and femur:
        groups = [
            _round_robin_by_fragment_count(pelvic, n_splits),
            _round_robin_by_fragment_count(femur, n_splits),
        ]
    else:
        groups = [_round_robin_by_fragment_count(train_identifiers, n_splits)]
    all_ids = set(train_identifiers)
    splits = []
    for fold_idx in range(n_splits):
        val = sorted([identifier for group in groups for identifier in group[fold_idx]],
                     key=_identifier_sort_key)
        train = sorted(all_ids - set(val), key=_identifier_sort_key)
        splits.append({"train": train, "val": val})
    return splits


def _generate_crossval_splits(train_identifiers: list[str],
                              seed: int = 12345,
                              n_splits: int = 5) -> list[dict[str, list[str]]]:
    """Generate the same style of split file nnU-Net expects.

    Prefer nnU-Net's own helper when importable. The deterministic fallback is
    intentionally simple and only used for audit bootstrapping if the helper is
    unavailable in a stripped environment.
    """
    try:
        from nnunetv2.utilities.crossval_split import generate_crossval_split
        return generate_crossval_split(train_identifiers, seed=seed, n_splits=n_splits)
    except Exception:
        rng = np.random.RandomState(seed)
        ids = np.array(train_identifiers)
        perm = rng.permutation(len(ids))
        folds = np.array_split(perm, n_splits)
        out = []
        for i in range(n_splits):
            val_idx = set(int(x) for x in folds[i])
            train = [str(ids[j]) for j in range(len(ids)) if j not in val_idx]
            val = [str(ids[j]) for j in range(len(ids)) if j in val_idx]
            out.append({"train": train, "val": val})
        return out


def ensure_splits_final(ds_id: int, overwrite: bool = False) -> Path:
    """Create `splits_final.json` for a preprocessed dataset if missing.

    nnU-Net normally creates this lazily at training time. The Task1 rebuild
    gate wants the file before training so the fold schedule is auditable and
    stable. This helper writes the standard 5-fold split from rebuilt raw
    `labelsTr` identifiers.
    """
    cfg = DATASETS[ds_id]
    dst = NN_PREP / cfg["name"] / "splits_final.json"
    if dst.exists() and not overwrite:
        return dst
    if cfg["kind"] in {"abbc_official", "contact_instance", "boundary_fragment"} and cfg["foundation_dataset"] is not None:
        foundation_cfg = DATASETS[int(cfg["foundation_dataset"])]
        foundation_split = NN_PREP / foundation_cfg["name"] / "splits_final.json"
        if foundation_split.exists():
            # Full Dataset537 uses the same pelvic case set as Dataset532, but
            # root-cause mini builds can contain only six cases. Copy the
            # foundation split only when the identifiers match exactly.
            ids_here = set(_training_identifiers_from_raw(ds_id))
            foundation_splits = json.loads(foundation_split.read_text())
            ids_foundation = set()
            for split in foundation_splits:
                ids_foundation.update(split.get("train", []))
                ids_foundation.update(split.get("val", []))
            if ids_here == ids_foundation:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(foundation_split.read_text())
                return dst
    ids = _training_identifiers_from_raw(ds_id)
    if len(ids) < 5:
        raise RuntimeError(f"Ds{ds_id}: need at least 5 labelsTr cases for 5-fold split")
    if cfg["kind"] in {"anatomy_semantic", "abbc_official", "contact_instance", "boundary_fragment", "bicm_v5"}:
        splits = _generate_split_anatomy_splits(ids, n_splits=5)
    else:
        splits = _generate_crossval_splits(ids, seed=12345, n_splits=5)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(splits, indent=2))
    return dst


def audit_nnunet_preprocessed_datasets(ds_ids: list[int] | None = None,
                                       out_json: Path | None = None,
                                       ensure_splits: bool = False,
                                       fail_on_missing: bool = True) -> dict:
    """Audit planned/preprocessed nnU-Net datasets after plan_and_preprocess.

    Checks the concrete artifacts needed before training:
        - `dataset_fingerprint.json` from fingerprint extraction;
        - `nnUNetResEncUNetLPlans.json` from planning;
        - `nnUNetPlans_3d_fullres/*.npz` count matching raw labels;
        - `splits_final.json` for a stable fold schedule.

    Preprocessed arrays are `.npz/.pkl`, so physical orientation is no longer
    represented there. The orientation gate is therefore enforced at raw audit
    time, before nnU-Net resampling strips image direction metadata.
    """
    ds_ids = ds_ids or sorted(DATASETS)
    audit = {"datasets": {}, "ok": True}
    for ds_id in ds_ids:
        cfg = DATASETS[ds_id]
        root = NN_PREP / cfg["name"]
        config_dir = root / "nnUNetPlans_3d_fullres"
        ids = _training_identifiers_from_raw(ds_id)
        if ensure_splits and root.exists():
            ensure_splits_final(ds_id, overwrite=False)
        npz_files = sorted(config_dir.glob("*.npz")) if config_dir.is_dir() else []
        pkl_files = sorted(config_dir.glob("*.pkl")) if config_dir.is_dir() else []
        splits_path = root / "splits_final.json"
        split_summary = None
        if splits_path.exists():
            splits = json.loads(splits_path.read_text())
            split_summary = {
                "n_folds": len(splits),
                "fold_sizes": [
                    {
                        "train": len(s.get("train", [])),
                        "val": len(s.get("val", [])),
                        "val_pelvic": sum(
                            1 for identifier in s.get("val", [])
                            if is_pelvic(_case_int_from_identifier(identifier))
                        ),
                        "val_femur": sum(
                            1 for identifier in s.get("val", [])
                            if is_femur(_case_int_from_identifier(identifier))
                        ),
                    }
                    for s in splits
                ],
            }
        split_balanced = False
        if split_summary is not None and split_summary["n_folds"] > 0:
            # Six-case root-cause datasets are intentionally tiny. A correct
            # five-fold split is therefore 2/1/1/1/1 validation cases, not the
            # same integer count in every fold. Audit balanced fold sizes
            # instead of enforcing a single value.
            n_folds = int(split_summary["n_folds"])
            low = len(ids) // n_folds
            high = (len(ids) + n_folds - 1) // n_folds
            val_sizes = [int(item["val"]) for item in split_summary["fold_sizes"]]
            split_balanced = (
                sum(val_sizes) == len(ids)
                and all(low <= size <= high for size in val_sizes)
                and (max(val_sizes) - min(val_sizes) <= 1 if val_sizes else False)
            )
        entry = {
            "dataset_id": ds_id,
            "name": cfg["name"],
            "kind": cfg["kind"],
            "expected_cases": len(ids),
            "dataset_json": (root / "dataset.json").exists(),
            "fingerprint": (root / "dataset_fingerprint.json").exists(),
            "plans": (root / "nnUNetResEncUNetLPlans.json").exists(),
            "config_3d_fullres": config_dir.is_dir(),
            "n_npz": len(npz_files),
            "n_pkl": len(pkl_files),
            "splits_final": splits_path.exists(),
            "split_summary": split_summary,
        }
        entry["ok"] = (
            entry["dataset_json"]
            and entry["fingerprint"]
            and entry["plans"]
            and entry["config_3d_fullres"]
            and entry["n_npz"] == entry["expected_cases"]
            and entry["n_pkl"] == entry["expected_cases"]
            and entry["splits_final"]
            and (
                cfg["kind"] not in {"anatomy_semantic", "abbc_official", "contact_instance", "boundary_fragment"}
                or (
                    split_summary is not None
                    and split_balanced
                    and (
                        cfg["filter"] != "pelvic"
                        or all(item["val_pelvic"] == item["val"] and item["val_femur"] == 0
                               for item in split_summary["fold_sizes"])
                    )
                    and (
                        cfg["filter"] != "femur"
                        or all(item["val_femur"] == item["val"] and item["val_pelvic"] == 0
                               for item in split_summary["fold_sizes"])
                    )
                )
            )
        )
        audit["datasets"][str(ds_id)] = entry
        audit["ok"] = bool(audit["ok"] and entry["ok"])
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(audit, indent=2))
    if fail_on_missing and not audit["ok"]:
        failed = [k for k, v in audit["datasets"].items() if not v["ok"]]
        raise RuntimeError(f"nnUNet preprocessed dataset audit failed: {failed}")
    return audit


BOUNDARY_SAMPLING_DISTRIBUTIONS = {
    "boundary_fragment_v3": [(2, 0.50), (1, 0.20), (3, 0.20), (4, 0.10)],
    "boundary_fragment_v3_ridge80": [(2, 0.80), (1, 0.10), (3, 0.08), (4, 0.02)],
}


def _preprocessed_3d_fullres_folder(ds_id: int) -> Path:
    root = NN_PREP / DATASETS[ds_id]["name"]
    preferred = root / "nnUNetPlans_3d_fullres"
    if preferred.exists():
        return preferred
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_3d_fullres"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"no preprocessed 3d_fullres folder found under {root}")


def _plans_patch_size(ds_id: int, configuration: str = "3d_fullres") -> np.ndarray:
    root = NN_PREP / DATASETS[ds_id]["name"]
    plans_candidates = sorted(root.glob("*Plans.json"))
    if not plans_candidates:
        raise FileNotFoundError(f"plans json missing under {root}")
    plans = json.loads(plans_candidates[-1].read_text())
    cfg = plans.get("configurations", {}).get(configuration)
    if not cfg:
        raise KeyError(f"configuration {configuration!r} missing in {plans_candidates[-1]}")
    return np.asarray(cfg["patch_size"], dtype=np.int64)


def _class_location_key(class_locations: dict, label: int):
    for key, value in class_locations.items():
        try:
            if int(key) == int(label) and len(value) > 0:
                return key
        except (TypeError, ValueError):
            continue
    return None


def audit_boundary_sampling(ds_id: int = 537,
                            profile: str = "boundary_fragment_v3_ridge80",
                            batches: int = 100,
                            batch_size: int = 2,
                            cases: list[str] | None = None,
                            out_json: Path | None = None,
                            seed: int = 537031) -> dict:
    """Simulate BoundaryFragment forced crops and count class coverage.

    This intentionally avoids trainer initialization. It reads nnU-Net
    preprocessed segmentations and properties, then applies the same forced-class
    distribution used by the trainer to verify that the sparse class-2 ridge is
    actually present in sampled patches before spending GPU time.
    """

    if profile not in BOUNDARY_SAMPLING_DISTRIBUTIONS:
        raise ValueError(f"unknown sampling profile {profile!r}")
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

    folder = _preprocessed_3d_fullres_folder(ds_id)
    patch = _plans_patch_size(ds_id)
    case_ids = None
    if cases:
        case_ids = []
        for case in cases:
            text = str(case).strip()
            case_ids.append(text if text.startswith("PENGWIN_") else f"PENGWIN_{int(text):03d}")
    dataset = nnUNetDataset(str(folder), case_ids, num_images_properties_loading_threshold=0)
    keys = sorted(dataset.keys())
    if not keys:
        raise RuntimeError(f"no cases available for sampling audit in {folder}")
    case_cache = {}
    for key in keys:
        _data, seg, props = dataset.load_case(key)
        case_cache[str(key)] = {
            "labels": seg[0].astype(np.int16, copy=False),
            "class_locations": props.get("class_locations", {}),
        }

    rng = np.random.default_rng(int(seed))
    distribution = BOUNDARY_SAMPLING_DISTRIBUTIONS[profile]
    rows = []
    aggregate_counts = np.zeros(5, dtype=np.int64)
    class2_present = 0
    forced_label_counts: dict[str, int] = defaultdict(int)
    for _ in range(int(batches) * int(batch_size)):
        key = str(rng.choice(keys))
        cached = case_cache[key]
        labels = cached["labels"]
        class_locations = cached["class_locations"]
        available = [(label, prob) for label, prob in distribution if _class_location_key(class_locations, label) is not None]
        if not available:
            chosen_label = 0
            center = np.asarray(labels.shape, dtype=np.int64) // 2
        else:
            label_values = [x[0] for x in available]
            probs = np.asarray([x[1] for x in available], dtype=np.float64)
            probs /= probs.sum()
            chosen_label = int(rng.choice(label_values, p=probs))
            loc_key = _class_location_key(class_locations, chosen_label)
            locs = np.asarray(class_locations[loc_key])
            center = np.asarray(locs[int(rng.integers(0, locs.shape[0]))], dtype=np.int64)
            # nnU-Net properties may store class locations as (channel, z, y, x).
            # Sampling audits operate in segmentation voxel coordinates, so only
            # the final spatial triplet is meaningful for patch extraction.
            if center.size > 3:
                center = center[-3:]
        lo = np.maximum(center - patch // 2, 0)
        hi = np.minimum(lo + patch, np.asarray(labels.shape, dtype=np.int64))
        lo = np.maximum(hi - patch, 0)
        crop = labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        counts = np.bincount(crop.ravel().clip(0, 4), minlength=5).astype(np.int64)
        aggregate_counts += counts
        if counts[2] > 0:
            class2_present += 1
        forced_label_counts[str(chosen_label)] += 1
        rows.append({
            "case": key,
            "forced_label": int(chosen_label),
            "center_zyx": [int(v) for v in center],
            "label_counts": {str(i): int(counts[i]) for i in range(5)},
        })

    class2_counts = [int(row["label_counts"]["2"]) for row in rows]
    class1_counts = [int(row["label_counts"]["1"]) for row in rows]
    class3_counts = [int(row["label_counts"]["3"]) for row in rows]
    audit = {
        "dataset_id": int(ds_id),
        "name": DATASETS[ds_id]["name"],
        "profile": profile,
        "distribution": {str(k): float(v) for k, v in distribution},
        "cases": keys,
        "patch_size": [int(v) for v in patch],
        "n_patches": int(len(rows)),
        "class2_present_patch_ratio": float(class2_present / max(1, len(rows))),
        "class2_voxels_mean": float(np.mean(class2_counts)) if class2_counts else 0.0,
        "class2_voxels_median": float(np.median(class2_counts)) if class2_counts else 0.0,
        "class1_voxels_mean": float(np.mean(class1_counts)) if class1_counts else 0.0,
        "class3_voxels_mean": float(np.mean(class3_counts)) if class3_counts else 0.0,
        "forced_label_counts": dict(sorted(forced_label_counts.items())),
        "aggregate_label_counts": {str(i): int(aggregate_counts[i]) for i in range(5)},
        "rows": rows,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(audit, indent=2))
    return audit


CONTACT_INSTANCE_SIDECAR_DIR = "contact_instance_targets"
BICM_V5_INSTANCE_SIDECAR_DIR = "bicm_v5_instance_targets"


def _preprocessed_config_dir(ds_id: int, plans: str = "nnUNetPlans_3d_fullres") -> Path:
    return NN_PREP / DATASETS[ds_id]["name"] / plans


def _raw_bicm_v5_instance_roi(sample_id: str) -> tuple[np.ndarray, str]:
    """Return the original fragment instance ROI for one materialized V5 sample.

    Args:
        sample_id: nnU-Net sample key, for example `PENGWIN_003_LeftHip`.

    Returns:
        `(instance_roi, anatomy_name)` in the raw V5 sample geometry.

    Raises:
        FileNotFoundError: If the raw V5 sample or source PENGWIN GT label is
            missing.
        ValueError: If the sample name is not a V5 per-anatomy identifier.

    [AUDIT][Risk:High][Scope:target_traceability]
    V5 raw labels are semantic BICM labels (`0..4`), so the trainer cannot
    derive true cross-fragment edges from `_seg.npy` alone. This helper restores
    the source GT instance IDs by resampling the original case label into the
    exact raw ROI geometry already materialized by `build_bicm_v5_dataset`.
    That keeps Dataset537's public label contract unchanged while making the
    V38 edge-affinity target auditable and reproducible.
    """
    parts = str(sample_id).split("_")
    if len(parts) != 3 or parts[0] != "PENGWIN":
        raise ValueError(f"expected V5 sample id PENGWIN_<case>_<anatomy>, got {sample_id!r}")
    case_id = str(parts[1]).zfill(3)
    anatomy = str(parts[2])
    lo, hi = anatomy_range(anatomy)
    raw_ref_path = NN_RAW / DATASETS[537]["name"] / "labelsTr" / f"{sample_id}.mha"
    if not raw_ref_path.exists():
        raise FileNotFoundError(f"raw V5 label missing for sidecar generation: {raw_ref_path}")
    case_map = {cd.name.zfill(3): cd for cd in list_cases("pelvic")}
    if case_id not in case_map:
        raise FileNotFoundError(f"source pelvic case {case_id} missing under {DATA_RAW}")
    raw_ref = sitk.ReadImage(str(raw_ref_path))
    source_label = canonicalize_sitk(sitk.ReadImage(str(case_map[case_id] / "label.mha")))
    inst_img = sitk.Resample(
        source_label,
        raw_ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )
    inst = sitk.GetArrayFromImage(inst_img).astype(np.uint16, copy=False)
    inst = np.where((inst >= lo) & (inst <= hi), inst, 0).astype(np.uint16, copy=False)
    return inst, anatomy


def _resample_raw_seg_to_preprocessed(raw_seg: np.ndarray,
                                      target_shape: tuple[int, int, int],
                                      props: dict,
                                      plans_json: dict) -> np.ndarray:
    """Match a raw ROI segmentation to nnU-Net's `_seg.npy` geometry.

    [QC][Invariant:resampling_contract]
    The V5 sidecar must align with training crops loaded from preprocessed
    `_seg.npy`, not with the raw MHA. Use nnU-Net's own segmentation resampler
    and the same current/new spacing metadata recorded during preprocessing.
    A mismatch here would silently corrupt edge targets and invalidate every
    V38 metric.
    """
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape

    current_spacing = props.get("spacing")
    new_spacing = plans_json["configurations"]["3d_fullres"]["spacing"]
    if current_spacing is None or len(current_spacing) != 3:
        raise RuntimeError(f"invalid nnU-Net preprocessing spacing in properties: {current_spacing!r}")
    resampled = resample_data_or_seg_to_shape(
        raw_seg.astype(np.int16, copy=False)[None],
        tuple(int(v) for v in target_shape),
        current_spacing,
        new_spacing,
        is_seg=True,
        order=1,
        order_z=0,
        force_separate_z=None,
    )[0]
    return np.where((resampled >= 1) & (resampled <= 150), resampled, 0).astype(np.uint16, copy=False)


def _sample_locations(mask: np.ndarray,
                      max_locations: int,
                      seed: int) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)
    if coords.shape[0] > int(max_locations):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(coords.shape[0], size=int(max_locations), replace=False)
        coords = coords[idx]
    return coords.astype(np.int32, copy=False)


def _fast_same_anatomy_contact_mask(inst: np.ndarray) -> np.ndarray:
    """Fast sidecar contact sampler: adjacent different fragments, same anatomy.

    The trainer computes crop-level contact targets again. This full-volume map
    is only used to choose useful crop centers, so 6-neighbor fragment contact is
    enough and avoids the expensive morphology used by the legacy ABBC target.
    """
    inst = inst.astype(np.uint16, copy=False)
    out = np.zeros(inst.shape, dtype=bool)
    for axis in range(3):
        a_sl = [slice(None), slice(None), slice(None)]
        b_sl = [slice(None), slice(None), slice(None)]
        a_sl[axis] = slice(1, None)
        b_sl[axis] = slice(None, -1)
        a = inst[tuple(a_sl)]
        b = inst[tuple(b_sl)]
        same_anatomy = (
            (a > 0) & (b > 0) & (a != b)
            & (((a.astype(np.int32) - 1) // 50) == ((b.astype(np.int32) - 1) // 50))
        )
        if same_anatomy.any():
            out_a = out[tuple(a_sl)]
            out_b = out[tuple(b_sl)]
            out_a[same_anatomy] = True
            out_b[same_anatomy] = True
            out[tuple(a_sl)] = out_a
            out[tuple(b_sl)] = out_b
    return out


def _instance_centers_and_sizes(inst: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return centroid per fragment in preprocessed voxel coordinates."""
    centers = np.full((151, 3), np.nan, dtype=np.float32)
    sizes = np.zeros(151, dtype=np.int64)
    zero_center = []
    for fragment_id in [int(v) for v in np.unique(inst) if 1 <= int(v) <= 150]:
        coords = np.argwhere(inst == fragment_id)
        sizes[fragment_id] = int(coords.shape[0])
        if coords.shape[0] == 0:
            zero_center.append(fragment_id)
            continue
        centers[fragment_id] = coords.mean(axis=0).astype(np.float32)
    return centers, sizes, zero_center


def generate_contact_instance_sidecars(ds_id: int = 537,
                                       plans: str = "nnUNetPlans_3d_fullres",
                                       force: bool = False,
                                       out_json: Path | None = None,
                                       fail_on_error: bool = True) -> dict:
    """Generate compact sidecars from preprocessed instance labels.

    The sidecars intentionally store metadata and crop-sampling coordinates
    rather than full dense offset volumes. Dense support/core/contact/offset
    targets are deterministic functions of the resampled instance crop and are
    created in the trainer dataloader, which keeps Dataset537 from growing by
    another very large target tree.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] not in {"contact_instance", "factorized_instance"}:
        raise ValueError(f"Dataset{ds_id} is not a contact/factorized instance dataset")
    config_dir = _preprocessed_config_dir(ds_id, plans=plans)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"preprocessed config missing: {config_dir}")
    sidecar_name = FACTOR_INSTANCE_SIDECAR_DIR if cfg["kind"] == "factorized_instance" else CONTACT_INSTANCE_SIDECAR_DIR
    sidecar_dir = config_dir / sidecar_name
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ok = True
    max_locations = int(os.environ.get("PENGWIN_INSTANCE_SIDECAR_MAX_LOCATIONS", "30000"))
    case_names = sorted({p.name.replace("_seg.npy", "") for p in config_dir.glob("*_seg.npy")})
    case_names.extend(
        p.stem for p in sorted(config_dir.glob("*.npz"))
        if p.stem not in set(case_names)
    )
    for case in case_names:
        seg_path = config_dir / f"{case}_seg.npy"
        npz_path = config_dir / f"{case}.npz"
        out_path = sidecar_dir / f"{case}.npz"
        if out_path.exists() and not force:
            try:
                data = np.load(out_path)
                rows.append(json.loads(str(data["audit_json"].item())))
                continue
            except Exception:
                pass
        if seg_path.exists():
            seg = np.load(seg_path, "r")[0].astype(np.uint16, copy=False)
        elif npz_path.exists():
            seg = np.load(npz_path)["seg"][0].astype(np.uint16, copy=False)
        else:
            raise FileNotFoundError(f"missing preprocessed seg for {case} in {config_dir}")
        support = seg > 0
        contact = _fast_same_anatomy_contact_mask(seg)
        from scipy import ndimage as ndi

        outside = ~support
        distance_to_support = ndi.distance_transform_edt(outside) if support.any() else np.zeros_like(support, dtype=float)
        hard_negative = outside & (distance_to_support <= 3.0)
        centers, sizes, zero_center = _instance_centers_and_sizes(seg)
        tiny_ids = [int(i) for i in np.where((sizes > 0) & (sizes < 1000))[0]]
        tiny_mask = np.isin(seg, tiny_ids) if tiny_ids else np.zeros_like(support, dtype=bool)
        cid_int = int(case.split("_")[-1])
        contact_locations = _sample_locations(contact, max_locations, seed=cid_int * 17 + 3)
        hard_locations = _sample_locations(hard_negative, max_locations, seed=cid_int * 17 + 5)
        tiny_locations = _sample_locations(tiny_mask, max_locations, seed=cid_int * 17 + 7)
        support_locations = _sample_locations(support, max_locations, seed=cid_int * 17 + 11)
        audit = {
            "case": case,
            "shape": [int(v) for v in seg.shape],
            "fragment_count": int(np.count_nonzero(sizes)),
            "support_voxels": int(support.sum()),
            "contact_voxels": int(contact.sum()),
            "hard_negative_voxels": int(hard_negative.sum()),
            "tiny_fragment_ids": tiny_ids,
            "zero_center_fragments": [int(v) for v in zero_center],
            "offset_finite_inside_support": not zero_center,
            "contact_fraction_of_support": float(contact.sum()) / max(1, int(support.sum())),
            "hard_negative_overlaps_contact": int((hard_negative & contact).sum()),
        }
        ok = bool(ok and not zero_center and audit["hard_negative_overlaps_contact"] == 0)
        np.savez_compressed(
            out_path,
            centers_zyx=centers.astype(np.float32),
            fragment_sizes=sizes.astype(np.int64),
            contact_locations=contact_locations,
            hard_negative_locations=hard_locations,
            tiny_locations=tiny_locations,
            support_locations=support_locations,
            audit_json=np.array(json.dumps(audit)),
        )
        rows.append(audit)
    audit_payload = {
        "dataset_id": int(ds_id),
        "name": cfg["name"],
        "plans": plans,
        "sidecar_dir": str(sidecar_dir),
        "sidecar_policy": (
            "compact metadata; dense crop targets generated by "
            + ("PengwinTrainerFactorizedInstanceV4" if cfg["kind"] == "factorized_instance" else "PengwinTrainerLegacyTopologyV1")
        ),
        "n_cases": len(rows),
        "ok": bool(ok and len(rows) > 0),
        "cases": rows,
        "summary": {
            "zero_center_case_count": int(sum(1 for r in rows if r["zero_center_fragments"])),
            "mean_contact_fraction_of_support": float(np.mean([r["contact_fraction_of_support"] for r in rows])) if rows else 0.0,
            "total_contact_voxels": int(sum(r["contact_voxels"] for r in rows)),
            "total_hard_negative_voxels": int(sum(r["hard_negative_voxels"] for r in rows)),
        },
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(audit_payload, indent=2))
    if fail_on_error and not audit_payload["ok"]:
        raise RuntimeError(f"contact-instance sidecar audit failed for Dataset{ds_id}")
    return audit_payload


def generate_bicm_v5_instance_sidecars(ds_id: int = 537,
                                       plans: str = "nnUNetPlans_3d_fullres",
                                       force: bool = False,
                                       out_json: Path | None = None,
                                       fail_on_error: bool = True) -> dict:
    """Generate V5 sidecars with instance IDs aligned to preprocessed crops.

    The active V5 dataset stores semantic labels because its public nnU-Net
    contract is a simple BICM target. V38 needs an additional relational target:
    whether neighboring voxels belong to different fragments in the same ROI.
    This function builds that supervision as a sidecar from the original GT
    labels without changing `labelsTr` or the V5 evaluator.

    [DATA][Leakage]
    These sidecars are generated only from training GT labels for the locked
    overfit/root-cause dataset. They must not be generated for test/inference
    cases, and they are not model inputs.

    [QA][Gate:v38_before_training]
    Training with edge-affinity supervision is blocked unless this audit reports
    support alignment with semantic `_seg.npy`, nonzero contact/edge positives
    for the expected contact-positive ROIs, and zero invalid instance IDs.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "bicm_v5":
        raise ValueError(f"Dataset{ds_id} is not a BICM V5 dataset")
    config_dir = _preprocessed_config_dir(ds_id, plans=plans)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"preprocessed config missing: {config_dir}")
    plans_path = NN_PREP / cfg["name"] / "nnUNetResEncUNetLPlans.json"
    if not plans_path.exists():
        raise FileNotFoundError(f"plans JSON missing: {plans_path}")
    plans_json = json.loads(plans_path.read_text())
    sidecar_dir = config_dir / BICM_V5_INSTANCE_SIDECAR_DIR
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    case_names = sorted({p.name.replace("_seg.npy", "") for p in config_dir.glob("*_seg.npy")})
    rows: list[dict] = []
    ok = bool(case_names)
    max_locations = int(os.environ.get("PENGWIN_BICM_V5_SIDECAR_MAX_LOCATIONS", "30000"))
    for sample_id in case_names:
        seg_path = config_dir / f"{sample_id}_seg.npy"
        pkl_path = config_dir / f"{sample_id}.pkl"
        out_path = sidecar_dir / f"{sample_id}.npz"
        if out_path.exists() and not force:
            try:
                data = np.load(out_path)
                rows.append(json.loads(str(data["audit_json"].item())))
                continue
            except Exception:
                pass
        if not seg_path.exists() or not pkl_path.exists():
            raise FileNotFoundError(f"missing preprocessed seg/properties for {sample_id}")
        import pickle

        semantic = np.load(seg_path, "r")[0].astype(np.uint8, copy=False)
        with open(pkl_path, "rb") as f:
            props = pickle.load(f)
        raw_inst, anatomy = _raw_bicm_v5_instance_roi(sample_id)
        inst = _resample_raw_seg_to_preprocessed(
            raw_inst,
            target_shape=tuple(int(v) for v in semantic.shape),
            props=props,
            plans_json=plans_json,
        )
        support_from_instance = inst > 0
        support_from_semantic = np.isin(
            semantic,
            [
                V5_LABELS["interior_shell"],
                V5_LABELS["core"],
                V5_LABELS["contact_surface"],
            ],
        )
        support_mismatch = int(np.count_nonzero(support_from_instance != support_from_semantic))
        support_mismatch_fraction = float(support_mismatch) / max(1, int(support_from_semantic.sum()))
        contact = _fast_same_anatomy_contact_mask(inst)
        edge_break, edge_valid = _instance_edge_break_target(inst)
        centers, sizes, zero_center = _instance_centers_and_sizes(inst)
        tiny_ids = [int(i) for i in np.where((sizes > 0) & (sizes < 1000))[0]]
        tiny_mask = np.isin(inst, tiny_ids) if tiny_ids else np.zeros_like(support_from_instance, dtype=bool)
        from scipy import ndimage as ndi

        hard_negative = (~support_from_instance) & (
            ndi.distance_transform_edt(~support_from_instance) <= 3.0
        ) if support_from_instance.any() else np.zeros_like(support_from_instance, dtype=bool)
        seed_base = sum(ord(ch) for ch in sample_id)
        contact_locations = _sample_locations(contact, max_locations, seed=seed_base + 3)
        hard_locations = _sample_locations(hard_negative, max_locations, seed=seed_base + 5)
        tiny_locations = _sample_locations(tiny_mask, max_locations, seed=seed_base + 7)
        support_locations = _sample_locations(support_from_instance, max_locations, seed=seed_base + 11)
        edge_locations = _sample_locations(edge_break.any(axis=0), max_locations, seed=seed_base + 13)
        invalid_ids = sorted(int(v) for v in np.unique(inst) if int(v) < 0 or int(v) > 150)
        row = {
            "sample": sample_id,
            "anatomy": anatomy,
            "shape": [int(v) for v in inst.shape],
            "fragment_count": int(np.count_nonzero(sizes)),
            "instance_support_voxels": int(support_from_instance.sum()),
            "semantic_support_voxels": int(support_from_semantic.sum()),
            "support_mismatch_voxels": support_mismatch,
            "support_mismatch_fraction": support_mismatch_fraction,
            "contact_voxels": int(contact.sum()),
            "edge_break_voxels": int((edge_break > 0.5).sum()),
            "edge_valid_voxels": int((edge_valid > 0.5).sum()),
            "edge_break_fraction_valid": float((edge_break > 0.5).sum()) / max(1, int((edge_valid > 0.5).sum())),
            "contact_fraction_support": float(contact.sum()) / max(1, int(support_from_instance.sum())),
            "tiny_fragment_ids": tiny_ids,
            "zero_center_fragments": [int(v) for v in zero_center],
            "invalid_instance_ids": invalid_ids,
        }
        # [QC][Invariant:alignment]
        # Support mismatch is expected to be near zero, but not exactly zero:
        # semantic BICM labels and restored instance IDs are resampled through
        # separate nearest-neighbor label maps. Tiny boundary differences are
        # acceptable for edge-affinity QA; larger drift would mean V38 is
        # supervising topology on a different foreground than the support head.
        # The 0.5% cap is intentionally stricter than a metric tolerance because
        # this is a target-alignment guard, not a model-performance threshold.
        ok = bool(
            ok
            and support_mismatch_fraction <= 0.005
            and not invalid_ids
            and not zero_center
            and int(support_from_instance.sum()) > 0
        )
        np.savez_compressed(
            out_path,
            instance=inst.astype(np.uint16, copy=False),
            centers_zyx=centers.astype(np.float32),
            fragment_sizes=sizes.astype(np.int64),
            contact_locations=contact_locations,
            edge_locations=edge_locations,
            hard_negative_locations=hard_locations,
            tiny_locations=tiny_locations,
            support_locations=support_locations,
            audit_json=np.array(json.dumps(row)),
        )
        rows.append(row)
    contact_positive = [r for r in rows if int(r["contact_voxels"]) > 0]
    audit_payload = {
        "dataset_id": int(ds_id),
        "name": cfg["name"],
        "plans": plans,
        "sidecar_dir": str(sidecar_dir),
        "sidecar_policy": (
            "V5 semantic labels stay public; sidecars store resampled original "
            "instance IDs and edge/contact sampling metadata for V38 root-cause training."
        ),
        "n_samples": len(rows),
        "ok": bool(ok and rows),
        "summary": {
            "support_mismatch_sample_count": int(sum(1 for r in rows if r["support_mismatch_voxels"])),
            "max_support_mismatch_fraction": float(max([r["support_mismatch_fraction"] for r in rows], default=0.0)),
            "zero_center_sample_count": int(sum(1 for r in rows if r["zero_center_fragments"])),
            "contact_positive_sample_count": int(len(contact_positive)),
            "total_edge_break_voxels": int(sum(r["edge_break_voxels"] for r in rows)),
            "mean_edge_break_fraction_valid": float(np.mean([r["edge_break_fraction_valid"] for r in rows])) if rows else 0.0,
        },
        "samples": rows,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(audit_payload, indent=2))
    if fail_on_error and not audit_payload["ok"]:
        raise RuntimeError(f"BICM V5 instance sidecar audit failed for Dataset{ds_id}")
    return audit_payload


# =============================================================================
# Unpack (.npz → .npy after nnUNet preprocess)
# =============================================================================
def unpack_dataset(ds_id: int, plans: str = "nnUNetPlans_3d_fullres",
                   num_processes: int = 8):
    """Convert nnUNet .npz to .npy for fast training loading."""
    from nnunetv2.training.dataloading.utils import unpack_dataset as _unpack
    name = DATASETS[ds_id]["name"]
    folder = NN_PREP / name / plans
    if not folder.is_dir():
        print(f"  ⚠ skip Ds{ds_id} — {folder} does not exist")
        return
    print(f"[unpack] {folder}")
    t0 = time.time()
    _unpack(str(folder), unpack_segmentation=True, overwrite_existing=False,
            num_processes=num_processes)
    n_npy = len(list(folder.glob("*.npy")))
    print(f"  done in {time.time() - t0:.1f}s. .npy = {n_npy}")


# =============================================================================
# Statistics (analyze_dataset 통합)
# =============================================================================
def case_stats(case_dir: Path):
    img_path = case_dir / "image.mha"
    lbl_path = case_dir / "label.mha"
    if not (img_path.exists() and lbl_path.exists()):
        return None
    img = sitk.ReadImage(str(img_path))
    lbl = sitk.ReadImage(str(lbl_path))
    arr = sitk.GetArrayFromImage(lbl)
    img_arr = sitk.GetArrayFromImage(img)
    fg_ids = sorted(int(v) for v in np.unique(arr) if v > 0)
    per_anat = {}
    for name, lo, hi in ANATOMY_RANGES:
        ids = [v for v in fg_ids if lo <= v <= hi]
        vox = int(((arr >= lo) & (arr <= hi)).sum())
        per_anat[name] = {"n_fragments": len(ids), "voxels": vox, "ids": ids}
    cid = int(case_dir.name)
    return {
        "case_id": case_dir.name, "case_int": cid,
        "subject_type": case_subject_type(cid),
        "n_fragments": len(fg_ids),
        "spacing_xyz": list(img.GetSpacing()),
        "size_xyz": list(img.GetSize()),
        "intensity": {"min": float(img_arr.min()), "max": float(img_arr.max()),
                      "mean": float(img_arr.mean()), "std": float(img_arr.std())},
        "per_anatomy": per_anat,
        "fg_voxels": int((arr > 0).sum()),
    }


def aggregate_stats(rows: list) -> dict:
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["subject_type"]].append(r)
    summary = {}
    for stype, recs in by_type.items():
        n_frags = [r["n_fragments"] for r in recs]
        anat_hist = Counter()
        anat_voxels = defaultdict(list)
        for r in recs:
            for k, v in r["per_anatomy"].items():
                if v["n_fragments"] > 0:
                    anat_hist[k] += 1
                    anat_voxels[k].append(v["voxels"])
        spacings = np.array([r["spacing_xyz"] for r in recs])
        sizes = np.array([r["size_xyz"] for r in recs])
        summary[stype] = {
            "n_cases": len(recs),
            "fragments": {"min": int(np.min(n_frags)), "max": int(np.max(n_frags)),
                          "mean": float(np.mean(n_frags)), "median": float(np.median(n_frags)),
                          "histogram": dict(Counter(n_frags).most_common())},
            "anatomy_presence": {k: f"{anat_hist[k]}/{len(recs)}" for k in ANATOMY_NAMES},
            "anatomy_voxels_mean": {k: int(np.mean(v)) if v else 0
                                       for k, v in anat_voxels.items()},
            "spacing_xyz_median": [float(x) for x in np.median(spacings, axis=0)],
            "size_xyz_median": [int(x) for x in np.median(sizes, axis=0)],
        }
    return summary


def run_stats(out_path: Path):
    """Scan all cases, compute per-case stats, aggregate to summary, save JSON.

    Catches only I/O / ITK-image errors (FileNotFoundError, OSError, RuntimeError)
    so that programming bugs (KeyError, AttributeError, TypeError) still propagate.
    """
    cases = list_cases()
    print(f"scanning {len(cases)} cases ...")
    rows = []
    for i, cd in enumerate(cases):
        try:
            r = case_stats(cd)
            if r:
                rows.append(r)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(cases)}")
        except (FileNotFoundError, OSError, RuntimeError) as e:
            # I/O / ITK errors only — let logic bugs (KeyError, etc.) crash loud.
            print(f"  ERR {cd.name}: {e}")
    summary = aggregate_stats(rows)
    out = {"per_case": rows, "summary": summary}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n✅ saved {out_path}")
    for stype, s in summary.items():
        print(f"\n=== {stype} (n={s['n_cases']}) ===")
        print(f"  fragments  : min={s['fragments']['min']}  med={s['fragments']['median']}  max={s['fragments']['max']}")
        print(f"  anatomy_presence: {s['anatomy_presence']}")


# =============================================================================
# Sanity case inspection (absorbed from data/preprocess/inspect_cases.py)
# =============================================================================
def inspect_sanity_cases(out_text: Path) -> None:
    """Inspect sanity cases and write a compact text table."""
    sanity = Path("/workspace/nnunet/raw/Dataset999_PENGWIN_Sanity")
    out_text.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for img_path in sorted(sanity.glob("imagesTs/PENGWIN_*.mha")):
        cid = img_path.stem.replace("PENGWIN_", "").replace("_0000", "")
        lbl_path = sanity / "labelsTs" / f"PENGWIN_{cid}.mha"
        if not lbl_path.exists():
            continue
        img = sitk.ReadImage(str(img_path))
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_path)))
        uniq = np.unique(lbl)
        region = {"Sacrum": 0, "L Hip": 0, "R Hip": 0, "Femur": 0}
        for v in uniq:
            if v == 0: continue
            if 1 <= v <= 50: region["Sacrum"] += 1
            elif 51 <= v <= 100: region["L Hip"] += 1
            elif 101 <= v <= 150: region["R Hip"] += 1
            elif 151 <= v <= 200: region["Femur"] += 1
        rows.append({
            "case": cid,
            "size": list(img.GetSize()),
            "spacing": [round(s, 3) for s in img.GetSpacing()],
            "n_fragments": len(uniq) - 1,
            "fragments_by_region": region,
            "label_values": [int(v) for v in uniq if v != 0],
        })
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    with open(out_text, "w") as f:
        f.write("| Case | Region | Size (X×Y×Z) | Spacing (mm) | # frag | SA/LH/RH/FE | Labels |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            sa = r["fragments_by_region"]
            region = "Femur" if sa["Femur"] > 0 and sa["Sacrum"] == 0 else "Pelvic"
            f.write(f"| {r['case']} | {region} | {r['size'][0]}×{r['size'][1]}×{r['size'][2]} | "
                    f"{r['spacing']} | {r['n_fragments']} | "
                    f"{sa['Sacrum']}/{sa['L Hip']}/{sa['R Hip']}/{sa['Femur']} | {r['label_values']} |\n")
    print(f"Saved: {out_text}")


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="convert raw → nnUNet raw")
    p_build.add_argument("--dataset", default="all",
                         help="active dataset id (532/533/537) or 'all'")
    p_build.add_argument("--force", action="store_true",
                         help="rewrite existing imagesTr/labelsTr files")
    p_build.add_argument("--abbc-official", action="store_true",
                         help="legacy compatibility flag; ignored by active V5 builder")
    p_build.add_argument("--v3-input", choices=V3_INPUT_VARIANTS, default="ct_hu",
                         help="legacy compatibility flag; ignored by active V5 builder")
    p_build.add_argument("--v4-input", choices=V4_INPUT_VARIANTS, default="ct_lut",
                         help="legacy compatibility flag; ignored by active V5 builder")
    p_build.add_argument("--v5-input", choices=V5_INPUT_VARIANTS, default="ct_lut",
                         help="Dataset537 V5 per-anatomy BICM input contract")
    p_build.add_argument("--v5-target-profile", choices=V5_TARGET_PROFILES, default="v5_tiny_marker",
                         help="Dataset537 V5 target profile; v5_core_body is the connected-core V5.2 ablation")
    p_build.add_argument("--v5-core-ball-radius-mm", type=float, default=2.5,
                         help="physical core radius for --v5-target-profile v5_core_ball")
    p_build.add_argument("--v5-core-body-mm", type=float, default=3.0,
                         help="minimum interior distance for --v5-target-profile v5_core_body")
    p_build.add_argument("--v5-contact-band-mm", type=float, default=2.0,
                         help="support-limited contact band radius for --v5-target-profile v5_core_body_contact_band")
    p_build.add_argument("--case-subset", nargs="*", default=None,
                         help="optional zero-padded case IDs for root-cause mini datasets")
    p_build.add_argument("--no-audit", action="store_true",
                         help="skip strict generated raw dataset audit")

    p_unpack = sub.add_parser("unpack", help=".npz → .npy after nnUNet preprocess")
    p_unpack.add_argument("--dataset", required=True, type=int)
    p_unpack.add_argument("--plans", default="nnUNetPlans_3d_fullres")

    p_stats = sub.add_parser("stats", help="dataset statistics")
    p_stats.add_argument("--out", default=str(RESULT_REPORT / "dataset_stats.json"))

    p_audit = sub.add_parser("audit-raw", help="audit generated nnUNet raw datasets")
    p_audit.add_argument("--dataset", default="all")
    p_audit.add_argument("--out", default=str(RESULT_REPORT / f"audit_dataset_rebuild_{RESULT_DATE}.json"))

    p_audit_orientation = sub.add_parser(
        "audit-orientation",
        help="audit source->raw LPS canonicalization contract for active datasets",
    )
    p_audit_orientation.add_argument("--dataset", default="all")
    p_audit_orientation.add_argument(
        "--out",
        default=str(RESULT_REPORT / f"orientation_contract_audit_{RESULT_DATE}.json"),
    )

    p_inspect = sub.add_parser("inspect-sanity",
                               help="inspect sanity case sizes/labels (absorbed from data/preprocess/inspect_cases.py)")
    p_inspect.add_argument("--out", default="/workspace/_scratch/sanity_inspect.txt")

    p_audit_pre = sub.add_parser("audit-preprocessed",
                                 help="audit planned/preprocessed nnUNet datasets")
    p_audit_pre.add_argument("--dataset", default="all")
    p_audit_pre.add_argument("--out", default=str(RESULT_REPORT / f"audit_preprocessed_{RESULT_DATE}.json"))
    p_audit_pre.add_argument("--ensure-splits", action="store_true",
                             help="create standard 5-fold splits_final.json if missing")
    p_audit_pre.add_argument("--allow-incomplete", action="store_true",
                             help="write audit JSON without raising on missing artifacts")

    p_audit_abbc = sub.add_parser(
        "audit-abbc-targets",
        help="audit official/contact ABBC target seed contract for active ABBC datasets",
    )
    p_audit_abbc.add_argument("--dataset", nargs="*", type=int, default=None,
                              help="official ABBC dataset ids; default all active ABBC datasets")
    p_audit_abbc.add_argument("--out", default=str(RESULT_REPORT / f"audit_abbc_targets_{RESULT_DATE}.json"))
    p_audit_abbc.add_argument("--allow-fail", action="store_true")

    p_audit_bfv3 = sub.add_parser(
        "audit-boundary-targets",
        help="audit Dataset537 BoundaryFragment V3 target invariants",
    )
    p_audit_bfv3.add_argument("--dataset", nargs="*", type=int, default=None,
                              help="BoundaryFragment dataset ids; default all active V3 datasets")
    p_audit_bfv3.add_argument("--out", default=str(RESULT_REPORT / f"audit_boundary_fragment_targets_{RESULT_DATE}.json"))
    p_audit_bfv3.add_argument("--allow-fail", action="store_true")

    p_audit_sampling = sub.add_parser(
        "audit-boundary-sampling",
        help="simulate Dataset537 BoundaryFragment V3 forced crop sampling",
    )
    p_audit_sampling.add_argument("--dataset", type=int, default=537)
    p_audit_sampling.add_argument("--profile", choices=sorted(BOUNDARY_SAMPLING_DISTRIBUTIONS),
                                  default="boundary_fragment_v3_ridge80")
    p_audit_sampling.add_argument("--batches", type=int, default=100)
    p_audit_sampling.add_argument("--batch-size", type=int, default=2)
    p_audit_sampling.add_argument("--cases", nargs="*", default=None)
    p_audit_sampling.add_argument("--seed", type=int, default=537031)
    p_audit_sampling.add_argument("--out", default=str(RESULT_REPORT / f"audit_boundary_sampling_{RESULT_DATE}.json"))

    p_sidecar = sub.add_parser(
        "build-instance-sidecars",
        help="generate compact Dataset537 Contact-Instance sidecars after nnU-Net preprocessing",
        aliases=["build-factorized-sidecars"],
    )
    p_sidecar.add_argument("--dataset", type=int, default=537)
    p_sidecar.add_argument("--plans", default="nnUNetPlans_3d_fullres")
    p_sidecar.add_argument("--force", action="store_true")
    p_sidecar.add_argument("--out", default=str(RESULT_REPORT / f"audit_factorized_instance_targets_{RESULT_DATE}.json"))
    p_sidecar.add_argument("--allow-fail", action="store_true")

    p_audit_inst = sub.add_parser(
        "audit-instance-targets",
        help="audit compact Contact-Instance sidecars and target invariants",
        aliases=["audit-factorized-targets"],
    )
    p_audit_inst.add_argument("--dataset", type=int, default=537)
    p_audit_inst.add_argument("--plans", default="nnUNetPlans_3d_fullres")
    p_audit_inst.add_argument("--out", default=str(RESULT_REPORT / f"audit_factorized_instance_targets_{RESULT_DATE}.json"))
    p_audit_inst.add_argument("--allow-fail", action="store_true")

    p_bicm_sidecar = sub.add_parser(
        "build-bicm-v5-sidecars",
        help="generate Dataset537 V5 instance/edge sidecars for V38 root-cause experiments",
        aliases=["audit-bicm-v5-sidecars"],
    )
    p_bicm_sidecar.add_argument("--dataset", type=int, default=537)
    p_bicm_sidecar.add_argument("--plans", default="nnUNetPlans_3d_fullres")
    p_bicm_sidecar.add_argument("--force", action="store_true")
    p_bicm_sidecar.add_argument("--out", default=str(RESULT_REPORT / f"audit_bicm_v5_instance_sidecars_{RESULT_DATE}.json"))
    p_bicm_sidecar.add_argument("--allow-fail", action="store_true")

    args = p.parse_args()

    if args.cmd == "build":
        if args.dataset == "all":
            for ds_id in sorted(DATASETS):
                build_dataset(
                    ds_id,
                    force=args.force,
                    abbc_official=args.abbc_official,
                    v3_input=args.v3_input,
                    v4_input=args.v4_input,
                    v5_input=args.v5_input,
                    v5_target_profile=args.v5_target_profile,
                    v5_core_ball_radius_mm=args.v5_core_ball_radius_mm,
                    v5_core_body_mm=args.v5_core_body_mm,
                    v5_contact_band_mm=args.v5_contact_band_mm,
                    case_subset=args.case_subset if ds_id == 537 else None,
                )
            if not args.no_audit:
                audit_nnunet_raw_datasets(sorted(DATASETS),
                                          RESULT_REPORT / f"audit_dataset_rebuild_{RESULT_DATE}.json")
        else:
            build_dataset(
                int(args.dataset),
                force=args.force,
                abbc_official=args.abbc_official,
                v3_input=args.v3_input,
                v4_input=args.v4_input,
                v5_input=args.v5_input,
                v5_target_profile=args.v5_target_profile,
                v5_core_ball_radius_mm=args.v5_core_ball_radius_mm,
                v5_core_body_mm=args.v5_core_body_mm,
                v5_contact_band_mm=args.v5_contact_band_mm,
                case_subset=args.case_subset if int(args.dataset) == 537 else None,
            )
            if not args.no_audit:
                audit_name = (
                    f"audit_dataset_rebuild_537_{args.v5_input}_{RESULT_DATE}.json"
                    if int(args.dataset) == 537 else
                    f"audit_dataset_rebuild_{RESULT_DATE}.json"
                )
                audit_nnunet_raw_datasets([int(args.dataset)],
                                          RESULT_REPORT / audit_name)
    elif args.cmd == "unpack":
        unpack_dataset(args.dataset, args.plans)
    elif args.cmd == "stats":
        run_stats(Path(args.out))
    elif args.cmd == "audit-raw":
        ds_ids = sorted(DATASETS) if args.dataset == "all" else [int(args.dataset)]
        audit = audit_nnunet_raw_datasets(ds_ids, Path(args.out))
        print(json.dumps(audit, indent=2))
    elif args.cmd == "audit-orientation":
        ds_ids = sorted(DATASETS) if args.dataset == "all" else [int(args.dataset)]
        audit = audit_orientation_contract(ds_ids, Path(args.out))
        print(json.dumps(audit, indent=2))
    elif args.cmd == "audit-preprocessed":
        ds_ids = sorted(DATASETS) if args.dataset == "all" else [int(args.dataset)]
        audit = audit_nnunet_preprocessed_datasets(
            ds_ids, Path(args.out),
            ensure_splits=args.ensure_splits,
            fail_on_missing=not args.allow_incomplete,
        )
        print(json.dumps(audit, indent=2))
    elif args.cmd == "audit-abbc-targets":
        audit = audit_abbc_targets(
            args.dataset,
            Path(args.out),
            fail_on_error=not args.allow_fail,
        )
        print(json.dumps(audit, indent=2))
    elif args.cmd == "audit-boundary-targets":
        audit = audit_boundary_fragment_targets(
            args.dataset,
            Path(args.out),
            fail_on_error=not args.allow_fail,
        )
        print(json.dumps(audit, indent=2))
    elif args.cmd == "audit-boundary-sampling":
        audit = audit_boundary_sampling(
            ds_id=args.dataset,
            profile=args.profile,
            batches=args.batches,
            batch_size=args.batch_size,
            cases=args.cases,
            out_json=Path(args.out),
            seed=args.seed,
        )
        print(json.dumps(audit, indent=2))
    elif args.cmd in {"build-instance-sidecars", "build-factorized-sidecars", "audit-instance-targets", "audit-factorized-targets"}:
        audit = generate_contact_instance_sidecars(
            ds_id=args.dataset,
            plans=args.plans,
            force=args.force if args.cmd in {"build-instance-sidecars", "build-factorized-sidecars"} else False,
            out_json=Path(args.out),
            fail_on_error=not args.allow_fail,
        )
        print(json.dumps(audit, indent=2))
    elif args.cmd in {"build-bicm-v5-sidecars", "audit-bicm-v5-sidecars"}:
        audit = generate_bicm_v5_instance_sidecars(
            ds_id=args.dataset,
            plans=args.plans,
            force=args.force if args.cmd == "build-bicm-v5-sidecars" else False,
            out_json=Path(args.out),
            fail_on_error=not args.allow_fail,
        )
        print(json.dumps(audit, indent=2))
    elif args.cmd == "inspect-sanity":
        inspect_sanity_cases(Path(args.out))


if __name__ == "__main__":
    main()

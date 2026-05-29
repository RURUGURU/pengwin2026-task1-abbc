#!/usr/bin/env python3
"""PENGWIN 2026 Task 1 result visualizer.

This module is intentionally result-artifact only:
  - reads completed GPU eval JSON files from code_task1/result
  - writes text/JSON tables and dashboards into code_task1/result
  - scans nnU-Net checkpoint files and writes a lightweight weight manifest

It does not create log files. If a command needs durable output, it should be a
JSON/text result artifact under code_task1/result, not a .log side channel.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import sys
sys.path.insert(0, str(Path(__file__).parent))
from core import (
    ANATOMY_NAMES, ANATOMY_RANGES, DATASETS, DATA_RAW, NN_PREP,
    NN_RES, RESULT, RESULT_REPORT, RESULT_VISUALIZE, RESULT_WEIGHT, SOTA,
    CONTACT_INSTANCE_CORE_CH, CONTACT_INSTANCE_EDGE_BREAK_CHS,
    CONTACT_INSTANCE_OUTPUT_CHANNELS, configure_nnunet_env,
)
from eval import (
    SOTA_CASE_SETS, _contact_instance_head_probs, _json_sanitize, binary_metrics,
    case_dataset_id, decode_contact_instance_prediction, fragment_matching_metrics,
    run_boundary_fragment_eval, write_eval_visualization,
)
from utils import (
    ORIENTATION_CONTRACT_VERSION, canonicalize_sitk, find_case_dir, inst_to_anat,
    orientation_code,
    BoundaryFragmentParams, BFV3_LABELS, contact_barrier_distance,
    compute_boundary_fragment_target, decode_boundary_fragment, instance_iouf,
    oracle_topology_diagnostics,
)

configure_nnunet_env()


CHECKPOINT_NAMES = {
    "checkpoint_best.pth": "best",
    "checkpoint_final.pth": "final",
    "checkpoint_latest.pth": "latest",
    "checkpoint_swa.pth": "swa",
}


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
        if np.isnan(f):
            return "n/a"
        return f"{f:.{digits}f}"
    except Exception:
        return str(v)


def _bar(value: float, width: int = 30) -> str:
    if value is None:
        return "|" + " " * width + "|"
    try:
        v = max(0.0, min(1.0, float(value)))
    except Exception:
        return "|" + " " * width + "|"
    n = int(round(v * width))
    return "|" + "#" * n + " " * (width - n) + "|"


def load_completed_eval(path: Path) -> dict | None:
    """Load one completed eval JSON, skipping legacy/non-eval JSONs."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    required = {"iou_f_mean", "iou_a_mean", "case_results"}
    if not required.issubset(data):
        return None
    return data


def scan_eval_jsons(result_dir: Path = RESULT) -> list[dict]:
    """Return completed eval records sorted by IoU-F descending.

    Recursive: scans `<experiment_tag>/eval_jsons/` subfolders too. Skips
    `archive_*` (frozen historic experiments).
    """
    rows = []
    paths = [p for p in result_dir.rglob("eval*.json")
             if not any(part.startswith("archive_") for part in p.parts)]
    for path in sorted(paths):
        data = load_completed_eval(path)
        if data is None:
            continue
        rows.append({
            "path": str(path),
            "file": path.name,
            "mode": data.get("mode"),
            "scope": data.get("case_scope", "custom"),
            "n_cases": data.get("n_cases"),
            "iou_a": data.get("iou_a_mean"),
            "iou_f": data.get("iou_f_mean"),
            "iou_f_highest": data.get("iou_f_highest_mean"),
            "hd95_f": data.get("hd95_f_mean"),
            "assd_f": data.get("assd_f_mean"),
            "unmatched": data.get("match_stats", {}).get("unmatched_gt_fragments"),
            "total_gt": data.get("match_stats", {}).get("total_gt_fragments"),
            "judgement": data.get("sota_gap_diagnosis", {}).get("judgement"),
            "primary_bottleneck": data.get("sota_gap_diagnosis", {}).get("primary_bottleneck"),
        })
    rows.sort(key=lambda r: float(r["iou_f"]) if r["iou_f"] is not None else -1.0,
              reverse=True)
    return rows


def scan_weight_manifest(results_root: Path = NN_RES,
                         derived_root: Path = RESULT_WEIGHT) -> list[dict]:
    """Scan checkpoint files without loading tensor payloads.

    Raw nnU-Net weights stay in nnunet/results; derived weights such as SWA
    outputs live under the active RESULT_WEIGHT directory. The manifest gives
    result-side traceability without loading tensor arrays, which would be slow
    and memory-heavy for 1GB+ checkpoints.
    """
    rows = []
    if not results_root.exists():
        raw_paths = []
    else:
        raw_paths = sorted(results_root.glob("Dataset*/*/fold_*/checkpoint*.pth"))
    for path in raw_paths:
        stat = path.stat()
        fold = path.parent.name
        run_dir = path.parent.parent
        dataset_dir = run_dir.parent
        rows.append({
            "dataset": dataset_dir.name,
            "run": run_dir.name,
            "fold": fold,
            "checkpoint": path.name,
            "kind": CHECKPOINT_NAMES.get(path.name, "other"),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime(stat.st_mtime)),
            "path": str(path),
        })
    for path in sorted(derived_root.glob("*.pth")) if derived_root.exists() else []:
        stat = path.stat()
        rows.append({
            "dataset": "derived",
            "run": "active/weights",
            "fold": "n/a",
            "checkpoint": path.name,
            "kind": "derived",
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime(stat.st_mtime)),
            "path": str(path),
        })
    return rows


def write_weight_manifest(rows: list[dict], stamp: str,
                          result_dir: Path = RESULT) -> dict:
    """Write checkpoint/weight manifest as JSON and text."""
    result_dir.mkdir(parents=True, exist_ok=True)
    out_json = result_dir / f"weights_manifest_{stamp}.json"
    out_text = result_dir / f"weights_manifest_{stamp}.txt"

    by_dataset: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for row in rows:
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1

    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(rows),
        "by_dataset": by_dataset,
        "by_kind": by_kind,
        "weights": rows,
    }
    out_json.write_text(json.dumps(_json_sanitize(payload), indent=2,
                                   allow_nan=False))

    lines = [
        f"# Weight Manifest {stamp}",
        "",
        f"- Checkpoints found: `{len(rows)}`.",
        "- Raw .pth files remain under `nnunet/results`; derived .pth files live under the active `RESULT_WEIGHT` directory.",
        "",
        "## By Dataset",
        "",
        "| Dataset | Checkpoints |",
        "|---|---:|",
    ]
    for dataset, count in sorted(by_dataset.items()):
        lines.append(f"| {dataset} | {count} |")
    lines.extend([
        "",
        "## Checkpoints",
        "",
        "| Dataset | Fold | Kind | Size MB | Modified UTC | Path |",
        "|---|---|---|---:|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['fold']} | {row['kind']} | "
            f"{row['size_mb']:.2f} | {row['modified_utc']} | `{row['path']}` |"
        )
    lines.append("")
    out_text.write_text("\n".join(lines))
    return {"json": str(out_json), "text": str(out_text), "count": len(rows)}


def write_result_summary(eval_rows: list[dict], stamp: str,
                         result_dir: Path = RESULT) -> dict:
    """Write a compact leaderboard-style summary for completed eval JSONs."""
    result_dir.mkdir(parents=True, exist_ok=True)
    out_json = result_dir / f"visual_summary_{stamp}.json"
    out_text = result_dir / f"visual_summary_{stamp}.txt"

    best = eval_rows[0] if eval_rows else None
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sota": SOTA,
        "best_by_iou_f": best,
        "evals": eval_rows,
    }
    out_json.write_text(json.dumps(_json_sanitize(payload), indent=2,
                                   allow_nan=False))

    lines = [
        f"# Result Summary {stamp}",
        "",
        f"- Completed eval JSONs: `{len(eval_rows)}`.",
        f"- SOTA IoU-F target: `{SOTA['iou_f']:.4f}`.",
    ]
    if best:
        lines.append(
            f"- Best current eval: `{best['file']}` IoU-F `{_fmt(best['iou_f'])}`, "
            f"gap `{_fmt(float(best['iou_f']) - SOTA['iou_f'])}`."
        )
    lines.extend([
        "",
        "## Eval Table",
        "",
        "| File | Mode | Scope | N | IoU-A | IoU-F | Bar | HD95 | ASSD | Unmatched | Judgement |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|",
    ])
    for row in eval_rows:
        unmatched = "n/a"
        if row.get("unmatched") is not None and row.get("total_gt") is not None:
            unmatched = f"{row['unmatched']}/{row['total_gt']}"
        lines.append(
            f"| {row['file']} | {row.get('mode')} | {row.get('scope')} | "
            f"{row.get('n_cases')} | {_fmt(row.get('iou_a'))} | {_fmt(row.get('iou_f'))} | "
            f"`{_bar(row.get('iou_f'))}` | {_fmt(row.get('hd95_f'), 2)} | "
            f"{_fmt(row.get('assd_f'), 2)} | {unmatched} | {row.get('judgement')} |"
        )
    lines.append("")
    out_text.write_text("\n".join(lines))
    return {"json": str(out_json), "text": str(out_text), "count": len(eval_rows)}


def _visual_suffix(eval_path: Path) -> str:
    """Stable dashboard suffix that cannot collide across eval variants."""
    return eval_path.stem.replace("eval_", "", 1)


def run_all(result_dir: Path = RESULT,
            artifact_dir: Path = RESULT_VISUALIZE) -> dict:
    """Generate current visual artifacts into the active V2 result folder."""
    stamp = time.strftime("%Y%m%d", time.gmtime())
    eval_rows = scan_eval_jsons(result_dir)
    summary = write_result_summary(eval_rows, stamp, artifact_dir)

    # Per-eval dashboards use eval.py's canonical SOTA visualization helper.
    dashboards = []
    for row in eval_rows:
        path = Path(row["path"])
        suffix = _visual_suffix(path)
        out_json = artifact_dir / f"visual_{suffix}_{stamp}.json"
        out_text = artifact_dir / f"visual_{suffix}_{stamp}.txt"
        write_eval_visualization(path, out_text=out_text, out_json=out_json)
        dashboards.append({"source": str(path), "json": str(out_json),
                           "text": str(out_text)})

    weights = write_weight_manifest(scan_weight_manifest(), stamp, artifact_dir)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "dashboards": dashboards,
        "weights": weights,
    }
    manifest_path = artifact_dir / f"visual_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), indent=2,
                                        allow_nan=False))
    return manifest


# =============================================================================
# Case volume visualization: GT now, prediction later
# =============================================================================
ANATOMY_COLORS = {
    "Sacrum": (0.95, 0.55, 0.12),
    "LeftHip": (0.10, 0.55, 0.95),
    "RightHip": (0.10, 0.78, 0.35),
    "Femur": (0.95, 0.20, 0.35),
}


def _read_mha_array(path: Path, *, canonical_lps: bool = True) -> tuple[np.ndarray, object]:
    """Read an MHA volume and default visual QC to canonical LPS orientation."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    if canonical_lps:
        img = canonicalize_sitk(img)
    return sitk.GetArrayFromImage(img), img


def _sitk_geometry_matches(a: object, b: object) -> bool:
    """Return true when two SimpleITK images share the same physical grid."""
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-5)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-4)
        and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-5)
    )


def _resample_label_to_reference(label_img: object, reference_img: object) -> object:
    """Nearest-neighbor resample for visual overlays on the source CT grid."""
    import SimpleITK as sitk

    return sitk.Resample(
        label_img,
        reference_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        label_img.GetPixelID(),
    )


def _ct_window(arr: np.ndarray, lo: float = -500.0, hi: float = 1500.0) -> np.ndarray:
    """Bone-friendly CT window normalized to [0, 1] for overlays."""
    x = np.clip(arr.astype(np.float32), lo, hi)
    return (x - lo) / max(1e-6, hi - lo)


def _label_rgb(label_proj: np.ndarray, mode: str) -> np.ndarray:
    """Map a projected label image to RGB colors."""
    rgb = np.zeros(label_proj.shape + (3,), dtype=np.float32)
    if mode == "anatomy":
        for idx, name in enumerate(ANATOMY_NAMES, start=1):
            rgb[label_proj == idx] = ANATOMY_COLORS[name]
        return rgb

    ids = [int(v) for v in np.unique(label_proj) if int(v) > 0]
    for label_id in ids:
        rng = np.random.default_rng(label_id * 2654435761 % (2**32))
        rgb[label_proj == label_id] = rng.uniform(0.20, 0.95, size=3)
    return rgb


def _project_label_for_mip(label: np.ndarray, axis: int, mode: str) -> np.ndarray:
    """Project labels for QC overlays without using label-id priority.

    For semantic anatomy, `label.max(axis=...)` is misleading because class 3
    would win over class 2 whenever both exist along the ray. Use the dominant
    non-background class along each projected pixel instead. Instance fragment
    MIPs keep the historical max-ID projection because the visual is a compact
    overview, not a per-pixel metric.
    """
    if mode != "anatomy":
        return label.max(axis=axis)
    class_ids = np.array(list(range(1, len(ANATOMY_NAMES) + 1)), dtype=np.int16)
    counts = np.stack([(label == int(class_id)).sum(axis=axis) for class_id in class_ids], axis=0)
    max_counts = counts.max(axis=0)
    projected = class_ids[counts.argmax(axis=0)].astype(np.int16, copy=False)
    projected[max_counts == 0] = 0
    return projected


def render_label_mips(image: np.ndarray, label: np.ndarray, out_path: Path,
                      *, title: str, mode: str = "anatomy") -> None:
    """Write 3-axis MIP overlay PNG for one GT or prediction volume.

    `mode="anatomy"` expects labels 0..4. `mode="fragment"` can receive raw
    PENGWIN instance IDs; colors are deterministic per ID so GT/pred IDs can be
    visually compared without assuming the numbers match semantically.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_norm = _ct_window(image)
    axes_spec = [(0, "axial"), (1, "coronal"), (2, "sagittal")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)
    for ax, (axis, name) in zip(axes, axes_spec):
        img_mip = img_norm.max(axis=axis)
        lbl_mip = _project_label_for_mip(label, axis, mode)
        gray = np.stack([img_mip] * 3, axis=-1)
        color = _label_rgb(lbl_mip.astype(np.int32), mode)
        mask = lbl_mip > 0
        overlay = gray.copy()
        overlay[mask] = 0.48 * gray[mask] + 0.52 * color[mask]
        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(f"{title} - {name}")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


ABBC_COLORS = {
    1: (0.95, 0.72, 0.15),  # boundary
    2: (0.18, 0.72, 0.95),  # core
    3: (0.95, 0.18, 0.18),  # border
}


def _class_rgb(label_proj: np.ndarray, colors: dict[int, tuple[float, float, float]]) -> np.ndarray:
    """Map a small class-label projection to fixed QC colors."""
    rgb = np.zeros(label_proj.shape + (3,), dtype=np.float32)
    for value, color in colors.items():
        rgb[label_proj == value] = color
    return rgb


def render_abbc_class_mips(image: np.ndarray, label: np.ndarray, out_path: Path,
                           *, title: str) -> None:
    """Render ABBC classes with fixed colors so GT/pred maps are comparable.

    ABBC pseudo dice excludes background and is logged as
    `[boundary_shell, core, contact_surface]`; class 3 is the sparse
    fracture/contact cue.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_norm = _ct_window(image)
    axes_spec = [(0, "axial"), (1, "coronal"), (2, "sagittal")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)
    for ax, (axis, name) in zip(axes, axes_spec):
        img_mip = img_norm.max(axis=axis)
        lbl_mip = label.max(axis=axis)
        gray = np.stack([img_mip] * 3, axis=-1)
        color = _class_rgb(lbl_mip.astype(np.uint8), ABBC_COLORS)
        mask = lbl_mip > 0
        overlay = gray.copy()
        overlay[mask] = 0.42 * gray[mask] + 0.58 * color[mask]
        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(f"{title} - {name}")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_binary_error_mips(image: np.ndarray, error_label: np.ndarray,
                             out_path: Path, *, title: str) -> None:
    """Render TP/FP/FN maps for one binary class.

    Label convention: 1=TP, 2=FP, 3=FN. This makes border diagnosis explicit:
    a low border dice can mean a harmless one-voxel shift, an absent target, or
    a real missed contact separator. The overlay separates those cases.
    """
    colors = {
        1: (0.20, 0.85, 0.25),  # TP green
        2: (0.95, 0.25, 0.20),  # FP red
        3: (0.20, 0.45, 0.95),  # FN blue
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_norm = _ct_window(image)
    axes_spec = [(0, "axial"), (1, "coronal"), (2, "sagittal")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)
    for ax, (axis, name) in zip(axes, axes_spec):
        img_mip = img_norm.max(axis=axis)
        err_mip = error_label.max(axis=axis)
        gray = np.stack([img_mip] * 3, axis=-1)
        color = _class_rgb(err_mip.astype(np.uint8), colors)
        mask = err_mip > 0
        overlay = gray.copy()
        overlay[mask] = 0.35 * gray[mask] + 0.65 * color[mask]
        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(f"{title} - {name}")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_obj(path: Path, verts_xyz: np.ndarray, faces: np.ndarray) -> None:
    """Write a simple Wavefront OBJ mesh for external 3D viewers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for x, y, z in verts_xyz:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def render_3d_surfaces(label_inst: np.ndarray, ref_img, out_png: Path,
                       mesh_dir: Path, *, title: str,
                       step_size: int = 2, max_faces_per_anatomy: int = 35_000,
                       labels_are_anatomy: bool = False) -> dict:
    """Render anatomy-colored 3D surfaces and export OBJ meshes.

    This keeps the 3D artifact independent from a notebook or GUI. The PNG is a
    quick QC view, while the OBJ files can be opened later in MeshLab, Blender,
    3D Slicer, or any standard mesh viewer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure

    spacing_xyz = tuple(float(v) for v in ref_img.GetSpacing())
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    anat = label_inst.astype(np.uint8, copy=False) if labels_are_anatomy else inst_to_anat(label_inst)
    fig = plt.figure(figsize=(9, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    meshes = {}
    mins = []
    maxs = []

    for idx, name in enumerate(ANATOMY_NAMES, start=1):
        mask = anat == idx
        if not mask.any():
            continue
        try:
            verts_zyx, faces, _normals, _values = measure.marching_cubes(
                mask.astype(np.uint8),
                level=0.5,
                spacing=spacing_zyx,
                step_size=max(1, int(step_size)),
            )
        except ValueError:
            continue
        verts_xyz = verts_zyx[:, [2, 1, 0]]
        obj_path = mesh_dir / f"{name}.obj"
        _write_obj(obj_path, verts_xyz, faces)

        plot_faces = faces
        if len(plot_faces) > max_faces_per_anatomy:
            stride = int(np.ceil(len(plot_faces) / max_faces_per_anatomy))
            plot_faces = plot_faces[::stride]
        poly = Poly3DCollection(
            verts_xyz[plot_faces],
            alpha=0.42,
            facecolor=ANATOMY_COLORS[name],
            edgecolor="none",
        )
        ax.add_collection3d(poly)
        mins.append(verts_xyz.min(axis=0))
        maxs.append(verts_xyz.max(axis=0))
        meshes[name] = {
            "obj": str(obj_path),
            "vertices": int(len(verts_xyz)),
            "faces": int(len(faces)),
        }

    if mins:
        mn = np.min(np.vstack(mins), axis=0)
        mx = np.max(np.vstack(maxs), axis=0)
        center = (mn + mx) / 2.0
        radius = max(float((mx - mn).max()) / 2.0, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(title)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.view_init(elev=22, azim=-62)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return meshes


def semantic_pred_to_anatomy(pred: np.ndarray, ds_id: int) -> np.ndarray:
    """Map split semantic predictions to the shared anatomy color convention."""
    pred = pred.astype(np.uint8, copy=False)
    out = np.zeros_like(pred, dtype=np.uint8)
    if ds_id == 532:
        mask = (pred >= 1) & (pred <= 3)
        out[mask] = pred[mask]
    elif ds_id == 533:
        out[pred == 1] = 4
    else:
        raise ValueError(f"unsupported split semantic dataset id: {ds_id}")
    return out


def _is_split_semantic_prediction(pred: np.ndarray, ds_id: int) -> bool:
    """Return true when `pred` uses split semantic labels rather than instances."""
    max_label = int(pred.max()) if pred.size else 0
    if ds_id == 532:
        return max_label <= 3
    if ds_id == 533:
        return max_label <= 1
    return False


def visualize_case(case_id: str, pred_path: Path | None = None,
                   out_dir: Path | None = None,
                   step_size: int = 2) -> dict:
    """Create GT and optional prediction visual artifacts for one case."""
    import SimpleITK as sitk

    cid = str(case_id).zfill(3)
    case_dir = find_case_dir(cid)
    if case_dir is None:
        raise FileNotFoundError(f"case {cid} not found under {DATA_RAW}")
    out_dir = out_dir or (RESULT_VISUALIZE / "case_visuals" / cid)
    out_dir.mkdir(parents=True, exist_ok=True)

    image, img_ref = _read_mha_array(case_dir / "image.mha")
    gt, gt_ref = _read_mha_array(case_dir / "label.mha")
    gt_resampled = False
    if not _sitk_geometry_matches(gt_ref, img_ref):
        gt_ref = _resample_label_to_reference(gt_ref, img_ref)
        import SimpleITK as sitk
        gt = sitk.GetArrayFromImage(gt_ref)
        gt_resampled = True
    gt = gt.astype(np.int16, copy=False)

    artifacts: dict[str, object] = {
        "case": cid,
        "case_dir": str(case_dir),
        "out_dir": str(out_dir),
        "orientation_contract": ORIENTATION_CONTRACT_VERSION,
        "source_orientation": orientation_code(sitk.ReadImage(str(case_dir / "image.mha"))),
        "visual_orientation": "LPS",
        "visual_geometry": {
            "size_xyz": list(img_ref.GetSize()),
            "spacing_xyz": [float(v) for v in img_ref.GetSpacing()],
            "origin_xyz": [float(v) for v in img_ref.GetOrigin()],
            "direction": [float(v) for v in img_ref.GetDirection()],
        },
        "gt": {},
        "pred": None,
    }

    gt_anat = inst_to_anat(gt)
    gt_anat_png = out_dir / f"PENGWIN_{cid}_gt_anatomy_mip.png"
    gt_frag_png = out_dir / f"PENGWIN_{cid}_gt_fragments_mip.png"
    gt_3d_png = out_dir / f"PENGWIN_{cid}_gt_3d.png"
    render_label_mips(image, gt_anat, gt_anat_png, title=f"{cid} GT anatomy", mode="anatomy")
    render_label_mips(image, gt, gt_frag_png, title=f"{cid} GT fragments", mode="fragment")
    gt_meshes = render_3d_surfaces(
        gt, gt_ref, gt_3d_png, out_dir / "gt_meshes",
        title=f"PENGWIN {cid} GT 3D", step_size=step_size)
    artifacts["gt"] = {
        "anatomy_mip_png": str(gt_anat_png),
        "fragment_mip_png": str(gt_frag_png),
        "fragment_mip_note": None,
        "geometry_resampled_to_ct": gt_resampled,
        "surface_png": str(gt_3d_png),
        "meshes": gt_meshes,
    }

    if pred_path is not None:
        pred, pred_ref = _read_mha_array(pred_path)
        pred_resampled = False
        if not _sitk_geometry_matches(pred_ref, img_ref):
            pred_ref = _resample_label_to_reference(pred_ref, img_ref)
            pred = sitk.GetArrayFromImage(pred_ref)
            pred_resampled = True
        pred = pred.astype(np.int16, copy=False)
        pred_ds = case_dataset_id(cid)
        pred_is_semantic = _is_split_semantic_prediction(pred, pred_ds)
        pred_anat = semantic_pred_to_anatomy(pred, pred_ds) if pred_is_semantic else inst_to_anat(pred)
        pred_anat_png = out_dir / f"PENGWIN_{cid}_pred_anatomy_mip.png"
        pred_frag_png = out_dir / f"PENGWIN_{cid}_pred_fragments_mip.png"
        pred_3d_png = out_dir / f"PENGWIN_{cid}_pred_3d.png"
        render_label_mips(image, pred_anat, pred_anat_png, title=f"{cid} Pred anatomy", mode="anatomy")
        # Split anatomy datasets output semantic labels, not fragment instance IDs.
        # Rendering those semantic values with the fragment palette makes, for
        # example, class-2 LeftHip look like GT fragment ID 2. Keep the legacy
        # filename for downstream manifests, but color semantic predictions by
        # anatomy so GT/pred visual comparisons do not mix label contracts.
        pred_fragment_note = None
        if pred_is_semantic:
            gt_instance_frag_png = out_dir / f"PENGWIN_{cid}_gt_instance_fragments_mip.png"
            render_label_mips(
                image,
                gt,
                gt_instance_frag_png,
                title=f"{cid} GT instance fragments",
                mode="fragment",
            )
            render_label_mips(
                image,
                gt_anat,
                gt_frag_png,
                title=f"{cid} GT semantic anatomy",
                mode="anatomy",
            )
            artifacts["gt"]["instance_fragment_mip_png"] = str(gt_instance_frag_png)
            artifacts["gt"]["fragment_mip_note"] = (
                "Prediction is split semantic anatomy; this legacy GT fragment "
                "PNG is rendered with anatomy colors for direct comparison. "
                "Raw GT instance colors are preserved in gt_instance_fragments_mip.png."
            )
            render_label_mips(
                image,
                pred_anat,
                pred_frag_png,
                title=f"{cid} Pred semantic anatomy",
                mode="anatomy",
            )
            pred_fragment_note = (
                "Prediction is split semantic anatomy, not fragment IDs; "
                "this legacy fragment PNG is rendered with anatomy colors."
            )
        else:
            render_label_mips(image, pred, pred_frag_png, title=f"{cid} Pred fragments", mode="fragment")
        pred_meshes = render_3d_surfaces(
            pred_anat, pred_ref, pred_3d_png, out_dir / "pred_meshes",
            title=f"PENGWIN {cid} Pred 3D", step_size=step_size,
            labels_are_anatomy=True)
        artifacts["pred"] = {
            "source": str(pred_path),
            "dataset_id": pred_ds,
            "prediction_contract": "split_semantic" if pred_is_semantic else "instance",
            "source_orientation": orientation_code(sitk.ReadImage(str(pred_path))),
            "visual_orientation": "LPS",
            "orientation_contract": ORIENTATION_CONTRACT_VERSION,
            "geometry_resampled_to_ct": pred_resampled,
            "anatomy_mip_png": str(pred_anat_png),
            "fragment_mip_png": str(pred_frag_png),
            "fragment_mip_note": pred_fragment_note,
            "surface_png": str(pred_3d_png),
            "meshes": pred_meshes,
        }

    manifest_path = out_dir / f"PENGWIN_{cid}_visual_manifest.json"
    manifest_path.write_text(json.dumps(_json_sanitize(artifacts), indent=2, allow_nan=False))
    artifacts["manifest"] = str(manifest_path)
    return artifacts


def fragment_stats(label_inst: np.ndarray) -> dict:
    """Summarize GT/pred fragment IDs for visual case review."""
    rows = []
    for label_id in sorted(int(v) for v in np.unique(label_inst) if int(v) > 0):
        mask = label_inst == label_id
        coords = np.argwhere(mask)
        anatomy = "Unknown"
        for name, lo, hi in ANATOMY_RANGES:
            if lo <= label_id <= hi:
                anatomy = name
                break
        bbox = None
        if len(coords):
            mn = coords.min(axis=0)
            mx = coords.max(axis=0) + 1
            bbox = {
                "z": [int(mn[0]), int(mx[0])],
                "y": [int(mn[1]), int(mx[1])],
                "x": [int(mn[2]), int(mx[2])],
            }
        rows.append({
            "label_id": label_id,
            "anatomy": anatomy,
            "voxels": int(mask.sum()),
            "bbox_zyx": bbox,
        })
    by_anatomy = {name: 0 for name in ANATOMY_NAMES}
    for row in rows:
        if row["anatomy"] in by_anatomy:
            by_anatomy[row["anatomy"]] += 1
    return {
        "n_fragments": len(rows),
        "fragments_by_anatomy": by_anatomy,
        "fragments": rows,
    }


def _dice_binary(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Dice for sparse binary masks, treating absent/absent as perfect."""
    denom = int(gt_mask.sum() + pred_mask.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * int((gt_mask & pred_mask).sum()) / denom)


def _select_border_diagnosis_cases(ds_id: int, fold: int) -> list[str]:
    """Pick zero-, mid-, and high-border validation cases for QC.

    This intentionally selects by target distribution, not model score. The
    question is whether low `border` pseudo dice reflects true contact failures
    or the fact that many validation cases have almost no border voxels.
    """
    split_path = NN_PREP / DATASETS[ds_id]["name"] / "splits_final.json"
    splits = json.loads(split_path.read_text())
    rows = []
    for token in splits[int(fold)]["val"]:
        cid = str(token).split("_")[-1].zfill(3)
        seg_path = NN_PREP / DATASETS[ds_id]["name"] / "nnUNetPlans_3d_fullres" / f"PENGWIN_{cid}.npz"
        if not seg_path.exists():
            continue
        seg = np.load(seg_path)["seg"][0].astype(np.int16, copy=False)
        valid = seg[seg >= 0]
        counts = np.bincount(valid.ravel(), minlength=4)[:4]
        fg = int(counts[1] + counts[2] + counts[3])
        frac = float(counts[2] / fg) if fg else 0.0
        rows.append({"case": cid, "counts": counts.tolist(), "border_fg_fraction": frac})
    zero = [r["case"] for r in rows if r["counts"][2] == 0][:3]
    nonzero = [r for r in rows if r["counts"][2] > 0]
    nonzero.sort(key=lambda r: r["border_fg_fraction"])
    mid = []
    if nonzero:
        for idx in sorted({len(nonzero) // 3, len(nonzero) // 2, (2 * len(nonzero)) // 3}):
            mid.append(nonzero[min(idx, len(nonzero) - 1)]["case"])
    high = [r["case"] for r in nonzero[-3:]]
    ordered = []
    for cid in zero + mid + high:
        if cid not in ordered:
            ordered.append(cid)
    return ordered


def _read_case_image_and_label(cid: str) -> tuple[np.ndarray, object, np.ndarray, object, Path]:
    """Read source CT and GT label in canonical LPS for aligned visual QC."""
    case_dir = find_case_dir(cid)
    if case_dir is None:
        raise FileNotFoundError(f"case {cid} not found under {DATA_RAW}")
    image, img_ref = _read_mha_array(case_dir / "image.mha")
    label, label_ref = _read_mha_array(case_dir / "label.mha")
    return image, img_ref, label.astype(np.uint16, copy=False), label_ref, case_dir


def export_hard_view(case_set: str = "hard10",
                     out_dir: Path | None = None,
                     step_size: int = 2) -> dict:
    """Create an analysis-ready GT hard-case folder tree."""
    if case_set not in SOTA_CASE_SETS:
        raise ValueError(f"unknown case_set={case_set!r}; choose from {sorted(SOTA_CASE_SETS)}")
    out_dir = out_dir or (RESULT_VISUALIZE / "hard_view" / case_set)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case_set": case_set,
        "out_dir": str(out_dir),
        "cases": [],
    }
    for rank, cid in enumerate(SOTA_CASE_SETS[case_set], start=1):
        cid = str(cid).zfill(3)
        case_dir = find_case_dir(cid)
        if case_dir is None:
            raise FileNotFoundError(f"case {cid} not found under {DATA_RAW}")
        dst = out_dir / f"hard{rank:02d}_id_{cid}"
        image_dir = dst / "image"
        segment_dir = dst / "segment"
        visual_dir = dst / "visual"
        image_dir.mkdir(parents=True, exist_ok=True)
        segment_dir.mkdir(parents=True, exist_ok=True)
        visual_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(case_dir / "image.mha", image_dir / "image.mha")
        shutil.copy2(case_dir / "label.mha", segment_dir / "label.mha")

        image, _img_ref = _read_mha_array(case_dir / "image.mha")
        gt, gt_ref = _read_mha_array(case_dir / "label.mha")
        gt = gt.astype(np.int16, copy=False)
        gt_anat = inst_to_anat(gt)
        anat_png = visual_dir / "gt_anatomy_mip.png"
        frag_png = visual_dir / "gt_fragments_mip.png"
        surf_png = visual_dir / "gt_3d_surface.png"
        render_label_mips(image, gt_anat, anat_png, title=f"{cid} GT anatomy", mode="anatomy")
        render_label_mips(image, gt, frag_png, title=f"{cid} GT fragments", mode="fragment")
        meshes = render_3d_surfaces(
            gt, gt_ref, surf_png, visual_dir / "gt_meshes",
            title=f"PENGWIN {cid} GT 3D", step_size=step_size)
        stats = fragment_stats(gt)
        stats_path = visual_dir / "gt_fragment_stats.json"
        stats_path.write_text(json.dumps(_json_sanitize(stats), indent=2, allow_nan=False))
        row = {
            "rank": rank,
            "case": cid,
            "folder": str(dst),
            "image": str(image_dir / "image.mha"),
            "segment": str(segment_dir / "label.mha"),
            "visual": {
                "gt_anatomy_mip": str(anat_png),
                "gt_fragments_mip": str(frag_png),
                "gt_3d_surface": str(surf_png),
                "gt_meshes": meshes,
                "gt_fragment_stats": str(stats_path),
            },
            "stats": {
                "n_fragments": stats["n_fragments"],
                "fragments_by_anatomy": stats["fragments_by_anatomy"],
            },
        }
        manifest["cases"].append(row)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), indent=2, allow_nan=False))
    manifest["manifest"] = str(manifest_path)
    return manifest


ROOT_CAUSE_CASES = ["003", "011", "017", "018", "025", "159"]
BICM_ROI_ANATOMIES = ("Sacrum", "LeftHip", "RightHip")


def _ds537_case_name(cid: str) -> str:
    return f"PENGWIN_{str(cid).zfill(3)}"


def _load_ds537_preprocessed_case(cid: str,
                                  plans: str = "nnUNetPlans_3d_fullres") -> tuple[np.ndarray, np.ndarray, Path]:
    """Load one Dataset537 preprocessed data/instance label pair.

    Dataset537 labels are instance IDs. The root-cause probes intentionally use
    the preprocessed tensors so oracle/separability tests exercise the same
    spacing and input channels seen by the trainer.
    """
    config_dir = NN_PREP / DATASETS[537]["name"] / plans
    case_name = _ds537_case_name(cid)
    data_npy = config_dir / f"{case_name}.npy"
    seg_npy = config_dir / f"{case_name}_seg.npy"
    npz_path = config_dir / f"{case_name}.npz"
    if data_npy.exists() and seg_npy.exists():
        data = np.asarray(np.load(data_npy, mmap_mode="r"), dtype=np.float32)
        seg = np.asarray(np.load(seg_npy, mmap_mode="r")[0], dtype=np.uint16)
        return data, seg, config_dir
    if npz_path.exists():
        payload = np.load(npz_path)
        data = np.asarray(payload["data"], dtype=np.float32)
        seg = np.asarray(payload["seg"][0], dtype=np.uint16)
        return data, seg, config_dir
    raise FileNotFoundError(
        f"Dataset537 preprocessed case not found: {case_name} under {config_dir}. "
        "Run preprocessing.py build, nnUNetv2_plan_and_preprocess, and build-instance-sidecars first."
    )


def _load_ds537_bicm_preprocessed_sample(sample: str,
                                         plans: str = "nnUNetPlans_3d_fullres") -> tuple[np.ndarray, np.ndarray, Path]:
    """Load one per-anatomy Dataset537 BICM ROI sample.

    [AUDIT][Risk:High][Scope:root_cause_input_contract]
    The current Dataset537 materialization is no longer the old global
    instance-label dataset. It contains per-anatomy ROI samples such as
    `PENGWIN_003_LeftHip` with semantic BICM labels 0..4. Reusing the older
    global-contact audit would silently inspect the wrong sample contract.

    [QC][Invariant:shape_label_contract]
    Returned `data` is channel-first `[C,Z,Y,X]`; returned `seg` is `[Z,Y,X]`
    with labels 0..4. The audit fails early if either shape is inconsistent.
    """
    config_dir = NN_PREP / DATASETS[537]["name"] / plans
    data_npy = config_dir / f"{sample}.npy"
    seg_npy = config_dir / f"{sample}_seg.npy"
    npz_path = config_dir / f"{sample}.npz"
    if data_npy.exists() and seg_npy.exists():
        data = np.asarray(np.load(data_npy, mmap_mode="r"), dtype=np.float32)
        seg = np.asarray(np.load(seg_npy, mmap_mode="r")[0], dtype=np.uint8)
    elif npz_path.exists():
        payload = np.load(npz_path)
        data = np.asarray(payload["data"], dtype=np.float32)
        seg = np.asarray(payload["seg"][0], dtype=np.uint8)
    else:
        raise FileNotFoundError(f"Dataset537 BICM sample missing: {sample} under {config_dir}")
    if data.ndim != 4:
        raise ValueError(f"{sample}: expected data [C,Z,Y,X], got {data.shape}")
    if seg.ndim != 3 or tuple(seg.shape) != tuple(data.shape[1:]):
        raise ValueError(f"{sample}: expected seg shape {data.shape[1:]}, got {seg.shape}")
    if seg.size and (int(seg.min()) < 0 or int(seg.max()) > 4):
        raise ValueError(f"{sample}: BICM labels must be in 0..4, got min={int(seg.min())}, max={int(seg.max())}")
    return data, seg, config_dir


def _same_anatomy_contact_mask(inst: np.ndarray) -> np.ndarray:
    """6-neighbor same-anatomy, different-fragment contact voxels.

    This is intentionally stricter than "bone surface". A voxel is contact only
    when an immediate neighbor belongs to another fragment in the same pelvic
    anatomy range. The broad-contact failure came from treating ordinary
    external cortex as if it were fracture/contact; keeping this primitive
    narrow lets the root-cause audit distinguish real contact from surface-like
    false positives.
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
        hit = (
            (a > 0) & (b > 0) & (a != b)
            & (((a.astype(np.int32) - 1) // 50) == ((b.astype(np.int32) - 1) // 50))
        )
        if hit.any():
            oa = out[tuple(a_sl)]
            ob = out[tuple(b_sl)]
            oa[hit] = True
            ob[hit] = True
            out[tuple(a_sl)] = oa
            out[tuple(b_sl)] = ob
    return out


def _fragment_surface_mask(inst: np.ndarray) -> np.ndarray:
    """Support voxels touching non-self neighbors; used as hard near-negative.

    This includes external fragment surface and anatomy/background interfaces.
    It is deliberately not the positive target. Separability is only meaningful
    if true contact is compared against this confusing near-negative class,
    because that is where the previous contact head produced most false
    positives.
    """
    inst = inst.astype(np.uint16, copy=False)
    support = inst > 0
    out = np.zeros(inst.shape, dtype=bool)
    for axis in range(3):
        a_sl = [slice(None), slice(None), slice(None)]
        b_sl = [slice(None), slice(None), slice(None)]
        a_sl[axis] = slice(1, None)
        b_sl[axis] = slice(None, -1)
        a = inst[tuple(a_sl)]
        b = inst[tuple(b_sl)]
        diff = (a != b) & ((a > 0) | (b > 0))
        if diff.any():
            oa = out[tuple(a_sl)]
            ob = out[tuple(b_sl)]
            oa[diff & (a > 0)] = True
            ob[diff & (b > 0)] = True
            out[tuple(a_sl)] = oa
            out[tuple(b_sl)] = ob
    return out & support


def _ball(radius: int) -> np.ndarray:
    radius = int(max(0, radius))
    grid = np.indices((2 * radius + 1, 2 * radius + 1, 2 * radius + 1)) - radius
    return (np.sum(grid * grid, axis=0) <= radius * radius)


def _core_seed_mask(inst: np.ndarray) -> tuple[np.ndarray, dict]:
    """One compact center seed per fragment for oracle decoder tests."""
    seeds = np.zeros(inst.shape, dtype=bool)
    rows = []
    structure = _ball(1)
    for fid in [int(v) for v in np.unique(inst) if 1 <= int(v) <= 150]:
        mask = inst == fid
        coords = np.argwhere(mask)
        if coords.size == 0:
            rows.append({"fragment_id": fid, "voxels": 0, "seed_voxels": 0})
            continue
        dist = ndi.distance_transform_edt(mask)
        max_dist = float(dist.max())
        candidates = np.argwhere(dist == dist.max())
        centroid = coords.mean(axis=0)
        center = candidates[int(np.argmin(np.sum((candidates - centroid) ** 2, axis=1)))]
        single = np.zeros(inst.shape, dtype=bool)
        single[tuple(int(v) for v in center)] = True
        seed = ndi.binary_dilation(single, structure=structure) & mask
        if not seed.any():
            seed[tuple(int(v) for v in center)] = True
        seeds |= seed
        rows.append({
            "fragment_id": fid,
            "voxels": int(mask.sum()),
            "max_center_distance": max_dist,
            "seed_voxels": int(seed.sum()),
        })
    missing = [r["fragment_id"] for r in rows if r["voxels"] > 0 and r["seed_voxels"] == 0]
    return seeds, {"fragments": rows, "missing_core_seed_ids": missing}


def _edge_break_masks(inst: np.ndarray) -> list[np.ndarray]:
    """Per-axis same-anatomy cross-fragment edge masks for diagnostics."""
    out = [np.zeros(inst.shape, dtype=bool) for _ in range(3)]
    for axis in range(3):
        a_sl = [slice(None), slice(None), slice(None)]
        b_sl = [slice(None), slice(None), slice(None)]
        a_sl[axis] = slice(1, None)
        b_sl[axis] = slice(None, -1)
        a = inst[tuple(a_sl)]
        b = inst[tuple(b_sl)]
        hit = (
            (a > 0) & (b > 0) & (a != b)
            & (((a.astype(np.int32) - 1) // 50) == ((b.astype(np.int32) - 1) // 50))
        )
        if hit.any():
            oa = out[axis][tuple(a_sl)]
            ob = out[axis][tuple(b_sl)]
            oa[hit] = True
            ob[hit] = True
            out[axis][tuple(a_sl)] = oa
            out[axis][tuple(b_sl)] = ob
    return out


def _oracle_contact_instance_output(inst: np.ndarray,
                                    data: np.ndarray,
                                    target_variant: str) -> tuple[np.ndarray, dict]:
    """Build oracle logits for the current 19-channel Contact-Instance head.

    This does not simulate model quality. It answers a narrower blocking
    question: if the network produced perfect GT-derived support/core/contact
    signals in the current channel layout, could the decoder recover fragment
    topology? A failing oracle means training is pointless until the target or
    decoder contract is fixed.
    """
    support = inst > 0
    contact = _same_anatomy_contact_mask(inst)
    if target_variant == "edge_dilated_pair":
        contact_for_logit = ndi.binary_dilation(contact, structure=np.ones((3, 3, 3), dtype=bool)) & support
    else:
        contact_for_logit = contact
    distance = ndi.distance_transform_edt(~contact_for_logit)
    ridge = np.exp(-distance / 1.25).astype(np.float32)
    ridge *= support.astype(np.float32, copy=False)
    seeds, seed_info = _core_seed_mask(inst)

    out = np.full((CONTACT_INSTANCE_OUTPUT_CHANNELS,) + inst.shape, -8.0, dtype=np.float32)
    out[0, ~support] = 8.0
    out[1, support & ~contact_for_logit] = 8.0
    out[2, contact_for_logit] = 8.0
    # Keep dense contact energy in the logit without crossing the decoder wall
    # threshold far from the actual fracture/contact surface.
    if target_variant == "dense_contact_energy":
        out[2, support & ~contact_for_logit] = np.maximum(
            out[2, support & ~contact_for_logit],
            (-8.0 + 8.0 * ridge[support & ~contact_for_logit]).astype(np.float32),
        )
    if data.shape[0] >= 6:
        hard = (data[5] > 0.50) & ~support
        out[3, hard] = 8.0
    out[CONTACT_INSTANCE_CORE_CH, seeds] = 8.0
    for ch, edge in zip(CONTACT_INSTANCE_EDGE_BREAK_CHS, _edge_break_masks(inst)):
        out[ch, edge] = 8.0
    return out, {
        "support_voxels": int(support.sum()),
        "contact_voxels": int(contact.sum()),
        "contact_logit_wall_voxels": int(contact_for_logit.sum()),
        "dense_contact_ridge_sum": float(ridge[support].sum()) if support.any() else 0.0,
        "core_seed_info": seed_info,
    }


def _tiny_fragment_rows(inst: np.ndarray, matching: dict, max_voxels: int = 1000) -> list[dict]:
    best = {int(r["gt_id"]): r for r in matching["official_style"]["best_matches"]}
    rows = []
    for fid in [int(v) for v in np.unique(inst) if 1 <= int(v) <= 150]:
        voxels = int((inst == fid).sum())
        if 0 < voxels < int(max_voxels):
            row = best.get(fid, {"gt_id": fid, "pred_id": None, "iou": 0.0})
            rows.append({
                "fragment_id": fid,
                "voxels": voxels,
                "best_pred_id": row.get("pred_id"),
                "best_iou": float(row.get("iou", 0.0)),
            })
    return rows


def _run_contact_root_cause_oracle(cases: list[str],
                                   target_variant: str,
                                   plans: str) -> dict:
    rows = []
    for cid in cases:
        data, inst, _config_dir = _load_ds537_preprocessed_case(cid, plans=plans)
        foundation_oracle = inst_to_anat(inst).astype(np.uint8, copy=False)
        output, target_info = _oracle_contact_instance_output(inst, data, target_variant)
        decoded, trace = decode_contact_instance_prediction(
            foundation_oracle, output, profile=f"oracle_{target_variant}"
        )
        matching = fragment_matching_metrics(inst, decoded, [537])
        contact_target = _same_anatomy_contact_mask(inst)
        _support_p, _core_p, contact_p, _hard_p = _contact_instance_head_probs(output)
        contact_metrics = binary_metrics(contact_target, contact_p >= 0.45)
        tiny = _tiny_fragment_rows(inst, matching)
        decoder_missing = trace.get("missing_seed_component", [])
        row = {
            "case": str(cid).zfill(3),
            "target_variant": target_variant,
            "iou_f_oracle": matching["official_style"]["mean_best_iou"],
            "hungarian_iou": matching["hungarian"]["mean_iou_over_gt"],
            "gt_fragment_count": matching["gt_fragment_count"],
            "pred_fragment_count": matching["pred_fragment_count"],
            "unmatched_gt_ids": matching["official_style"]["unmatched_gt_ids"],
            "contact_metrics": contact_metrics,
            "target_info": target_info,
            "decoder_trace_summary": {
                "seed_components": trace.get("seed_components"),
                "support_voxels": trace.get("support_voxels"),
                "contact_wall_voxels": trace.get("contact_wall_voxels"),
                "missing_seed_component_count": len(decoder_missing),
                "missing_seed_components": decoder_missing[:20],
                "removed_count": len(trace.get("removed", [])),
            },
            "tiny_fragments": tiny,
        }
        rows.append(row)

    ious = [float(r["iou_f_oracle"]) for r in rows]
    decoder_missing_total = int(sum(r["decoder_trace_summary"]["missing_seed_component_count"] for r in rows))
    core_missing_total = int(sum(len(r["target_info"]["core_seed_info"]["missing_core_seed_ids"]) for r in rows))
    case003 = next((r for r in rows if r["case"] == "003"), None)
    case003_tiny_preserved = True
    if case003 is not None:
        # Case003 has known very small pelvic fragments. The oracle target is
        # considered topology-viable only if those fragments still receive a
        # meaningful best IoU after decoding.
        case003_tiny_preserved = all(float(r["best_iou"]) >= 0.50 for r in case003["tiny_fragments"])
    passed = bool(
        rows
        and float(np.mean(ious)) >= 0.95
        and min(ious) >= 0.90
        and decoder_missing_total == 0
        and core_missing_total == 0
        and case003_tiny_preserved
    )
    return {
        "gate": "oracle_target_decoder",
        "target_variant": target_variant,
        "passed": passed,
        "criteria": {
            "mean_oracle_iou_f_min": 0.95,
            "per_case_oracle_iou_f_floor": 0.90,
            "missing_seed_total_required": 0,
            "case003_tiny_fragment_iou_min": 0.50,
        },
        "summary": {
            "mean_oracle_iou_f": float(np.mean(ious)) if ious else 0.0,
            "min_oracle_iou_f": float(np.min(ious)) if ious else 0.0,
            "decoder_missing_seed_component_total": decoder_missing_total,
            "core_seed_missing_fragment_total": core_missing_total,
            "case003_tiny_preserved": bool(case003_tiny_preserved),
        },
        "cases": rows,
    }


def _sample_values(features: dict[str, np.ndarray],
                   mask: np.ndarray,
                   rng: np.random.Generator,
                   max_samples: int) -> dict[str, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return {name: np.asarray([], dtype=np.float32) for name in features}
    if coords.shape[0] > int(max_samples):
        coords = coords[rng.choice(coords.shape[0], size=int(max_samples), replace=False)]
    idx = (coords[:, 0], coords[:, 1], coords[:, 2])
    return {name: np.asarray(arr[idx], dtype=np.float32) for name, arr in features.items()}


def _auc_rank(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """Tie-tolerant enough rank AUC; direction is made feature-agnostic."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return None
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=bool), np.zeros(len(neg), dtype=bool)])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
    auc = (float(ranks[labels].sum()) - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(max(auc, 1.0 - auc))


def _run_contact_root_cause_separability(cases: list[str],
                                         plans: str,
                                         max_samples: int = 50000) -> dict:
    """Measure whether current input channels can separate contact from negatives.

    The matched near-negative is same-anatomy support that is close to contact
    but not contact. This is the important comparison. External hard negatives
    are easier; passing only that group would not explain or remove the broad
    support-noncontact false positives observed in full-volume probes.
    """
    rows = []
    for cid in cases:
        data, inst, _config_dir = _load_ds537_preprocessed_case(cid, plans=plans)
        support = inst > 0
        contact = _same_anatomy_contact_mask(inst)
        surface = _fragment_surface_mask(inst) & ~contact
        near_noncontact = ndi.binary_dilation(contact, iterations=4) & support & ~contact
        if int(near_noncontact.sum()) < 1000:
            near_noncontact = support & ~contact
        hard_external = np.zeros(inst.shape, dtype=bool)
        if data.shape[0] >= 6:
            hard_external = (data[5] > 0.35) & ~support
        if int(hard_external.sum()) < 1000:
            ct = data[0]
            thresh = float(np.percentile(ct[~support], 97)) if (~support).any() else float(np.percentile(ct, 97))
            hard_external = (ct >= thresh) & ~support

        ct = data[0].astype(np.float32, copy=False)
        grad = np.zeros_like(ct, dtype=np.float32)
        for g in np.gradient(ct):
            grad += np.asarray(g, dtype=np.float32) ** 2
        grad = np.sqrt(grad).astype(np.float32, copy=False)
        local_mean = ndi.uniform_filter(ct, size=3)
        local_sq = ndi.uniform_filter(ct * ct, size=3)
        local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0)).astype(np.float32, copy=False)
        if data.shape[0] >= 4:
            anat_max = np.max(data[1:4], axis=0).astype(np.float32)
            anat_margin = (np.max(data[1:4], axis=0) - np.partition(data[1:4], -2, axis=0)[-2]).astype(np.float32)
        else:
            anat_max = np.zeros_like(ct, dtype=np.float32)
            anat_margin = np.zeros_like(ct, dtype=np.float32)
        sdf = data[4].astype(np.float32, copy=False) if data.shape[0] >= 5 else np.zeros_like(ct, dtype=np.float32)
        hard = data[5].astype(np.float32, copy=False) if data.shape[0] >= 6 else np.zeros_like(ct, dtype=np.float32)
        features = {
            "ct": ct,
            "ct_gradient": grad,
            "anatomy_prob_max": anat_max,
            "anatomy_prob_margin": anat_margin,
            "pelvis_sdf": sdf,
            "hard_negative_prior": hard,
            "local_ct_mean3": local_mean,
            "local_ct_std3": local_std,
        }
        rng = np.random.default_rng(int(cid) * 997 + 537)
        samples = {
            "contact": _sample_values(features, contact, rng, max_samples),
            "same_fragment_internal_surface": _sample_values(features, surface, rng, max_samples),
            "same_anatomy_noncontact": _sample_values(features, near_noncontact, rng, max_samples),
            "external_hard_negative": _sample_values(features, hard_external, rng, max_samples),
        }
        aucs = {}
        for neg_name in ("same_fragment_internal_surface", "same_anatomy_noncontact", "external_hard_negative"):
            aucs[neg_name] = {}
            for feat in features:
                aucs[neg_name][feat] = _auc_rank(samples["contact"][feat], samples[neg_name][feat])
        best_by_group = {}
        for group, values in aucs.items():
            valid = [(k, v) for k, v in values.items() if v is not None]
            if valid:
                feat, value = max(valid, key=lambda item: float(item[1]))
                best_by_group[group] = {"feature": feat, "auc": float(value)}
            else:
                best_by_group[group] = {"feature": None, "auc": None}
        rows.append({
            "case": str(cid).zfill(3),
            "voxel_counts": {
                "contact": int(contact.sum()),
                "same_fragment_internal_surface": int(surface.sum()),
                "same_anatomy_noncontact": int(near_noncontact.sum()),
                "external_hard_negative": int(hard_external.sum()),
                "support": int(support.sum()),
            },
            "sample_counts": {
                name: int(len(next(iter(vals.values())))) if vals else 0
                for name, vals in samples.items()
            },
            "best_auc_by_negative_group": best_by_group,
            "auc_by_feature": aucs,
        })

    matched_aucs = [
        r["best_auc_by_negative_group"]["same_anatomy_noncontact"]["auc"]
        for r in rows
        if r["best_auc_by_negative_group"]["same_anatomy_noncontact"]["auc"] is not None
    ]
    mean_matched_auc = float(np.mean(matched_aucs)) if matched_aucs else 0.0
    passed = bool(matched_aucs and mean_matched_auc >= 0.70 and min(matched_aucs) >= 0.60)
    return {
        "gate": "target_input_separability",
        "passed": passed,
        "criteria": {
            "mean_matched_near_negative_auc_min": 0.70,
            "per_case_matched_near_negative_auc_floor": 0.60,
        },
        "summary": {
            "mean_matched_near_negative_auc": mean_matched_auc,
            "min_matched_near_negative_auc": float(np.min(matched_aucs)) if matched_aucs else 0.0,
            "next_target_variant": "dense_contact_energy" if passed else "local_contrastive_pair",
            "interpretation": (
                "single-voxel input features separate GT contact from matched non-contact well enough"
                if passed else
                "voxel contact is not separable enough; use local contrastive pair/ranking before fold0 training"
            ),
        },
        "cases": rows,
    }


def run_contact_root_cause(cases: list[str],
                           target_variant: str = "dense_contact_energy",
                           mode: str = "all",
                           plans: str = "nnUNetPlans_3d_fullres",
                           out_path: Path | None = None) -> dict:
    """Run JSON-only Dataset537 contact/edge root-cause gates.

    This command is a pre-training gate. It intentionally writes compact JSON
    only and does not create prediction caches or overlays. If it reports
    `training_allowed=false`, the next action is target/decoder/loss design,
    not another fold0 training run.
    """
    cases = [str(c).zfill(3) for c in cases]
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": DATASETS[537]["name"],
        "cases": cases,
        "mode": mode,
        "target_variant": target_variant,
        "plans": plans,
        "note": "JSON-only root-cause probe; no training, prediction cache, PNG, or NPZ artifact is created.",
    }
    if mode in {"oracle", "all"}:
        payload["oracle"] = _run_contact_root_cause_oracle(cases, target_variant, plans)
    if mode in {"separability", "all"}:
        payload["separability"] = _run_contact_root_cause_separability(cases, plans)

    oracle_ok = payload.get("oracle", {}).get("passed", True)
    separability_ok = payload.get("separability", {}).get("passed", True)
    payload["gate_summary"] = {
        "oracle_ok": bool(oracle_ok),
        "separability_ok": bool(separability_ok),
        "training_allowed": bool(oracle_ok and separability_ok),
        "next_step": (
            "1-case full-volume overfit may start"
            if oracle_ok and separability_ok else
            "do not train; fix target/decoder if oracle failed, otherwise use local_contrastive_pair target/loss"
        ),
    }
    out_path = out_path or (RESULT_REPORT / f"contact_root_cause537_{time.strftime('%Y%m%d', time.gmtime())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False))
    payload["out"] = str(out_path)
    return payload


def _auc_rank(pos: np.ndarray, neg: np.ndarray, max_samples: int = 50000) -> float | None:
    """Distribution-free separability AUC with deterministic downsampling."""
    pos = np.asarray(pos, dtype=np.float32).ravel()
    neg = np.asarray(neg, dtype=np.float32).ravel()
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None
    rng = np.random.default_rng(537)
    if pos.size > max_samples:
        pos = pos[rng.choice(pos.size, size=max_samples, replace=False)]
    if neg.size > max_samples:
        neg = neg[rng.choice(neg.size, size=max_samples, replace=False)]
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    rank_sum_pos = ranks[:pos.size].sum()
    auc = (rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / max(1.0, pos.size * neg.size)
    return float(max(auc, 1.0 - auc))


def _auc_rank_directional(pos: np.ndarray,
                          neg: np.ndarray,
                          max_samples: int = 50000,
                          seed: int = 537) -> dict | None:
    """Return direction-aware rank AUC for one feature.

    [METRIC][Risk:Major][Scope:separability]
    The previous helper reports `max(AUC, 1-AUC)`, which is useful for
    distribution-free separability but loses whether positives are higher or
    lower. For contact root-cause analysis we keep both: `auc_raw` and
    `auc_separable`. This prevents a later loss design from assuming the wrong
    feature direction.
    """
    pos = np.asarray(pos, dtype=np.float32).ravel()
    neg = np.asarray(neg, dtype=np.float32).ravel()
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None
    rng = np.random.default_rng(int(seed))
    if pos.size > max_samples:
        pos = pos[rng.choice(pos.size, size=max_samples, replace=False)]
    if neg.size > max_samples:
        neg = neg[rng.choice(neg.size, size=max_samples, replace=False)]
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size, dtype=bool), np.zeros(neg.size, dtype=bool)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    auc_raw = (float(ranks[labels].sum()) - pos.size * (pos.size + 1) / 2.0) / max(1.0, pos.size * neg.size)
    auc_sep = float(max(auc_raw, 1.0 - auc_raw))
    return {
        "auc_raw": float(auc_raw),
        "auc_separable": auc_sep,
        "positive_direction": "higher" if auc_raw >= 0.5 else "lower",
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
    }


def _describe_values(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float32).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _bicm_input_feature_maps(data: np.ndarray) -> dict[str, np.ndarray]:
    """Build input-only feature maps for BICM contact separability audit.

    [AUDIT][Risk:High][Scope:no_label_leakage]
    These maps are deterministic functions of the model input channels only.
    Label-derived distances are computed separately in the audit report and
    must not be interpreted as learnable input evidence.
    """
    ct = data[0].astype(np.float32, copy=False)
    grad_sq = np.zeros_like(ct, dtype=np.float32)
    for grad_axis in np.gradient(ct):
        grad_sq += np.asarray(grad_axis, dtype=np.float32) ** 2
    ct_gradient = np.sqrt(grad_sq).astype(np.float32, copy=False)
    local_mean3 = ndi.uniform_filter(ct, size=3)
    local_sq3 = ndi.uniform_filter(ct * ct, size=3)
    local_std3 = np.sqrt(np.maximum(local_sq3 - local_mean3 * local_mean3, 0.0)).astype(np.float32, copy=False)
    local_mean5 = ndi.uniform_filter(ct, size=5)
    local_sq5 = ndi.uniform_filter(ct * ct, size=5)
    local_std5 = np.sqrt(np.maximum(local_sq5 - local_mean5 * local_mean5, 0.0)).astype(np.float32, copy=False)
    features = {
        "ct_lut": ct,
        "ct_gradient": ct_gradient,
        "ct_laplacian_abs": np.abs(ndi.laplace(ct)).astype(np.float32, copy=False),
        "local_ct_std3": local_std3,
        "local_ct_std5": local_std5,
    }
    if data.shape[0] >= 2:
        anatomy_prob = data[1].astype(np.float32, copy=False)
        features["anatomy_prob"] = anatomy_prob
    if data.shape[0] >= 3:
        anatomy_sdf = data[2].astype(np.float32, copy=False)
        sdf_grad_sq = np.zeros_like(anatomy_sdf, dtype=np.float32)
        for grad_axis in np.gradient(anatomy_sdf):
            sdf_grad_sq += np.asarray(grad_axis, dtype=np.float32) ** 2
        features["anatomy_sdf"] = anatomy_sdf
        features["anatomy_sdf_abs"] = np.abs(anatomy_sdf).astype(np.float32, copy=False)
        features["anatomy_sdf_gradient"] = np.sqrt(sdf_grad_sq).astype(np.float32, copy=False)
    return features


def _bicm_contact_negative_masks(labels: np.ndarray,
                                 features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    support = (labels == 2) | (labels == 3) | (labels == 4)
    contact = labels == 4
    background = labels == 0
    ct = features["ct_lut"]
    high_ct_bg = np.zeros(labels.shape, dtype=bool)
    if background.any():
        threshold = float(np.percentile(ct[background], 95))
        high_ct_bg = background & (ct >= threshold)
    near_contact = ndi.binary_dilation(contact, iterations=3) & support & ~contact
    if int(near_contact.sum()) < 100:
        near_contact = (labels == 2) & ~contact
    return {
        "near_contact_support_noncontact": near_contact,
        "support_noncontact": support & ~contact,
        "shell_noncontact": labels == 2,
        "core_noncontact": labels == 3,
        "exterior_context": labels == 1,
        "high_ct_background": high_ct_bg,
    }


def _bicm_target_geometry_summary(labels: np.ndarray) -> dict:
    """Summarize target geometry using labels only.

    [DATA][Scope:target_audit]
    These numbers explain sparsity and topology but are not model input. If
    target geometry is clean while input-only separability is weak, the next
    design should move from voxel contact classification toward pairwise or
    distance-field supervision rather than just changing thresholds.
    """
    support = (labels == 2) | (labels == 3) | (labels == 4)
    contact = labels == 4
    core = labels == 3
    contact_cc, n_contact = ndi.label(contact)
    core_cc, n_core = ndi.label(core)
    support_cc, n_support = ndi.label(support)
    return {
        "support_voxels": int(support.sum()),
        "contact_voxels": int(contact.sum()),
        "core_voxels": int(core.sum()),
        "contact_fraction_of_support": float(contact.sum() / max(int(support.sum()), 1)),
        "contact_components": int(n_contact),
        "core_components": int(n_core),
        "support_components": int(n_support),
        "contact_component_sizes_top10": [
            int((contact_cc == comp_id).sum())
            for comp_id in range(1, min(int(n_contact), 10) + 1)
        ],
    }


def _load_bicm_v5_instance_sidecar(config_dir: Path, sample: str) -> np.ndarray | None:
    """Load the sidecar instance map for edge-pair separability audits.

    [DATA][Risk:High][Scope:sidecar_alignment]
    The BICM semantic label map stores support/contact/core classes, not
    original fragment IDs. Edge-positive and same-fragment-negative pairs must
    come from `bicm_v5_instance_targets/<sample>.npz`, which is generated in
    the same preprocessed geometry. Returning `None` keeps the older voxel-only
    audit usable when sidecars have not been built, but callers must report the
    missing edge evidence explicitly.
    """
    sidecar_path = config_dir / "bicm_v5_instance_targets" / f"{sample}.npz"
    if not sidecar_path.exists():
        return None
    sidecar = np.load(sidecar_path)
    inst = np.asarray(sidecar["instance"], dtype=np.uint16)
    return inst


def _sample_edge_values(feature: np.ndarray,
                        axis: int,
                        mask: np.ndarray,
                        rng: np.random.Generator,
                        max_samples: int,
                        mode: str) -> np.ndarray:
    """Sample pairwise feature values from a per-axis edge mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if coords.shape[0] > max_samples:
        coords = coords[rng.choice(coords.shape[0], size=max_samples, replace=False)]
    a = tuple(coords[:, d] for d in range(3))
    b_coords = coords.copy()
    b_coords[:, axis] += 1
    b = tuple(b_coords[:, d] for d in range(3))
    va = feature[a].astype(np.float32, copy=False)
    vb = feature[b].astype(np.float32, copy=False)
    if mode == "abs_delta":
        return np.abs(va - vb).astype(np.float32, copy=False)
    if mode == "mean":
        return ((va + vb) * 0.5).astype(np.float32, copy=False)
    raise ValueError(f"unknown edge feature mode: {mode}")


def _bicm_edge_pair_separability(data: np.ndarray,
                                 inst: np.ndarray,
                                 max_samples: int = 50000,
                                 seed: int = 537) -> dict:
    """Measure true fragment-break edges against same-fragment internal edges.

    [METRIC][Risk:High][Scope:contact_topology]
    V48 failed after giving the contact branch support/core context. This audit
    checks the representation that is closest to the decoder: adjacent voxel
    pairs. Positive pairs are same-ROI different-fragment adjacencies; negative
    pairs are same-fragment support-internal adjacencies. If these pairs are not
    separable by input-local edge features, a voxel/contact scalar loss is
    unlikely to solve the topology without a stronger pairwise representation.
    """
    features = _bicm_input_feature_maps(data)
    edge_auc: dict[str, dict] = {}
    axis_rows: list[dict] = []
    all_pos_by_feature: dict[str, list[np.ndarray]] = {}
    all_neg_by_feature: dict[str, list[np.ndarray]] = {}
    for axis in range(3):
        a_sl = [slice(None), slice(None), slice(None)]
        b_sl = [slice(None), slice(None), slice(None)]
        a_sl[axis] = slice(None, -1)
        b_sl[axis] = slice(1, None)
        a = inst[tuple(a_sl)]
        b = inst[tuple(b_sl)]
        support_pair = (a > 0) & (b > 0)
        pos = support_pair & (a != b)
        neg_internal = support_pair & (a == b)
        axis_pos = int(pos.sum())
        axis_neg = int(neg_internal.sum())
        axis_rows.append({
            "axis": int(axis),
            "positive_edge_pairs": axis_pos,
            "same_fragment_negative_edge_pairs": axis_neg,
        })
        if axis_pos == 0 or axis_neg == 0:
            continue
        rng = np.random.default_rng(int(seed) + 97 * int(axis))
        for feat_name, feat in features.items():
            for mode in ("abs_delta", "mean"):
                key = f"{feat_name}_{mode}"
                pos_vals = _sample_edge_values(feat, axis, pos, rng, max_samples, mode)
                neg_vals = _sample_edge_values(feat, axis, neg_internal, rng, max_samples, mode)
                all_pos_by_feature.setdefault(key, []).append(pos_vals)
                all_neg_by_feature.setdefault(key, []).append(neg_vals)

    best_feature = None
    best_auc = None
    best_direction = None
    for key in sorted(all_pos_by_feature):
        pos_vals = np.concatenate(all_pos_by_feature[key]) if all_pos_by_feature[key] else np.zeros((0,), dtype=np.float32)
        neg_vals = np.concatenate(all_neg_by_feature.get(key, [])) if all_neg_by_feature.get(key) else np.zeros((0,), dtype=np.float32)
        auc = _auc_rank_directional(pos_vals, neg_vals, max_samples=max_samples, seed=int(seed) + len(key))
        edge_auc[key] = auc
        if auc is not None and (best_auc is None or float(auc["auc_separable"]) > float(best_auc)):
            best_feature = key
            best_auc = float(auc["auc_separable"])
            best_direction = auc["positive_direction"]

    return {
        "axis_counts": axis_rows,
        "total_positive_edge_pairs": int(sum(r["positive_edge_pairs"] for r in axis_rows)),
        "total_same_fragment_negative_edge_pairs": int(sum(r["same_fragment_negative_edge_pairs"] for r in axis_rows)),
        "best_edge_pair_feature": best_feature,
        "best_edge_pair_auc": best_auc,
        "best_edge_pair_positive_direction": best_direction,
        "auc_by_edge_feature": edge_auc,
    }


def run_bicm_contact_audit(cases: list[str],
                           plans: str = "nnUNetPlans_3d_fullres",
                           max_samples: int = 50000,
                           out_path: Path | None = None) -> dict:
    """Audit whether current BICM input can separate contact from near negatives."""
    out_path = out_path or (RESULT_REPORT / f"bicm_contact_separability_audit_{time.strftime('%Y%m%d', time.gmtime())}.json")
    samples = []
    critical_auc_values = []
    edge_pair_auc_values = []
    positive_sample_count = 0
    edge_pair_positive_sample_count = 0
    for cid_raw in cases:
        cid = str(cid_raw).zfill(3)
        for anatomy in BICM_ROI_ANATOMIES:
            sample = f"PENGWIN_{cid}_{anatomy}"
            data, labels, config_dir = _load_ds537_bicm_preprocessed_sample(sample, plans=plans)
            features = _bicm_input_feature_maps(data)
            contact = labels == 4
            negatives = _bicm_contact_negative_masks(labels, features)
            rng_seed = int(cid) * 1009 + sum(ord(ch) for ch in anatomy)
            rng = np.random.default_rng(rng_seed)
            pos_values = _sample_values(features, contact, rng, max_samples)
            feature_distributions = {
                "contact": {name: _describe_values(values) for name, values in pos_values.items()}
            }
            auc_by_negative = {}
            best_by_negative = {}
            for neg_name, neg_mask in negatives.items():
                neg_values = _sample_values(features, neg_mask, rng, max_samples)
                feature_distributions[neg_name] = {
                    name: _describe_values(values) for name, values in neg_values.items()
                }
                auc_by_negative[neg_name] = {}
                best_feature = None
                best_auc = None
                best_direction = None
                for feat_name in features:
                    auc = _auc_rank_directional(
                        pos_values[feat_name],
                        neg_values[feat_name],
                        max_samples=max_samples,
                        seed=rng_seed + len(feat_name) + len(neg_name),
                    )
                    auc_by_negative[neg_name][feat_name] = auc
                    if auc is not None and (best_auc is None or float(auc["auc_separable"]) > float(best_auc)):
                        best_feature = feat_name
                        best_auc = float(auc["auc_separable"])
                        best_direction = auc["positive_direction"]
                best_by_negative[neg_name] = {
                    "feature": best_feature,
                    "auc": best_auc,
                    "positive_direction": best_direction,
                    "negative_voxels": int(neg_mask.sum()),
                }
            target_geometry = _bicm_target_geometry_summary(labels)
            if target_geometry["contact_voxels"] > 0:
                positive_sample_count += 1
                critical = best_by_negative["near_contact_support_noncontact"]["auc"]
                if critical is not None:
                    critical_auc_values.append(float(critical))
            edge_pair = None
            inst = _load_bicm_v5_instance_sidecar(config_dir, sample)
            if inst is not None:
                if tuple(inst.shape) != tuple(labels.shape):
                    raise ValueError(f"{sample}: sidecar instance shape {inst.shape} does not match labels {labels.shape}")
                edge_pair = _bicm_edge_pair_separability(
                    data,
                    inst,
                    max_samples=max_samples,
                    seed=int(cid) * 1009 + sum(ord(ch) for ch in anatomy),
                )
                if int(edge_pair["total_positive_edge_pairs"]) > 0:
                    edge_pair_positive_sample_count += 1
                    if edge_pair["best_edge_pair_auc"] is not None:
                        edge_pair_auc_values.append(float(edge_pair["best_edge_pair_auc"]))
            samples.append({
                "case": cid,
                "anatomy": anatomy,
                "sample": sample,
                "config_dir": str(config_dir),
                "input_channels": int(data.shape[0]),
                "shape": [int(v) for v in labels.shape],
                "target_geometry": target_geometry,
                "negative_voxels": {name: int(mask.sum()) for name, mask in negatives.items()},
                "best_auc_by_negative": best_by_negative,
                "auc_by_feature": auc_by_negative,
                "edge_pair_separability": edge_pair,
                "feature_distributions": feature_distributions,
            })
    mean_critical = float(np.mean(critical_auc_values)) if critical_auc_values else 0.0
    min_critical = float(np.min(critical_auc_values)) if critical_auc_values else 0.0
    mean_edge_pair = float(np.mean(edge_pair_auc_values)) if edge_pair_auc_values else 0.0
    min_edge_pair = float(np.min(edge_pair_auc_values)) if edge_pair_auc_values else 0.0
    voxel_passed = bool(critical_auc_values and mean_critical >= 0.70 and min_critical >= 0.60)
    edge_pair_passed = bool(edge_pair_auc_values and mean_edge_pair >= 0.70 and min_edge_pair >= 0.60)
    passed = bool(voxel_passed or edge_pair_passed)
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": "Dataset537 BICM contact separability audit",
        "Hypothesis": (
            "If contact voxels are not separable from support-internal near-negatives using current "
            "ct_lut_anat_sdf input features, more scalar contact loss or auxiliary heads should not be "
            "expected to pass the fixed 0.5 full-volume gate."
        ),
        "Input cases": [str(c).zfill(3) for c in cases],
        "Changed variable": "audit only; no training, no prediction, no threshold tuning",
        "Fixed variables": {
            "dataset": DATASETS[537]["name"],
            "plans": plans,
            "input_contract": "ct_lut_anat_sdf",
            "target_contract": "BICM labels 0..4, contact_surface=4",
        },
        "Pass criteria": {
            "mean_best_auc_contact_vs_near_support_noncontact_ge": 0.70,
            "min_best_auc_contact_vs_near_support_noncontact_ge": 0.60,
            "or_mean_best_auc_edge_break_vs_same_fragment_internal_edge_ge": 0.70,
            "or_min_best_auc_edge_break_vs_same_fragment_internal_edge_ge": 0.60,
        },
        "Result": {
            "positive_contact_roi_count": int(positive_sample_count),
            "mean_best_auc_contact_vs_near_support_noncontact": mean_critical,
            "min_best_auc_contact_vs_near_support_noncontact": min_critical,
            "voxel_contact_separability_passed": voxel_passed,
            "edge_pair_positive_roi_count": int(edge_pair_positive_sample_count),
            "mean_best_auc_edge_break_vs_same_fragment_internal_edge": mean_edge_pair,
            "min_best_auc_edge_break_vs_same_fragment_internal_edge": min_edge_pair,
            "edge_pair_separability_passed": edge_pair_passed,
            "passed": passed,
        },
        "Interpretation": (
            "edge-pair or voxel input features contain enough separability signal; next failure is likely model/loss calibration"
            if passed else
            "current input-local voxel and edge-pair features are weak against matched support near-negatives; next design should use deeper learned features or revise contact target/loss before more training"
        ),
        "Next decision": (
            "try one contact calibration objective only if it uses the separable feature family reported here"
            if passed else
            "do not run 159/20-case/fold0; inspect learned decoder features or revise contact target/loss before another training run"
        ),
        "samples": samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False))
    payload["out"] = str(out_path)
    return payload


def _run_boundary_separability(cases: list[str]) -> dict:
    """Audit whether CT/LUT-local features separate barrier from near negatives."""
    rows = []
    for case in cases:
        cd = find_case_dir(case)
        if cd is None:
            raise FileNotFoundError(f"source case not found: {case}")
        import SimpleITK as sitk

        img = canonicalize_sitk(sitk.ReadImage(str(cd / "image.mha")))
        lbl = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        ct = sitk.GetArrayFromImage(img).astype(np.float32)
        gt = sitk.GetArrayFromImage(lbl)
        sx, sy, sz = [float(v) for v in img.GetSpacing()]
        spacing_zyx = (sz, sy, sx)
        target, _audit = compute_boundary_fragment_target(gt, spacing_zyx, BoundaryFragmentParams())
        gz, gy, gx = np.gradient(ct, *spacing_zyx)
        grad = np.sqrt(gz * gz + gy * gy + gx * gx)
        barrier = target == 2
        near_negative = (target == 1) | (target == 3)
        same_support_negative = target == 3
        external_negative = target == 1
        ct_auc = _auc_rank(ct[barrier], ct[near_negative])
        grad_auc = _auc_rank(grad[barrier], grad[near_negative])
        support_grad_auc = _auc_rank(grad[barrier], grad[same_support_negative])
        external_grad_auc = _auc_rank(grad[barrier], grad[external_negative])
        best_auc = max(x for x in [ct_auc, grad_auc, support_grad_auc, external_grad_auc] if x is not None)
        rows.append({
            "case": case,
            "barrier_voxels": int(barrier.sum()),
            "near_negative_voxels": int(near_negative.sum()),
            "ct_vs_near_negative_auc": ct_auc,
            "gradient_vs_near_negative_auc": grad_auc,
            "gradient_vs_shell_auc": support_grad_auc,
            "gradient_vs_external_auc": external_grad_auc,
            "best_single_feature_auc": best_auc,
            "passed": bool(best_auc >= 0.70),
        })
    summary = {
        "mean_best_single_feature_auc": float(np.mean([r["best_single_feature_auc"] for r in rows])) if rows else 0.0,
        "min_best_single_feature_auc": float(np.min([r["best_single_feature_auc"] for r in rows])) if rows else 0.0,
        "passed": bool(rows and min(r["best_single_feature_auc"] for r in rows) >= 0.70),
        "interpretation": (
            "CT/LUT-local features are separable enough for 1-case V3 overfit"
            if rows and min(r["best_single_feature_auc"] for r in rows) >= 0.70
            else "single-voxel features are weak; keep CT-only ablation but expect loss/sampling to carry topology"
        ),
    }
    return {"summary": summary, "cases": rows}


def run_boundary_root_cause(cases: list[str],
                            mode: str = "all",
                            out_path: Path | None = None) -> dict:
    """Run V3 pre-training gates: oracle decoder and target separability."""
    cases = [str(c).zfill(3) for c in cases]
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": DATASETS[537]["name"],
        "mode": mode,
        "cases": cases,
        "note": "BoundaryFragment V3 root-cause probe; no training or prediction cache is created.",
    }
    if mode in {"oracle", "all"}:
        oracle_out = (out_path.parent / "boundary_fragment_oracle_detail.json") if out_path else (
            RESULT_REPORT / "boundary_fragment_oracle_detail.json"
        )
        oracle = run_boundary_fragment_eval(cases, out_path=oracle_out, oracle=True)
        summary = oracle["summary"]
        payload["oracle"] = {
            "summary": summary,
            "passed": bool(
                summary["iou_f_mean"] >= 0.95
                and summary["iou_f_min"] >= 0.90
                and not summary["missing_seed_cases"]
            ),
            "detail_json": str(oracle_out),
        }
    if mode == "oracle-sweep":
        sweep_out = (out_path.parent / "boundary_oracle_sweep_detail.json") if out_path else (
            RESULT_REPORT / "boundary_oracle_sweep_detail.json"
        )
        payload["oracle_sweep"] = run_boundary_oracle_sweep(cases, out_path=sweep_out)
    if mode in {"separability", "all"}:
        sep = _run_boundary_separability(cases)
        payload["separability"] = sep
    oracle_ok = payload.get("oracle", {}).get(
        "passed",
        payload.get("oracle_sweep", {}).get("passed", True),
    )
    sep_ok = payload.get("separability", {}).get("summary", {}).get("passed", True)
    payload["gate_summary"] = {
        "oracle_ok": bool(oracle_ok),
        "separability_ok": bool(sep_ok),
        "training_allowed": bool(oracle_ok),
        "next_step": (
            "run 1-case ct_hu/ct_lut overfit gate"
            if oracle_ok else
            "do not train; fix BoundaryFragment target/decoder definition"
        ),
        "note": "Separability informs input/loss choice; oracle failure blocks training.",
    }
    out_path = out_path or (RESULT_REPORT / f"boundary_root_cause537_{time.strftime('%Y%m%d', time.gmtime())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False))
    payload["out"] = str(out_path)
    return payload


TRACKED_LOW_IOU_FRAGMENTS = {
    "003": [52, 55],
    "011": [53],
    "017": [55, 53],
    "018": [101, 1],
    "025": [107, 110, 103],
    "159": [54, 1],
}


def _case_fragment_iou_map(case_row: dict) -> dict[int, float]:
    return {
        int(row["gt_id"]): float(row["best_iou"])
        for row in case_row["iou_f"].get("per_gt", [])
    }


def _raw_case_spacing_zyx(img_ref: object) -> tuple[float, float, float]:
    sx, sy, sz = [float(v) for v in img_ref.GetSpacing()]
    return (sz, sy, sx)


def _summarize_boundary_oracle_rows(rows: list[dict],
                                    decoder_profile: str,
                                    target_profile: str,
                                    contact_band_mm: float,
                                    contact_search_mm: float,
                                    contact_ridge_mm: float,
                                    contact_same_anatomy_only: bool,
                                    shell_mm: float) -> dict:
    return {
        "iou_f_mean": float(np.mean([r["iou_f"]["iou_f_mean"] for r in rows])) if rows else 0.0,
        "iou_f_min": float(np.min([r["iou_f"]["iou_f_mean"] for r in rows])) if rows else 0.0,
        "support_dice_mean": 1.0,
        "support_hd95_mean": 0.0,
        "barrier_precision_mean": 1.0,
        "barrier_recall_mean": 1.0,
        "barrier_ratio_mean": 1.0,
        "missing_seed_cases": [
            r["case"] for r in rows
            if r["target_audit"]["label_counts"].get(str(BFV3_LABELS["fragment_core"]), 0) == 0
        ],
        "multi_core_cases": [
            r["case"] for r in rows
            if r["target_audit"].get("multi_core_fragments")
        ],
        "decoder_profile": decoder_profile,
        "target_profile": target_profile,
        "contact_band_mm": float(contact_band_mm),
        "contact_search_mm": float(contact_search_mm),
        "contact_ridge_mm": float(contact_ridge_mm),
        "contact_same_anatomy_only": bool(contact_same_anatomy_only),
        "shell_mm": float(shell_mm),
    }


def run_boundary_oracle_sweep(cases: list[str],
                              out_path: Path,
                              contact_bands: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
                              decoder_profiles: tuple[str, ...] = ("hard_barrier", "geodesic_marker")) -> dict:
    """Compare the previous broad-band baseline against V3.1 thin-ridge target."""
    cases = [str(c).zfill(3) for c in cases]
    candidate_specs = [
        {
            "name": "baseline_v3_band_hard_barrier_3mm",
            "target_profile": "v3_band",
            "decoder_profile": "hard_barrier",
            "contact_band_mm": 3.0,
            "contact_search_mm": 3.0,
            "contact_ridge_mm": 1.0,
            "contact_same_anatomy_only": True,
        },
        {
            "name": "previous_best_v3_band_geodesic_1mm",
            "target_profile": "v3_band",
            "decoder_profile": "geodesic_marker",
            "contact_band_mm": 1.0,
            "contact_search_mm": 3.0,
            "contact_ridge_mm": 1.0,
            "contact_same_anatomy_only": True,
        },
        {
            "name": "v31_thin_ridge_geodesic",
            "target_profile": "v3_thin_ridge",
            "decoder_profile": "geodesic_marker",
            "contact_band_mm": 3.0,
            "contact_search_mm": 3.0,
            "contact_ridge_mm": 1.0,
            "contact_same_anatomy_only": False,
        },
        {
            "name": "v31_thin_ridge_geodesic_v2",
            "target_profile": "v3_thin_ridge",
            "decoder_profile": "geodesic_marker_v2",
            "contact_band_mm": 3.0,
            "contact_search_mm": 3.0,
            "contact_ridge_mm": 1.0,
            "contact_same_anatomy_only": False,
        },
    ]
    experiment = {
        "Experiment": "V3.1 oracle decoder topology sweep",
        "Hypothesis": (
            "oracle failure is caused by broad class2 support removal and multi-component "
            "fallback core markers. V3.1 should improve topology by using a thin ridge "
            "and one connected core marker per GT fragment."
        ),
        "Input cases": cases,
        "Changed variable": {
            "candidate_specs": candidate_specs,
        },
        "Fixed variables": {
            "source": "GT labels",
            "label_contract": "0 background, 1 external, 2 thin pairwise ridge/barrier, 3 shell, 4 core",
            "external_band_mm": "0.0 in oracle sweep because class1/background are both decoder-forbidden",
            "shell_mm": 8.0,
            "training": "forbidden",
            "threshold_tuning": "forbidden",
        },
        "Pass criteria": {
            "mean_iou_f": 0.95,
            "per_case_iou_f": 0.90,
            "missing_seed_cases": 0,
            "multi_core_cases": 0,
        },
        "Failure action": "inspect low-IoU topology rows; if core is fixed, revise pairwise ridge definition",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_dir = out_path.parent / "boundary_oracle_sweep_cases"
    detail_dir.mkdir(parents=True, exist_ok=True)
    case_cache = {}
    for case in cases:
        _image, img_ref, gt, _label_ref, _case_dir = _read_case_image_and_label(case)
        case_cache[case] = {
            "gt": gt.astype(np.uint16, copy=False),
            "spacing_zyx": _raw_case_spacing_zyx(img_ref),
        }
    target_cache = {}
    max_band = max(float(spec["contact_search_mm"]) for spec in candidate_specs)
    needed_same_anatomy_flags = sorted({bool(spec["contact_same_anatomy_only"]) for spec in candidate_specs})
    barrier_distance_cache = {
        (case, same_anatomy_only): contact_barrier_distance(
            payload["gt"],
            spacing_zyx=payload["spacing_zyx"],
            max_radius_mm=max_band,
            same_anatomy_only=same_anatomy_only,
        )
        for case, payload in case_cache.items()
        for same_anatomy_only in needed_same_anatomy_flags
    }
    for spec in candidate_specs:
        key = (
            spec["target_profile"],
            float(spec["contact_band_mm"]),
            float(spec["contact_search_mm"]),
            float(spec["contact_ridge_mm"]),
            bool(spec["contact_same_anatomy_only"]),
        )
        params = BoundaryFragmentParams(
            target_profile=str(spec["target_profile"]),
            contact_band_mm=float(spec["contact_band_mm"]),
            contact_search_mm=float(spec["contact_search_mm"]),
            contact_ridge_mm=float(spec["contact_ridge_mm"]),
            contact_same_anatomy_only=bool(spec["contact_same_anatomy_only"]),
            external_band_mm=0.0,
            shell_mm=8.0,
        )
        for case, payload in case_cache.items():
            if (case, key) in target_cache:
                continue
            barrier = None
            if spec["target_profile"] == "v3_band":
                barrier = barrier_distance_cache[(case, bool(spec["contact_same_anatomy_only"]))] <= float(spec["contact_band_mm"])
            target, target_audit = compute_boundary_fragment_target(
                payload["gt"],
                spacing_zyx=payload["spacing_zyx"],
                params=params,
                precomputed_barrier=barrier,
                precomputed_contact_distance=barrier_distance_cache[(case, bool(spec["contact_same_anatomy_only"]))],
            )
            target_cache[(case, key)] = (target, target_audit)
    candidates = []
    for spec in candidate_specs:
        decoder_profile = str(spec["decoder_profile"])
        key = (
            spec["target_profile"],
            float(spec["contact_band_mm"]),
            float(spec["contact_search_mm"]),
            float(spec["contact_ridge_mm"]),
            bool(spec["contact_same_anatomy_only"]),
        )
        detail_path = detail_dir / f"{spec['name']}.json"
        rows = []
        for case in cases:
            gt = case_cache[case]["gt"]
            target, target_audit = target_cache[(case, key)]
            decoded = decode_boundary_fragment(target, profile=decoder_profile)
            iouf = instance_iouf(decoded, gt)
            topology = oracle_topology_diagnostics(decoded, gt, target, iouf)
            rows.append({
                "case": case,
                "prediction": "oracle_target",
                "oracle": True,
                "iou_f": iouf,
                "support": {"dice": 1.0, "hd95_mm": 0.0, "nsd_2mm": 1.0},
                "class2_barrier": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f": 1.0,
                    "pred_voxels": int((target == BFV3_LABELS["fracture_barrier"]).sum()),
                    "target_voxels": int((target == BFV3_LABELS["fracture_barrier"]).sum()),
                    "pred_target_ratio": 1.0,
                },
                "target_audit": target_audit,
                "topology_diagnostics": topology,
            })
        summary = _summarize_boundary_oracle_rows(
            rows,
            decoder_profile=decoder_profile,
            target_profile=str(spec["target_profile"]),
            contact_band_mm=float(spec["contact_band_mm"]),
            contact_search_mm=float(spec["contact_search_mm"]),
            contact_ridge_mm=float(spec["contact_ridge_mm"]),
            contact_same_anatomy_only=bool(spec["contact_same_anatomy_only"]),
            shell_mm=8.0,
        )
        result = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "boundary_fragment_v31_oracle_sweep_candidate",
            "dataset": DATASETS[537]["name"],
            "candidate": spec["name"],
            "target_profile": spec["target_profile"],
            "decoder_profile": decoder_profile,
            "contact_band_mm": float(spec["contact_band_mm"]),
            "contact_search_mm": float(spec["contact_search_mm"]),
            "contact_ridge_mm": float(spec["contact_ridge_mm"]),
            "contact_same_anatomy_only": bool(spec["contact_same_anatomy_only"]),
            "shell_mm": 8.0,
            "cases": rows,
            "summary": summary,
        }
        detail_path.write_text(json.dumps(_json_sanitize(result), indent=2, allow_nan=False))
        tracked = {}
        for case_row in result["cases"]:
            case = str(case_row["case"]).zfill(3)
            ious = _case_fragment_iou_map(case_row)
            tracked[case] = {
                str(fragment_id): ious.get(int(fragment_id), 0.0)
                for fragment_id in TRACKED_LOW_IOU_FRAGMENTS.get(case, [])
            }
        low_fragments = []
        for case_row in result["cases"]:
            for row in case_row["topology_diagnostics"]["low_iou_fragments"]:
                low_fragments.append({
                    "case": case_row["case"],
                    "gt_id": row["gt_id"],
                    "gt_voxels": row["gt_voxels"],
                    "best_iou": row["best_iou"],
                    "pred_overlap_count": row["pred_overlap_count"],
                    "merged_with_gt_ids": row["merged_with_gt_ids"],
                    "barrier_assigned_outside_best_pred_voxels": row["barrier_assigned_outside_best_pred_voxels"],
                    "core_seed_components": row["core_seed_components"],
                })
        passed = bool(
            summary["iou_f_mean"] >= 0.95
            and summary["iou_f_min"] >= 0.90
            and not summary["missing_seed_cases"]
            and not summary["multi_core_cases"]
        )
        candidates.append({
            "candidate": spec["name"],
            "target_profile": spec["target_profile"],
            "decoder_profile": decoder_profile,
            "contact_band_mm": float(spec["contact_band_mm"]),
            "contact_search_mm": float(spec["contact_search_mm"]),
            "contact_ridge_mm": float(spec["contact_ridge_mm"]),
            "contact_same_anatomy_only": bool(spec["contact_same_anatomy_only"]),
            "detail_json": str(detail_path),
            "summary": summary,
            "tracked_low_iou_fragments": tracked,
            "low_iou_fragments": low_fragments,
            "passed": passed,
        })
    candidates.sort(
        key=lambda item: (
            bool(item["passed"]),
            bool(item["target_profile"] == "v3_thin_ridge"),
            bool(item["decoder_profile"] == "geodesic_marker_v2"),
            float(item["summary"]["iou_f_mean"]),
            float(item["summary"]["iou_f_min"]),
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else None
    passed = bool(best and best["passed"])
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_protocol": experiment,
        "cases": cases,
        "candidates": candidates,
        "best": best,
        "passed": passed,
        "Result": {
            "best_candidate": best["candidate"] if best else None,
            "best_target_profile": best["target_profile"] if best else None,
            "best_decoder_profile": best["decoder_profile"] if best else None,
            "best_contact_band_mm": best["contact_band_mm"] if best else None,
            "best_contact_search_mm": best["contact_search_mm"] if best else None,
            "best_contact_ridge_mm": best["contact_ridge_mm"] if best else None,
            "best_contact_same_anatomy_only": best["contact_same_anatomy_only"] if best else None,
            "best_mean_iou_f": best["summary"]["iou_f_mean"] if best else None,
            "best_min_iou_f": best["summary"]["iou_f_min"] if best else None,
        },
        "Interpretation": (
            "oracle target/decoder gate passed; freeze V3.1 target/decoder for the six-case rebuild"
            if passed else
            "oracle still fails; inspect low-IoU topology rows before any training"
        ),
        "Next decision": (
            "freeze target/decoder and run ct_hu/ct_lut six-case build"
            if passed else
            "do not train; revise pairwise ridge/core topology definition"
        ),
    }
    out_path.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", nargs="?", default="all",
	                        choices=[
	                            "all", "summary", "weights", "eval", "case", "hard-view",
                                "contact-root-cause", "boundary-root-cause", "bicm-contact-audit",
	                        ],
                        help="artifact group to generate")
    parser.add_argument("--eval-json", help="required for cmd=eval")
    parser.add_argument("--result-dir", default=str(RESULT),
                        help="root to scan for existing eval JSON files")
    parser.add_argument("--artifact-dir", default=str(RESULT_VISUALIZE),
                        help="active output folder for generated visual artifacts")
    parser.add_argument("--case", help="case ID for cmd=case, for example 025")
    parser.add_argument("--case-set", default="hard10",
                        choices=sorted(SOTA_CASE_SETS),
                        help="case set for cmd=hard-view")
    parser.add_argument("--pred", help="optional prediction .mha for cmd=case")
    parser.add_argument("--out-dir", help="output directory for cmd=case")
    parser.add_argument("--step-size", type=int, default=2,
                        help="marching-cubes step size for 3D surfaces; larger is faster/coarser")
    parser.add_argument("--cases", nargs="*", default=ROOT_CAUSE_CASES,
                        help="case IDs for cmd=contact-root-cause")
    parser.add_argument("--mode", choices=["oracle", "oracle-sweep", "separability", "all"], default="all",
                        help="root-cause probe mode for contact/boundary root-cause commands")
    parser.add_argument("--target-variant",
                        choices=["dense_contact_energy", "local_contrastive_pair", "edge_dilated_pair"],
                        default="dense_contact_energy",
                        help="target variant for oracle/root-cause probes")
    parser.add_argument("--plans", default="nnUNetPlans_3d_fullres",
                        help="nnU-Net plans/config folder for Dataset537 preprocessed data")
    parser.add_argument("--out", help="JSON output path for cmd=contact-root-cause")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    artifact_dir = Path(args.artifact_dir)
    stamp = time.strftime("%Y%m%d", time.gmtime())

    if args.cmd == "all":
        manifest = run_all(result_dir, artifact_dir)
        print(json.dumps(_json_sanitize(manifest), indent=2, allow_nan=False))
    elif args.cmd == "summary":
        print(json.dumps(write_result_summary(scan_eval_jsons(result_dir),
                                              stamp, artifact_dir),
                         indent=2, allow_nan=False))
    elif args.cmd == "weights":
        print(json.dumps(write_weight_manifest(scan_weight_manifest(),
                                               stamp, artifact_dir),
                         indent=2, allow_nan=False))
    elif args.cmd == "eval":
        if not args.eval_json:
            raise SystemExit("--eval-json is required for cmd=eval")
        eval_path = Path(args.eval_json)
        data = load_completed_eval(eval_path)
        if data is None:
            raise SystemExit(f"not a completed eval JSON: {args.eval_json}")
        suffix = _visual_suffix(eval_path)
        visual = write_eval_visualization(
            eval_path,
            out_text=artifact_dir / f"visual_{suffix}_{stamp}.txt",
            out_json=artifact_dir / f"visual_{suffix}_{stamp}.json",
        )
        print(json.dumps(_json_sanitize({
            "source": visual["source"],
            "scope": visual["scope"],
            "iou_f": visual["overall"]["iou_f"],
        }), indent=2, allow_nan=False))
    elif args.cmd == "case":
        if not args.case:
            raise SystemExit("--case is required for cmd=case")
        artifacts = visualize_case(
            args.case,
            pred_path=Path(args.pred) if args.pred else None,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            step_size=args.step_size,
        )
        print(json.dumps(_json_sanitize(artifacts), indent=2, allow_nan=False))
    elif args.cmd == "hard-view":
        manifest = export_hard_view(
            case_set=args.case_set,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            step_size=args.step_size,
        )
        print(json.dumps(_json_sanitize({
            "manifest": manifest["manifest"],
            "case_set": manifest["case_set"],
            "n_cases": len(manifest["cases"]),
        }), indent=2, allow_nan=False))
    elif args.cmd == "contact-root-cause":
        result = run_contact_root_cause(
            cases=args.cases,
            target_variant=args.target_variant,
            mode=args.mode,
            plans=args.plans,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(_json_sanitize({
            "out": result["out"],
            "gate_summary": result["gate_summary"],
            "oracle": result.get("oracle", {}).get("summary"),
            "separability": result.get("separability", {}).get("summary"),
        }), indent=2, allow_nan=False))
    elif args.cmd == "boundary-root-cause":
        result = run_boundary_root_cause(
            cases=args.cases,
            mode=args.mode,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(_json_sanitize({
            "out": result["out"],
            "gate_summary": result["gate_summary"],
            "oracle": result.get("oracle", {}).get("summary"),
            "separability": result.get("separability", {}).get("summary"),
        }), indent=2, allow_nan=False))
    elif args.cmd == "bicm-contact-audit":
        result = run_bicm_contact_audit(
            cases=args.cases,
            plans=args.plans,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(_json_sanitize({
            "out": result["out"],
            "result": result["Result"],
            "interpretation": result["Interpretation"],
            "next_decision": result["Next decision"],
        }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

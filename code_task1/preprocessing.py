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
    BICMV5Params, V5_ANATOMY_RANGES, V5_ANATOMY_RANGES_WITH_FEMUR,
    V5_LABELS, V5_TARGET_PROFILES,
    anatomy_range, anatomy_mask_from_instances, bbox_from_mask, compute_bicm_v5_target,
    label_distribution,
)
# Registry single source. The per-anatomy BICM V5 sidecar builder handles Femur
# ROIs (151-200) so it uses the FULL view; legacy pelvic-only V3/V4 builders keep
# the explicit pelvic_only=True view (their "drop femur" intent stays visible).
from anatomy_registry import (
    MAX_INSTANCE_ID,
    PELVIC_MAX_INSTANCE_ID,
    valid_instance_mask,
)
configure_nnunet_env()
log = get_logger(__name__)




# [V0.x][FIX:B1+B2][2026-05-31] Dataset539 5-class anatomy 의 softmax 채널 인덱스.
# Ds532 4-class (Sacrum=1/LeftHip=2/RightHip=3) 는 그대로 호환되고, Dataset539 의
# 5-class (Femur=4) 채널이 추가된다. 본 dict 는 Dataset537 (3-anatomy) 과
# Dataset538 (4-anatomy) 모두에서 anatomy-specific Ds532/539 prob 채널을 찾을 때
# 사용된다.


def _bone_lut_normalize(arr: np.ndarray) -> np.ndarray:
    """Deterministic bone-window LUT for the CT-LUT ablation."""
    x = arr.astype(np.float32, copy=False)
    xp = np.asarray([-1000.0, -200.0, 150.0, 700.0, 1500.0, 2000.0], dtype=np.float32)
    fp = np.asarray([0.0, 0.05, 0.35, 0.70, 0.92, 1.0], dtype=np.float32)
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp).astype(np.float32)


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


def _spacing_zyx(spacing_xyz: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    """Return spacing in numpy array order.

    SimpleITK exposes spacing as x/y/z, while `GetArrayFromImage` returns
    z/y/x. ABBC core generation uses Euclidean distance in millimetres; mixing
    the order would silently bias seed placement in anisotropic scans.
    """
    sx, sy, sz = [float(v) for v in spacing_xyz]
    return (sz, sy, sx)










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


def ndi_distance_transform(mask: np.ndarray,
                           spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.distance_transform_edt(mask, sampling=spacing_zyx).astype(np.float32, copy=False)





















































BICM_V5_INSTANCE_SIDECAR_DIR = "bicm_v5_instance_targets"
BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR = "boundary_fragment_v3_targets"


PENGWIN_BICM_V5_SIDECAR_MAX_LOCATIONS = int(
    os.environ.get("PENGWIN_BICM_V5_SIDECAR_MAX_LOCATIONS", "30000")
)


def _instance_centers_and_sizes(inst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-fragment (MAX_INSTANCE_ID+1, 3) centroids and voxel counts.

    Index 0 is reserved for background and stays NaN/0. Indices outside the valid
    1..MAX_INSTANCE_ID range (registry: Sacrum 1-50 ... Femur 151-200) are also
    left as NaN/0 so the sidecar contract matches the BICM V5 sidecar loader which
    addresses fragments by global PENGWIN fragment ID. Sized to the full registry
    range so per-anatomy Femur ROIs (IDs 151-200) no longer overflow the array.
    """
    n_slots = MAX_INSTANCE_ID + 1
    centers = np.full((n_slots, 3), np.nan, dtype=np.float32)
    sizes = np.zeros((n_slots,), dtype=np.int64)
    valid = valid_instance_mask(inst)
    if not valid.any():
        return centers, sizes
    flat_ids = inst[valid].astype(np.int64, copy=False)
    coords = np.argwhere(valid).astype(np.float32, copy=False)
    sizes_view = np.bincount(flat_ids, minlength=n_slots).astype(np.int64, copy=False)
    sizes[: sizes_view.shape[0]] = sizes_view[: sizes.shape[0]]
    sum_z = np.bincount(flat_ids, weights=coords[:, 0], minlength=n_slots)
    sum_y = np.bincount(flat_ids, weights=coords[:, 1], minlength=n_slots)
    sum_x = np.bincount(flat_ids, weights=coords[:, 2], minlength=n_slots)
    nonzero = np.flatnonzero(sizes_view[:n_slots] > 0)
    nonzero = nonzero[valid_instance_mask(nonzero)]
    if nonzero.size > 0:
        denom = sizes_view[nonzero].astype(np.float32)
        centers[nonzero, 0] = (sum_z[nonzero] / denom).astype(np.float32)
        centers[nonzero, 1] = (sum_y[nonzero] / denom).astype(np.float32)
        centers[nonzero, 2] = (sum_x[nonzero] / denom).astype(np.float32)
    return centers, sizes


def _fast_same_anatomy_contact_mask(inst: np.ndarray) -> np.ndarray:
    """Same-anatomy adjacency contact mask on a single anatomy ROI.

    Each V5 raw sample is already cropped to one anatomy (Sacrum / LeftHip /
    RightHip), so any pair of nonzero fragment voxels touching across a
    6-neighbor face is by construction same-anatomy. Keeping the same-anatomy
    guard makes the helper safe even when called on a full-pelvic instance
    array.
    """
    inst = inst.astype(np.int32, copy=False)
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
            & (((a - 1) // 50) == ((b - 1) // 50))
        )
        if same_anatomy.any():
            out_a = out[tuple(a_sl)]
            out_b = out[tuple(b_sl)]
            out_a[same_anatomy] = True
            out_b[same_anatomy] = True
            out[tuple(a_sl)] = out_a
            out[tuple(b_sl)] = out_b
    return out


def _sample_locations(mask: np.ndarray,
                      max_locations: int = PENGWIN_BICM_V5_SIDECAR_MAX_LOCATIONS,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Deterministically sample up to `max_locations` (z, y, x) coords from `mask`.

    Returns an (N, 3) int32 array. The sampler shrinks the mask down to a cap
    so the sidecar stays bounded for very large ROIs without losing positional
    diversity.
    """
    if mask.size == 0 or not np.any(mask):
        return np.zeros((0, 3), dtype=np.int32)
    coords = np.argwhere(mask).astype(np.int32, copy=False)
    cap = int(max(0, max_locations))
    if cap <= 0 or coords.shape[0] <= cap:
        return coords
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.choice(coords.shape[0], size=cap, replace=False)
    idx.sort()
    return coords[idx]


def _resample_instance_to_preprocessed(inst_roi: np.ndarray,
                                       raw_label_img: sitk.Image,
                                       pkl_properties: dict,
                                       target_shape_zyx: tuple[int, int, int],
                                       target_spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Resample an anatomy-ROI instance ID array to the preprocessed grid.

    The instance ID grid is aligned in voxel space to the raw V5 label MHA
    (since both come from the same anatomy bbox). nnUNet preprocessing changes
    spacing/shape but keeps origin/direction. We mirror that by constructing a
    reference image at the preprocessed spacing/shape with the same origin and
    direction, then nearest-neighbor resampling.

    `target_shape_zyx` is the preprocessed grid shape (matches the `seg` tensor
    in the preprocessed `.npz`) and `target_spacing_zyx` is the plan spacing
    from `nnUNetPlans.json`. Both must be passed explicitly because the per-
    sample pkl stores only the *source* spacing of the original case.
    """
    src = sitk.GetImageFromArray(inst_roi.astype(np.uint16, copy=False))
    src.SetSpacing(raw_label_img.GetSpacing())
    src.SetOrigin(raw_label_img.GetOrigin())
    src.SetDirection(raw_label_img.GetDirection())

    target_spacing_xyz = (
        float(target_spacing_zyx[2]),
        float(target_spacing_zyx[1]),
        float(target_spacing_zyx[0]),
    )
    out_size_xyz = (
        int(target_shape_zyx[2]),
        int(target_shape_zyx[1]),
        int(target_shape_zyx[0]),
    )

    bbox = pkl_properties.get("bbox_used_for_cropping")
    if bbox is not None:
        crop_lo_zyx = (int(bbox[0][0]), int(bbox[1][0]), int(bbox[2][0]))
    else:
        crop_lo_zyx = (0, 0, 0)
    direction = raw_label_img.GetDirection()
    direction_mat = np.array(direction, dtype=np.float64).reshape(3, 3)
    # bbox_used_for_cropping is in (z, y, x); convert to (x, y, z) for sitk origin shift.
    crop_lo_xyz = np.array(
        [crop_lo_zyx[2], crop_lo_zyx[1], crop_lo_zyx[0]], dtype=np.float64
    )
    raw_spacing_xyz = np.array(raw_label_img.GetSpacing(), dtype=np.float64)
    shift_world = direction_mat @ (crop_lo_xyz * raw_spacing_xyz)
    new_origin = tuple(float(o + s) for o, s in zip(raw_label_img.GetOrigin(), shift_world))

    ref = sitk.Image(out_size_xyz, sitk.sitkUInt16)
    ref.SetSpacing(target_spacing_xyz)
    ref.SetOrigin(new_origin)
    ref.SetDirection(direction)

    resampled = sitk.Resample(
        src,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )
    return sitk.GetArrayFromImage(resampled).astype(np.uint16, copy=False)


def _build_one_bicm_v5_instance_sidecar(sample_id: str,
                                        raw_label_path: Path,
                                        source_case_dir: Path,
                                        preprocessed_pkl: Path,
                                        preprocessed_npz: Path,
                                        out_path: Path,
                                        max_locations: int,
                                        plan_target_spacing_zyx: tuple[float, float, float]) -> dict:
    """Build the .npz sidecar for one anatomy ROI sample."""
    import pickle

    parts = sample_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected sample_id format: {sample_id!r}")
    cid = int(parts[1])
    anatomy = "_".join(parts[2:])
    # Full 4-anatomy view: Ds538 builds a Femur ROI sample (anatomy="Femur",
    # IDs 151-200). Ds537 (pelvic-only) simply never passes "Femur", so the
    # superset is safe and the femur ROI is no longer rejected here.
    if anatomy not in V5_ANATOMY_RANGES_WITH_FEMUR:
        raise ValueError(f"{sample_id}: anatomy {anatomy!r} not in {sorted(V5_ANATOMY_RANGES_WITH_FEMUR)}")
    lo, hi = V5_ANATOMY_RANGES_WITH_FEMUR[anatomy]

    raw_lbl_img = canonicalize_sitk(sitk.ReadImage(str(raw_label_path)))
    raw_v5_arr = sitk.GetArrayFromImage(raw_lbl_img)

    src_lbl_img = canonicalize_sitk(sitk.ReadImage(str(source_case_dir / "label.mha")))
    inst_full = sitk.GetArrayFromImage(src_lbl_img).astype(np.uint16, copy=False)
    anat_mask = anatomy_mask_from_instances(inst_full, anatomy)
    pad_vox = int(os.environ.get("PENGWIN_V5_ROI_PAD_VOX", "24"))
    bbox = bbox_from_mask(anat_mask, pad_vox=pad_vox)
    if bbox is None:
        raise RuntimeError(f"{sample_id}: empty anatomy ROI")
    inst_roi = inst_full[bbox]
    inst_roi = np.where((inst_roi >= lo) & (inst_roi <= hi), inst_roi, 0).astype(np.uint16, copy=False)
    if inst_roi.shape != raw_v5_arr.shape:
        raise RuntimeError(
            f"{sample_id}: raw instance ROI shape {inst_roi.shape} != raw V5 label "
            f"shape {raw_v5_arr.shape} (bbox replay mismatch)"
        )

    with open(preprocessed_pkl, "rb") as fh:
        props = pickle.load(fh)
    preprocessed = np.load(preprocessed_npz)
    seg = np.asarray(preprocessed["seg"])
    if seg.ndim == 4:
        seg = seg[0]
    seg = seg.astype(np.int16, copy=False)

    target_shape_zyx = tuple(int(v) for v in seg.shape)
    instance = _resample_instance_to_preprocessed(
        inst_roi, raw_lbl_img, props,
        target_shape_zyx=target_shape_zyx,
        target_spacing_zyx=plan_target_spacing_zyx,
    )
    if instance.shape != seg.shape:
        raise RuntimeError(
            f"{sample_id}: resampled instance shape {instance.shape} != preprocessed seg shape {seg.shape}"
        )
    # The V5 raw target labels (0..4) define support/exterior/contact regions and
    # are the source of truth for the spatial location pools below. Instance IDs
    # add fragment identity but do not redefine the location masks.
    bg = V5_LABELS["background"]
    ext = V5_LABELS["exterior_context"]
    shell = V5_LABELS["interior_shell"]
    core = V5_LABELS["core"]
    contact = V5_LABELS["contact_surface"]
    support_mask = (seg == shell) | (seg == core) | (seg == contact)
    contact_mask = (seg == contact)
    edge_mask = _fast_same_anatomy_contact_mask(instance)
    if not contact_mask.any() and edge_mask.any():
        # Some V5 target profiles emit no `contact_surface` voxels even when
        # adjacent same-anatomy fragments touch; treat the adjacency mask as a
        # contact-location fallback in that case so the sidecar pool is never
        # empty for multi-fragment ROIs.
        contact_mask = edge_mask
    hard_negative_mask = (seg == ext)
    # Tiny fragments: per-fragment support below a small voxel threshold.
    centers_zyx, fragment_sizes = _instance_centers_and_sizes(instance)
    tiny_threshold = int(os.environ.get("PENGWIN_BICM_V5_TINY_FRAGMENT_VOXELS", "512"))
    tiny_mask = np.zeros_like(support_mask, dtype=bool)
    if support_mask.any():
        for fid in range(1, MAX_INSTANCE_ID + 1):
            sz = int(fragment_sizes[fid])
            if 0 < sz <= tiny_threshold:
                tiny_mask |= (instance == fid)

    rng = np.random.default_rng(int(cid) * 1000 + (lo // 50))
    contact_locations = _sample_locations(contact_mask, max_locations, rng)
    tiny_locations = _sample_locations(tiny_mask, max_locations, rng)
    support_locations = _sample_locations(support_mask & ~contact_mask, max_locations, rng)
    hard_negative_locations = _sample_locations(hard_negative_mask, max_locations, rng)
    edge_locations = _sample_locations(edge_mask, max_locations, rng)

    valid_fragments = [int(fid) for fid in range(1, MAX_INSTANCE_ID + 1) if int(fragment_sizes[fid]) > 0]
    audit = {
        "sample_id": sample_id,
        "case_id": f"{cid:03d}",
        "anatomy": anatomy,
        "anatomy_range": [int(lo), int(hi)],
        "raw_v5_shape": list(int(v) for v in raw_v5_arr.shape),
        "preprocessed_shape": list(int(v) for v in seg.shape),
        "fragment_count": len(valid_fragments),
        "valid_fragment_ids": valid_fragments,
        "voxel_counts": {
            "support": int(support_mask.sum()),
            "contact": int(contact_mask.sum()),
            "edge_adjacency": int(edge_mask.sum()),
            "tiny": int(tiny_mask.sum()),
            "hard_negative": int(hard_negative_mask.sum()),
            "background": int((seg == bg).sum()),
        },
        "location_counts": {
            "contact_locations": int(contact_locations.shape[0]),
            "tiny_locations": int(tiny_locations.shape[0]),
            "support_locations": int(support_locations.shape[0]),
            "hard_negative_locations": int(hard_negative_locations.shape[0]),
            "edge_locations": int(edge_locations.shape[0]),
        },
        "tiny_fragment_voxel_threshold": tiny_threshold,
        "max_locations": int(max_locations),
        "v5_label_map": dict(V5_LABELS),
        "validations": {
            "instance_shape_matches_seg": bool(instance.shape == seg.shape),
            "support_covers_all_instance_voxels": bool(((instance > 0) & ~support_mask).sum() == 0),
        },
    }
    audit_json = np.array(json.dumps(audit, sort_keys=True), dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends `.npz` to filenames without that suffix; write to a
    # `.partial.npz` next to the final path so concurrent writers cannot read
    # a half-written sidecar.
    tmp_stem = out_path.with_suffix("")
    tmp_path = tmp_stem.with_name(tmp_stem.name + ".partial")
    tmp_npz = tmp_path.with_suffix(".partial.npz")
    if tmp_npz.exists():
        tmp_npz.unlink()
    np.savez(
        str(tmp_path),
        centers_zyx=centers_zyx,
        fragment_sizes=fragment_sizes,
        contact_locations=contact_locations,
        hard_negative_locations=hard_negative_locations,
        tiny_locations=tiny_locations,
        support_locations=support_locations,
        edge_locations=edge_locations,
        instance=instance,
        audit_json=audit_json,
    )
    os.replace(str(tmp_npz), str(out_path))
    return audit


def build_bicm_v5_instance_sidecars(ds_id: int = 537,
                                    force: bool = False,
                                    case_subset: list[str] | None = None,
                                    plan_configuration: str = "nnUNetPlans_3d_fullres",
                                    ) -> dict:
    """Build BICM V5 per-sample instance sidecars for the requested dataset.

    For each `PENGWIN_<cid>_<anatomy>.mha` in the raw `labelsTr` directory the
    function:
      1. Reads the raw V5 target MHA (the cropped anatomy ROI label, values 0..4).
      2. Reads the original case `label.mha` and rebuilds the anatomy-ROI
         instance ID array using the same bbox/pad_vox contract as
         `build_bicm_v5_dataset`.
      3. Resamples the instance ID ROI onto the preprocessed grid recorded in
         `nnUNetPlans_3d_fullres/<sample_id>.pkl` (nearest-neighbor).
      4. Combines the resampled instance map with the preprocessed V5 seg
         (`<sample_id>.npz['seg']`) to derive centers, sizes, and contact /
         tiny / support / hard-negative / edge location pools.
      5. Saves the sidecar `.npz` under
         `<preprocessed>/<plan_configuration>/bicm_v5_instance_targets/`.

    Returns a dict with `dataset_id`, `output_dir`, list of written `samples`,
    and `audit_path` pointing at the per-sample audit JSON.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "bicm_v5":
        raise ValueError(f"Dataset{ds_id} is not a BICM V5 dataset")

    preprocessed_root = NN_PREP / cfg["name"]

    # Resolve the plan, then take the preprocessed-DATA directory from the plan's
    # `data_identifier` — never assume the data dir name equals the plan name.
    # nnUNet's ResEnc planners reuse the DEFAULT preprocessor, so the plan FILE
    # (e.g. nnUNetResEncUNetLPlans.json) and the preprocessed-data dir
    # (data_identifier, e.g. "nnUNetPlans_3d_fullres") have DIFFERENT names. The
    # old code hardcoded "nnUNetPlans.json" + used plan_configuration as the data
    # dir, which (a) missed the ResEncL plan's spacing and (b) on a ResEncL-only
    # tree (no nnUNetPlans.json) failed outright. plan_configuration names the
    # PLAN: "<plans_name>_<config>" (e.g. "nnUNetResEncUNetLPlans_3d_fullres").
    if "_" in plan_configuration:
        plans_name, plan_cfg_key = plan_configuration.split("_", 1)
    else:
        plans_name, plan_cfg_key = "nnUNetPlans", plan_configuration
    plans_path = preprocessed_root / f"{plans_name}.json"
    if not plans_path.exists():
        raise FileNotFoundError(f"Missing {plans_name}.json under {preprocessed_root}")
    plans = json.loads(plans_path.read_text())
    plan_cfg = plans.get("configurations", {}).get(plan_cfg_key)
    if plan_cfg is None:
        raise KeyError(
            f"Plan configuration {plan_cfg_key!r} not found in {plans_path}; "
            f"available: {sorted(plans.get('configurations', {}))}"
        )
    plan_target_spacing_zyx = tuple(float(v) for v in plan_cfg["spacing"])
    data_identifier = plan_cfg.get("data_identifier", plan_configuration)

    raw_labels_dir = NN_RAW / cfg["name"] / "labelsTr"
    preprocessed_dir = preprocessed_root / data_identifier
    output_dir = preprocessed_dir / BICM_V5_INSTANCE_SIDECAR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_label_files = sorted(raw_labels_dir.glob("PENGWIN_*.mha"))
    if case_subset:
        wanted = {str(c).zfill(3) for c in case_subset}
        raw_label_files = [
            p for p in raw_label_files
            if p.name.split("_")[1] in wanted
        ]
    if not raw_label_files:
        raise FileNotFoundError(
            f"No raw V5 label MHA found under {raw_labels_dir} for subset={case_subset!r}"
        )

    max_locations = PENGWIN_BICM_V5_SIDECAR_MAX_LOCATIONS
    log.info(
        "[bicm_v5_sidecars] Ds%d %s — %d samples → %s (max_locations=%d)",
        ds_id, cfg["name"], len(raw_label_files), output_dir, max_locations,
    )

    audits: list[dict] = []
    written = 0
    skipped = 0
    for raw_lbl in raw_label_files:
        sample_id = raw_lbl.stem  # PENGWIN_003_LeftHip
        out_path = output_dir / f"{sample_id}.npz"
        if out_path.exists() and not force:
            skipped += 1
            continue
        cid_str = sample_id.split("_")[1]
        case_dir = find_case_dir(cid_str)
        if case_dir is None:
            raise FileNotFoundError(f"{sample_id}: source case {cid_str} not found under {DATA_RAW}")
        preprocessed_pkl = preprocessed_dir / f"{sample_id}.pkl"
        preprocessed_npz = preprocessed_dir / f"{sample_id}.npz"
        if not preprocessed_pkl.exists() or not preprocessed_npz.exists():
            raise FileNotFoundError(
                f"{sample_id}: missing preprocessed pkl/npz under {preprocessed_dir}; "
                "run `nnUNetv2_plan_and_preprocess` for the dataset first."
            )
        audit = _build_one_bicm_v5_instance_sidecar(
            sample_id=sample_id,
            raw_label_path=raw_lbl,
            source_case_dir=case_dir,
            preprocessed_pkl=preprocessed_pkl,
            preprocessed_npz=preprocessed_npz,
            out_path=out_path,
            max_locations=max_locations,
            plan_target_spacing_zyx=plan_target_spacing_zyx,
        )
        audits.append(audit)
        written += 1
        log.info("  [%d/%d] %s", written + skipped, len(raw_label_files), sample_id)

    audit_path = RESULT_REPORT / f"build_bicm_v5_instance_sidecars_ds{ds_id}_{RESULT_DATE}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "dataset_id": ds_id,
        "dataset_name": cfg["name"],
        "output_dir": str(output_dir),
        "plan_configuration": plan_configuration,
        "max_locations": max_locations,
        "force": bool(force),
        "case_subset": case_subset,
        "written": written,
        "skipped": skipped,
        "samples": audits,
    }, indent=2))
    log.info(
        "[bicm_v5_sidecars] wrote=%d skipped=%d audit=%s",
        written, skipped, audit_path,
    )
    return {
        "dataset_id": ds_id,
        "output_dir": str(output_dir),
        "written": written,
        "skipped": skipped,
        "samples": audits,
        "audit_path": str(audit_path),
    }


def generate_boundary_fragment_v3_target_sidecars(
    ds_id: int = 537,
    force: bool = False,
    case_subset: list[str] | None = None,
    plan_configuration: str = "nnUNetPlans_3d_fullres",
) -> dict:
    """Build BoundaryFragment V3 (5-class) target sidecars on the preprocessed ROI grid.

    For each preprocessed sample `PENGWIN_<cid>_<anatomy>.npz` the function:
      1. Loads the per-sample BICM V5 instance sidecar (`bicm_v5_instance_targets/<sample_id>.npz`)
         whose `instance` array is the global PENGWIN fragment-ID grid already
         resampled onto the preprocessed plan spacing.
      2. Calls `compute_boundary_fragment_target` with the plan target spacing
         to derive the dense V3 semantic target
         (0=background, 1=external_context, 2=fracture_barrier,
          3=fragment_shell, 4=fragment_core).
      3. Validates that every emitted label is within {0..4} and that every GT
         fragment receives at least one core voxel (same invariant the raw
         per-case worker enforces).
      4. Saves the sidecar `.npz` containing `target` (uint8, shape matches the
         preprocessed seg grid) under
         `<preprocessed>/<plan_configuration>/boundary_fragment_v3_targets/`.

    Returns a dict with `dataset_id`, `output_dir`, the list of written
    `samples`, and `audit_path` pointing at the per-sample audit JSON.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "bicm_v5":
        raise ValueError(
            f"Dataset{ds_id} ({cfg['name']}) is not a BICM V5 dataset; cannot "
            "derive BoundaryFragment V3 targets from instance sidecars."
        )

    preprocessed_root = NN_PREP / cfg["name"]

    # Resolve plan, then take the data dir from the plan's data_identifier (see
    # build_bicm_v5_instance_sidecars: under ResEnc the plan-file name and the
    # data-dir name differ). plan_configuration names the PLAN "<plans_name>_<config>".
    if "_" in plan_configuration:
        plans_name, plan_cfg_key = plan_configuration.split("_", 1)
    else:
        plans_name, plan_cfg_key = "nnUNetPlans", plan_configuration
    plans_path = preprocessed_root / f"{plans_name}.json"
    if not plans_path.exists():
        raise FileNotFoundError(f"Missing {plans_name}.json under {preprocessed_root}")
    plans = json.loads(plans_path.read_text())
    plan_cfg = plans.get("configurations", {}).get(plan_cfg_key)
    if plan_cfg is None:
        raise KeyError(
            f"Plan configuration {plan_cfg_key!r} not found in {plans_path}; "
            f"available: {sorted(plans.get('configurations', {}))}"
        )
    plan_target_spacing_zyx = tuple(float(v) for v in plan_cfg["spacing"])
    data_identifier = plan_cfg.get("data_identifier", plan_configuration)

    preprocessed_dir = preprocessed_root / data_identifier
    instance_dir = preprocessed_dir / BICM_V5_INSTANCE_SIDECAR_DIR
    output_dir = preprocessed_dir / BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not instance_dir.is_dir():
        raise FileNotFoundError(
            f"BICM V5 instance sidecar directory missing: {instance_dir}. "
            "Run `build-instance-sidecars` before deriving BoundaryFragment V3 targets."
        )
    instance_files = sorted(instance_dir.glob("PENGWIN_*.npz"))
    if case_subset:
        wanted = {str(c).zfill(3) for c in case_subset}
        instance_files = [
            p for p in instance_files
            if p.name.split("_")[1] in wanted
        ]
    if not instance_files:
        raise FileNotFoundError(
            f"No BICM V5 instance sidecars found under {instance_dir} for subset={case_subset!r}"
        )

    params = BoundaryFragmentParams()
    log.info(
        "[bfv3_targets] Ds%d %s — %d samples → %s (spacing_zyx=%s)",
        ds_id, cfg["name"], len(instance_files), output_dir, plan_target_spacing_zyx,
    )

    audits: list[dict] = []
    written = 0
    skipped = 0
    for inst_path in instance_files:
        sample_id = inst_path.stem  # PENGWIN_003_LeftHip
        out_path = output_dir / f"{sample_id}.npz"
        if out_path.exists() and not force:
            skipped += 1
            continue

        preprocessed_npz = preprocessed_dir / f"{sample_id}.npz"
        if not preprocessed_npz.exists():
            raise FileNotFoundError(
                f"{sample_id}: missing preprocessed npz {preprocessed_npz}; "
                "run `nnUNetv2_plan_and_preprocess` for the dataset first."
            )

        with np.load(inst_path) as payload:
            if "instance" not in payload:
                raise KeyError(
                    f"{inst_path} does not contain `instance`; rebuild bicm_v5 sidecars."
                )
            instance = np.asarray(payload["instance"], dtype=np.uint16)

        # Sanity-check the instance grid shape against the preprocessed seg.
        with np.load(preprocessed_npz) as prep:
            seg = np.asarray(prep["seg"])
        if seg.ndim == 4:
            seg = seg[0]
        if instance.shape != seg.shape:
            raise RuntimeError(
                f"{sample_id}: instance sidecar shape {instance.shape} != "
                f"preprocessed seg shape {seg.shape}"
            )

        target, target_audit = compute_boundary_fragment_target(
            instance,
            spacing_zyx=plan_target_spacing_zyx,
            params=params,
        )
        target = target.astype(np.uint8, copy=False)
        if target.shape != instance.shape:
            raise RuntimeError(
                f"{sample_id}: BFV3 target shape {target.shape} != instance shape {instance.shape}"
            )
        invalid = set(int(v) for v in np.unique(target)) - set(BFV3_LABELS.values())
        if invalid:
            raise RuntimeError(
                f"{sample_id}: invalid BFV3 labels {sorted(invalid)} (allowed {sorted(BFV3_LABELS.values())})"
            )
        if (instance > 0).any() and not (target == BFV3_LABELS["fragment_core"]).any():
            raise RuntimeError(
                f"{sample_id}: no class-4 core voxels emitted for a non-empty instance ROI"
            )

        audit = {
            "sample_id": sample_id,
            "preprocessed_shape": list(int(v) for v in target.shape),
            "plan_target_spacing_zyx": list(float(v) for v in plan_target_spacing_zyx),
            "target_audit": target_audit,
        }
        audit_json = np.array(json.dumps(audit, sort_keys=True), dtype=object)

        # Atomic write via `.partial.npz` so concurrent readers never see a
        # half-written sidecar (mirrors the BICM V5 sidecar contract).
        tmp_stem = out_path.with_suffix("")
        tmp_path = tmp_stem.with_name(tmp_stem.name + ".partial")
        tmp_npz = tmp_path.with_suffix(".partial.npz")
        if tmp_npz.exists():
            tmp_npz.unlink()
        np.savez(
            str(tmp_path),
            target=target,
            audit_json=audit_json,
        )
        os.replace(str(tmp_npz), str(out_path))

        audits.append(audit)
        written += 1
        if written % 25 == 0 or (written + skipped) == len(instance_files):
            log.info("  [%d/%d] %s", written + skipped, len(instance_files), sample_id)

    audit_path = RESULT_REPORT / f"build_boundary_fragment_v3_target_sidecars_ds{ds_id}_{RESULT_DATE}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "dataset_id": ds_id,
        "dataset_name": cfg["name"],
        "output_dir": str(output_dir),
        "plan_configuration": plan_configuration,
        "plan_target_spacing_zyx": list(float(v) for v in plan_target_spacing_zyx),
        "force": bool(force),
        "case_subset": case_subset,
        "written": written,
        "skipped": skipped,
        "params": {
            "target_profile": params.target_profile,
            "contact_search_mm": float(params.contact_search_mm),
            "contact_ridge_mm": float(params.contact_ridge_mm),
            "contact_same_anatomy_only": bool(params.contact_same_anatomy_only),
            "external_band_mm": float(params.external_band_mm),
            "shell_mm": float(params.shell_mm),
            "tiny_core_radius_mm": float(params.tiny_core_radius_mm),
        },
        "samples": audits,
    }, indent=2))
    log.info(
        "[bfv3_targets] wrote=%d skipped=%d audit=%s",
        written, skipped, audit_path,
    )
    return {
        "dataset_id": ds_id,
        "output_dir": str(output_dir),
        "written": written,
        "skipped": skipped,
        "samples": audits,
        "audit_path": str(audit_path),
    }


















# =============================================================================
# Unpack (.npz → .npy after nnUNet preprocess)
# =============================================================================


# =============================================================================
# Statistics (analyze_dataset 통합)
# =============================================================================






# =============================================================================
# Sanity case inspection (absorbed from data/preprocess/inspect_cases.py)
# =============================================================================


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """Command-line dispatcher for preprocessing utilities."""
    parser = argparse.ArgumentParser(
        prog="preprocessing",
        description="PENGWIN 2026 Task 1 preprocessing utilities.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sidecar = sub.add_parser(
        "build-instance-sidecars",
        help="Build BICM V5 instance sidecars under <preprocessed>/<plan>/bicm_v5_instance_targets/.",
    )
    p_sidecar.add_argument("--dataset", type=int, default=537, help="nnUNet dataset ID (default 537).")
    p_sidecar.add_argument("--force", action="store_true", help="Overwrite existing sidecars.")
    p_sidecar.add_argument(
        "--case-subset",
        nargs="*",
        default=None,
        help="Optional list of source case IDs (zero-padded) to restrict the build.",
    )
    p_sidecar.add_argument(
        "--plan-configuration",
        default="nnUNetPlans_3d_fullres",
        help="Preprocessed plan configuration subfolder (default nnUNetPlans_3d_fullres).",
    )

    p_bfv3 = sub.add_parser(
        "build-boundary-fragment-v3-sidecars",
        help=(
            "Build BoundaryFragment V3 (5-class) target sidecars under "
            "<preprocessed>/<plan>/boundary_fragment_v3_targets/."
        ),
    )
    p_bfv3.add_argument("--dataset", type=int, default=537, help="nnUNet dataset ID (default 537).")
    p_bfv3.add_argument("--force", action="store_true", help="Overwrite existing sidecars.")
    p_bfv3.add_argument(
        "--case-subset",
        nargs="*",
        default=None,
        help="Optional list of source case IDs (zero-padded) to restrict the build.",
    )
    p_bfv3.add_argument(
        "--plan-configuration",
        default="nnUNetPlans_3d_fullres",
        help="Preprocessed plan configuration subfolder (default nnUNetPlans_3d_fullres).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "build-instance-sidecars":
        result = build_bicm_v5_instance_sidecars(
            ds_id=int(args.dataset),
            force=bool(args.force),
            case_subset=args.case_subset,
            plan_configuration=args.plan_configuration,
        )
        print(json.dumps({
            "dataset_id": result["dataset_id"],
            "output_dir": result["output_dir"],
            "written": result["written"],
            "skipped": result["skipped"],
            "audit_path": result["audit_path"],
        }, indent=2))
        return 0
    if args.cmd == "build-boundary-fragment-v3-sidecars":
        result = generate_boundary_fragment_v3_target_sidecars(
            ds_id=int(args.dataset),
            force=bool(args.force),
            case_subset=args.case_subset,
            plan_configuration=args.plan_configuration,
        )
        print(json.dumps({
            "dataset_id": result["dataset_id"],
            "output_dir": result["output_dir"],
            "written": result["written"],
            "skipped": result["skipped"],
            "audit_path": result["audit_path"],
        }, indent=2))
        return 0
    parser.error(f"unknown command: {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

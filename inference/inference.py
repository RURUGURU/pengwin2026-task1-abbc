"""PENGWIN 2026 Task 1 -- V0 pipeline-test inference entrypoint.

Reads the input CT MHA from
    /input/images/peripelvic-fracture-ct/
and writes a fragment-instance label map MHA to
    /output/images/peripelvic-fracture-ct-segmentation/

Anatomy routing (spec):
    pelvic -> run V291 ABBC bw=10 fold_0 + V288 core-seed watershed decoder
    femur  -> ALL-ZERO output (UNHANDLED in V0)

This is a V0 pipeline-test, NOT a competitive scientific submission.
On any unhandled failure the entrypoint still writes an all-zero MHA so
Grand Challenge does not record a crashed case.

[pengwin_v0] prefix is used for all stdout/stderr.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


INPUT_DIR = Path("/input/images/peripelvic-fracture-ct")
OUTPUT_DIR = Path("/output/images/peripelvic-fracture-ct-segmentation")

# Model dir on Grand Challenge: the model.tar.gz is packed with the trailing-dot
# convention (`tar -C model_payload -czf model.tar.gz .`) so its contents are
# extracted directly under /opt/ml/model/ (no `model_payload/` prefix).
MODEL_ROOT = Path(os.environ.get("PENGWIN_MODEL_ROOT", "/opt/ml/model"))
NN_RES = Path(os.environ.get("nnUNet_results", str(MODEL_ROOT / "nnunet" / "results")))
NN_PREP = Path(os.environ.get("nnUNet_preprocessed", str(MODEL_ROOT / "nnunet" / "preprocessed")))
NN_RAW = Path(os.environ.get("nnUNet_raw", str(MODEL_ROOT / "nnunet" / "raw")))

# Set the env vars early so nnU-Net does not print "is not defined" warnings on import.
os.environ.setdefault("nnUNet_results", str(NN_RES))
os.environ.setdefault("nnUNet_preprocessed", str(NN_PREP))
os.environ.setdefault("nnUNet_raw", str(NN_RAW))
os.environ.setdefault("PENGWIN_ROOT", str(MODEL_ROOT))

DATASET_NAME = "Dataset537_PelvicBICMFragmentV5"
TRAINER = (
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025"
    "StrongPeakNoContactABBCV291"
)
PLANS = "nnUNetResEncUNetLPlans"
CONFIG = "3d_fullres"
FOLD = 0
CHECKPOINT_NAME = "checkpoint_best.pth"
ABBC_OUTPUT_CHANNELS = 4

# V0 post-processing config (matches the bw=10 + sr=0.05 row reported in
# description.md and Comment.txt):
ROI_PAD_VOX = 24
BACKGROUND_THRESHOLD = 0.50
CORE_THRESHOLD = 0.50
ANATOMY_SIZE_RATIO_KEEP = 0.05
# 1 cm^3 / (0.5496 mm^3/voxel) ~= 1820 voxels at the V5 plan resolution.
MIN_COMPONENT_VOXELS = 1820

# Anatomy ranges per PENGWIN 2026 official spec.
PELVIC_RANGES = {
    "Sacrum": (1, 50),
    "LeftHip": (51, 100),
    "RightHip": (101, 150),
}
FEMUR_RANGE = (151, 200)  # V0: never emitted.


def log(msg: str) -> None:
    print(f"[pengwin_v0] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Anatomy routing rule (verbatim from PENGWIN 2026 Task 1 spec).
# ---------------------------------------------------------------------------
def classify_pelvic_femur(spacing_x, spacing_y, spacing_z, physical_x_mm, physical_z_mm):
    if physical_x_mm <= 285.35:
        if spacing_x <= 0.71:
            return "pelvic"
        elif spacing_z <= 0.90:
            return "femur"
        else:
            return "pelvic" if spacing_y <= 0.91 else "femur"
    else:
        if spacing_z <= 0.68:
            return "pelvic" if physical_z_mm <= 193.55 else "femur"
        else:
            return "pelvic" if physical_z_mm <= 390.78 else "femur"


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------
def find_input_image() -> Path:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"input directory missing: {INPUT_DIR}")
    candidates = sorted(p for p in INPUT_DIR.iterdir() if p.suffix in {".mha", ".mhd"})
    if not candidates:
        raise FileNotFoundError(f"no .mha input under {INPUT_DIR}; got {list(INPUT_DIR.iterdir())}")
    return candidates[0]


def output_path(input_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Grand Challenge expects the output filename to mirror the input (per
    # `peripelvic-fracture-ct-segmentation`). Keep the same stem + .mha suffix.
    return OUTPUT_DIR / input_path.name


def write_label_map(label_arr: np.ndarray, ref_img: sitk.Image, out_path: Path) -> None:
    """Write a label MHA preserving the reference image's spatial metadata."""
    if label_arr.dtype != np.uint16:
        label_arr = label_arr.astype(np.uint16, copy=False)
    out_img = sitk.GetImageFromArray(label_arr)
    out_img.SetSpacing(ref_img.GetSpacing())
    out_img.SetOrigin(ref_img.GetOrigin())
    out_img.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out_img, str(out_path), useCompression=True)


def write_zero_output(input_path: Path, ref_img: sitk.Image, reason: str) -> None:
    out_path = output_path(input_path)
    zero = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint16)
    write_label_map(zero, ref_img, out_path)
    log(f"wrote ALL-ZERO output ({reason}) -> {out_path}")


# ---------------------------------------------------------------------------
# nnU-Net prediction (full-CT single-pass).
# ---------------------------------------------------------------------------
def build_predictor():
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    log(f"build_predictor: device={device} cuda_available={use_cuda}")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=use_cuda,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    model_dir = NN_RES / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"trained model folder missing: {model_dir}")
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=(int(FOLD),),
        checkpoint_name=CHECKPOINT_NAME,
    )
    return predictor


def predict_abbc_logits_full_ct(predictor, image_npy: np.ndarray, image_properties: dict) -> tuple[np.ndarray, dict]:
    """Run the V291 ABBC model on the full CT and return 4-channel logits + props.

    Returns (logits_zyx, data_properties) where logits has shape (4, Zp, Yp, Xp)
    at nnU-Net's preprocessed grid (NOT the original CT grid).
    """
    import torch
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy

    # PreprocessAdapterFromNpy expects a list of [C, Z, Y, X] arrays; nnU-Net
    # convention adds the channel axis at the front. The Dataset537 plan has
    # one image channel (CT) so we just unsqueeze on axis 0.
    if image_npy.ndim == 3:
        image_npy_4d = image_npy[None].astype(np.float32, copy=False)
    elif image_npy.ndim == 4:
        image_npy_4d = image_npy.astype(np.float32, copy=False)
    else:
        raise ValueError(f"expected 3D or [C,Z,Y,X] input, got {image_npy.shape}")

    ppa = PreprocessAdapterFromNpy(
        [image_npy_4d],
        [None],
        [image_properties],
        [None],
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        num_threads_in_multithreaded=1,
        verbose=False,
    )
    dct = next(ppa)
    data = dct["data"]
    if torch.is_tensor(data):
        data_tensor = data.float()
    else:
        data_tensor = torch.from_numpy(np.asarray(data)).float()

    # Custom 4-channel sliding-window export (mirrors
    # _predict_custom_logits_from_preprocessed_data in code_task1/eval.py).
    logits = _predict_custom_logits_from_preprocessed_data(
        predictor, data_tensor, output_channels=ABBC_OUTPUT_CHANNELS,
    )
    logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
    return logits_np, dct["data_properties"]


def _predict_custom_logits_from_preprocessed_data(predictor, data, output_channels: int):
    """Sliding-window inference that allocates `output_channels` for a custom head.

    Mirrors `code_task1/eval.py::_predict_custom_logits_from_preprocessed_data`
    but trimmed for V0 (no `allowed` check; trust the ABBC trainer to emit 4).
    """
    import torch
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from nnunetv2.utilities.helpers import dummy_context, empty_cache

    def _internal_predict(data_pad, slicers, do_on_device):
        results_device = predictor.device if do_on_device else torch.device("cpu")
        empty_cache(predictor.device)
        data_work = data_pad.to(results_device)
        predicted_logits = torch.zeros(
            (int(output_channels), *data_work.shape[1:]),
            dtype=torch.half, device=results_device,
        )
        n_predictions = torch.zeros(
            data_work.shape[1:], dtype=torch.half, device=results_device,
        )
        gaussian = compute_gaussian(
            tuple(predictor.configuration_manager.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=results_device,
        ) if predictor.use_gaussian else 1
        for sl in slicers:
            workon = data_work[sl][None].to(predictor.device)
            prediction = predictor._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
            if int(prediction.shape[0]) != int(output_channels):
                raise RuntimeError(
                    "Custom nnU-Net head mismatch: "
                    f"expected {output_channels} channels, got {tuple(prediction.shape)}"
                )
            if predictor.use_gaussian:
                prediction *= gaussian
            predicted_logits[sl] += prediction
            n_predictions[sl[1:]] += gaussian
        predicted_logits /= n_predictions
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError("inf in custom-head predicted logits")
        return predicted_logits

    with torch.no_grad():
        if data.ndim != 4:
            raise ValueError(f"expected [C, Z, Y, X], got {tuple(data.shape)}")
        predictor.network = predictor.network.to(predictor.device)
        predictor.network.eval()
        empty_cache(predictor.device)
        autocast_ctx = (
            torch.autocast(predictor.device.type, enabled=True)
            if predictor.device.type == "cuda" else dummy_context()
        )
        with autocast_ctx:
            data_pad, slicer_revert = pad_nd_image(
                data,
                predictor.configuration_manager.patch_size,
                "constant", {"value": 0}, True, None,
            )
            slicers = predictor._internal_get_sliding_window_slicers(data_pad.shape[1:])
            do_on_device = (
                predictor.perform_everything_on_device
                and predictor.device.type != "cpu"
            )
            try:
                predicted_logits = _internal_predict(data_pad, slicers, do_on_device)
            except RuntimeError as exc:
                log(f"on-device predict failed; retrying with CPU buffer: {exc}")
                empty_cache(predictor.device)
                predicted_logits = _internal_predict(data_pad, slicers, False)
            empty_cache(predictor.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
    return predicted_logits


# ---------------------------------------------------------------------------
# ABBC -> instance fragment decoder (V288 core-seed watershed).
# ---------------------------------------------------------------------------
def softmax_4ch(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != ABBC_OUTPUT_CHANNELS:
        raise ValueError(f"expected [4, Z, Y, X] logits, got {arr.shape}")
    arr = arr - np.max(arr, axis=0, keepdims=True)
    exp = np.exp(arr).astype(np.float32, copy=False)
    return exp / np.maximum(np.sum(exp, axis=0, keepdims=True), np.float32(1e-8))


def decode_abbc_core_seed_watershed(
    probs: np.ndarray,
    *,
    background_threshold: float = BACKGROUND_THRESHOLD,
    core_threshold: float = CORE_THRESHOLD,
    min_component_voxels: int = MIN_COMPONENT_VOXELS,
    size_ratio_keep: float = ANATOMY_SIZE_RATIO_KEEP,
) -> np.ndarray:
    """Mirror `decode_task1_v288_abbc` from code_task1/eval.py.

    Channels: 0=background, 1=border, 2=boundary, 3=core (softmax).
    Returns a uint16 instance map. Labels are 1..N within this single ROI.
    """
    import scipy.ndimage as ndi

    background = probs[0] >= float(background_threshold)
    support = ~background
    if not support.any():
        return np.zeros(probs.shape[1:], dtype=np.uint16)
    core = (probs[3] >= float(core_threshold)) & support
    core_labels, n_core = ndi.label(core, structure=np.ones((3, 3, 3), dtype=bool))
    core_labels = core_labels.astype(np.int32, copy=False)
    if int(n_core) <= 0:
        labels = np.where(support, 1, 0).astype(np.int32, copy=False)
    else:
        from skimage.segmentation import watershed as _ski_watershed
        priority = ndi.distance_transform_edt(core_labels == 0)
        labels = _ski_watershed(
            priority, markers=core_labels, mask=support,
        ).astype(np.int32, copy=False)
        fill_mask = support & (labels == 0)
        if fill_mask.any():
            nearest_indices = ndi.distance_transform_edt(
                labels == 0, return_indices=True,
            )[1]
            labels[fill_mask] = labels[
                tuple(idx[fill_mask] for idx in nearest_indices)
            ]
    decoded = _merge_small_components(labels, min_component_voxels=int(min_component_voxels))
    if float(size_ratio_keep) > 0.0:
        decoded = _merge_by_size_ratio(decoded, size_ratio_keep=float(size_ratio_keep))
    return decoded.astype(np.uint16, copy=False)


def _merge_small_components(labels: np.ndarray, *, min_component_voxels: int) -> np.ndarray:
    """Merge components smaller than `min_component_voxels` into the nearest larger one.

    Simplified version of `_task1_v277_merge_small_components`: drop components
    below threshold, then nearest-neighbor fill the resulting holes from the
    remaining large components.
    """
    import scipy.ndimage as ndi

    out = np.asarray(labels, dtype=np.int32).copy()
    if int(min_component_voxels) <= 1:
        return out
    ids, counts = np.unique(out, return_counts=True)
    small_ids = [int(i) for i, c in zip(ids, counts) if int(i) > 0 and int(c) < int(min_component_voxels)]
    if not small_ids:
        return out
    small_mask = np.isin(out, small_ids)
    out[small_mask] = 0
    if (out == 0).any() and (out > 0).any():
        # Refill original support with nearest-large-component IDs.
        original_support = labels > 0
        # Only refill voxels that were originally support but now empty.
        refill_mask = original_support & (out == 0)
        if refill_mask.any():
            non_zero = out > 0
            if non_zero.any():
                nearest = ndi.distance_transform_edt(~non_zero, return_indices=True)[1]
                out[refill_mask] = out[tuple(idx[refill_mask] for idx in nearest)]
    return out


def _merge_by_size_ratio(labels: np.ndarray, *, size_ratio_keep: float) -> np.ndarray:
    """Drop components smaller than size_ratio_keep * largest, refill via nearest.

    Approximates `_task1_v288_anatomy_size_ratio_merge` for single-ROI decoding.
    """
    import scipy.ndimage as ndi

    out = np.asarray(labels, dtype=np.int32).copy()
    ids, counts = np.unique(out, return_counts=True)
    pairs = [(int(i), int(c)) for i, c in zip(ids, counts) if int(i) > 0]
    if len(pairs) <= 1:
        return out
    largest = max(c for _, c in pairs)
    threshold = float(size_ratio_keep) * float(largest)
    small_ids = [i for i, c in pairs if float(c) < threshold]
    if not small_ids:
        return out
    refill_mask = np.isin(out, small_ids)
    out[refill_mask] = 0
    non_zero = out > 0
    if non_zero.any() and refill_mask.any():
        nearest = ndi.distance_transform_edt(~non_zero, return_indices=True)[1]
        out[refill_mask] = out[tuple(idx[refill_mask] for idx in nearest)]
    return out


# ---------------------------------------------------------------------------
# Resample a per-instance label map from preprocessed grid back to original.
# ---------------------------------------------------------------------------
def resample_label_map_to_original(
    label_pp: np.ndarray, data_properties: dict, predictor,
) -> np.ndarray:
    """Resample uint16 instance map at preprocessed grid to original CT grid.

    Uses nearest-neighbor resampling (`is_seg=True`) so fragment IDs are not
    interpolated. Mirrors the resample + cropping-bbox restore + transpose
    sequence in `nnunetv2.inference.export_prediction.export_prediction_from_logits`.
    """
    from nnunetv2.preprocessing.resampling.default_resampling import (
        resample_data_or_seg_to_shape,
    )

    # 4-D for nnU-Net: [C, x, y, z]; we only have one channel (label).
    label_in = label_pp.astype(np.uint16, copy=False)[None]

    shape_after_cropping = tuple(
        int(s) for s in data_properties["shape_after_cropping_and_before_resampling"]
    )
    current_spacing = predictor.configuration_manager.spacing
    # original_spacing in plans axis order:
    transpose_forward = predictor.plans_manager.transpose_forward
    original_spacing = [data_properties["spacing"][i] for i in transpose_forward]

    resampled = resample_data_or_seg_to_shape(
        label_in.astype(np.float32, copy=False),
        shape_after_cropping,
        current_spacing,
        original_spacing,
        is_seg=True,
        order=1,
        order_z=0,
    )
    resampled_np = np.asarray(resampled).astype(np.uint16, copy=False)[0]

    # Place into shape_before_cropping at bbox_used_for_cropping.
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

    shape_before_cropping = tuple(
        int(s) for s in data_properties["shape_before_cropping"]
    )
    bbox = data_properties.get("bbox_used_for_cropping", None)
    if bbox is None:
        full = resampled_np
    else:
        full = np.zeros(shape_before_cropping, dtype=np.uint16)
        slicer = bounding_box_to_slice(bbox)
        full[slicer] = resampled_np

    # Undo plans.transpose_forward (i.e. apply transpose_backward) so axes
    # match the SimpleITK array order ([Z, Y, X]).
    transpose_backward = predictor.plans_manager.transpose_backward
    full = np.transpose(full, tuple(transpose_backward))
    return full


# ---------------------------------------------------------------------------
# Top-level pelvic pipeline.
# ---------------------------------------------------------------------------
def run_pelvic(image_path: Path, ref_img: sitk.Image) -> np.ndarray:
    """Run the V291 ABBC pipeline on a pelvic CT and return a uint16 instance map.

    V0 simplification: Dataset537 was trained on per-anatomy ROI crops, but the
    container runs a single full-CT pass. We split fragments between the three
    PENGWIN pelvic ranges using a coarse physical-x centroid rule (LPS:
    +x = patient left), with the central band assigned to Sacrum. This is a
    heuristic and is acceptable for V0 (pipeline test); the description notes
    that a proper anatomy router is planned for the next submission.
    """
    predictor = build_predictor()

    image_npy = sitk.GetArrayFromImage(ref_img).astype(np.float32, copy=False)
    spacing_xyz = ref_img.GetSpacing()  # (sx, sy, sz)
    # nnU-Net expects spacing in (z, y, x) order matching the numpy array.
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    image_properties = {"spacing": spacing_zyx}

    log(f"pelvic predict: image shape={image_npy.shape} spacing_zyx={spacing_zyx}")
    logits_pp, data_props = predict_abbc_logits_full_ct(predictor, image_npy, image_properties)
    log(f"pelvic predict: logits shape (preprocessed) = {logits_pp.shape}")

    # Decode at the preprocessed grid (where the model is most confident), then
    # nearest-neighbor resample the integer label map back to the original CT
    # grid. This avoids per-channel resampling artefacts on the watershed seeds.
    probs = softmax_4ch(logits_pp)
    decoded_pp = decode_abbc_core_seed_watershed(probs)
    n_frag = int(len([v for v in np.unique(decoded_pp) if int(v) > 0]))
    log(f"pelvic decode: {n_frag} fragments after post-process (preprocessed grid)")

    decoded = resample_label_map_to_original(decoded_pp, data_props, predictor)
    log(f"pelvic resample: label map shape (original) = {decoded.shape}")

    # Split fragments into Sacrum / LeftHip / RightHip by physical-x centroid.
    # decoded is [Z, Y, X] in SimpleITK array order; index axis 2 corresponds
    # to the +x direction. In LPS, +x means patient left (LeftHip).
    remapped = _split_pelvic_by_x_centroid(decoded)
    return remapped


def _split_pelvic_by_x_centroid(decoded: np.ndarray) -> np.ndarray:
    """V0 heuristic anatomy router: split fragments by x-centroid into Sacrum /
    LeftHip / RightHip ranges. Center band [40 %, 60 %] of the volume's x extent
    is Sacrum (1..50); fragments with x-centroid > 60 % are LeftHip (51..100);
    fragments with x-centroid < 40 % are RightHip (101..150).
    """
    import scipy.ndimage as ndi

    if decoded.ndim != 3:
        raise ValueError(f"expected [Z, Y, X] label map, got {decoded.shape}")
    x_dim = float(decoded.shape[2])
    if x_dim <= 0:
        return decoded.astype(np.uint16, copy=False)
    ids = sorted(int(v) for v in np.unique(decoded) if int(v) > 0)
    if not ids:
        return decoded.astype(np.uint16, copy=False)
    # Use centroid via center_of_mass for each ID. Coordinates are (Z, Y, X).
    centroids = ndi.center_of_mass(np.ones_like(decoded, dtype=np.uint8), decoded, ids)
    if len(ids) == 1:
        centroids = [centroids]
    remapped = np.zeros_like(decoded, dtype=np.uint16)
    bucket_counters = {"Sacrum": 0, "LeftHip": 0, "RightHip": 0}
    for old_id, (_cz, _cy, cx) in zip(ids, centroids):
        if not np.isfinite(cx):
            anat = "Sacrum"
        else:
            frac = float(cx) / x_dim
            if frac >= 0.60:
                anat = "LeftHip"
            elif frac <= 0.40:
                anat = "RightHip"
            else:
                anat = "Sacrum"
        lo, hi = PELVIC_RANGES[anat]
        bucket_counters[anat] += 1
        new_id = lo + bucket_counters[anat] - 1
        if new_id > hi:
            new_id = hi  # collapse overflow into the last slot.
        remapped[decoded == int(old_id)] = np.uint16(new_id)
    log(
        "pelvic split: "
        f"Sacrum={bucket_counters['Sacrum']} "
        f"LeftHip={bucket_counters['LeftHip']} "
        f"RightHip={bucket_counters['RightHip']}"
    )
    return remapped


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    log(f"start: model_root={MODEL_ROOT} nn_results={NN_RES}")
    try:
        image_path = find_input_image()
        log(f"input image: {image_path}")
        ref_img = sitk.ReadImage(str(image_path))

        size = ref_img.GetSize()       # (X, Y, Z)
        spacing = ref_img.GetSpacing() # (sx, sy, sz)
        spacing_x, spacing_y, spacing_z = (float(s) for s in spacing)
        physical_x_mm = float(size[0]) * spacing_x
        physical_z_mm = float(size[2]) * spacing_z
        log(
            "spatial: "
            f"size={size} spacing=({spacing_x:.4f},{spacing_y:.4f},{spacing_z:.4f}) "
            f"physical_x_mm={physical_x_mm:.2f} physical_z_mm={physical_z_mm:.2f}"
        )
        route = classify_pelvic_femur(
            spacing_x, spacing_y, spacing_z, physical_x_mm, physical_z_mm,
        )
        log(f"anatomy routing: {route}")

        out_path = output_path(image_path)
        if route == "femur":
            # V0 short-circuit: femur is unmodeled.
            write_zero_output(image_path, ref_img, "femur route unmodeled in V0")
        else:
            label_arr = run_pelvic(image_path, ref_img)
            write_label_map(label_arr, ref_img, out_path)
            log(f"wrote pelvic prediction -> {out_path}")
    except Exception as exc:
        # Never crash the container -- write an all-zero MHA so Grand Challenge
        # records a (poor) score instead of a failed run.
        log(f"FATAL: {exc}")
        traceback.print_exc()
        try:
            image_path = find_input_image()
            ref_img = sitk.ReadImage(str(image_path))
            write_zero_output(image_path, ref_img, f"fallback after exception: {exc}")
        except Exception as fallback_exc:
            log(f"fallback also failed: {fallback_exc}")
            traceback.print_exc()
            return 1
    finally:
        dt = time.time() - t0
        log(f"done in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

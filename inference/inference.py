"""PENGWIN 2026 Task 1 — V0.2 inference entrypoint.

V0.2 fixes the V0.1 contract mismatch (model expected 3-channel anatomy-aware
input, container fed 1-channel full CT). The new pipeline matches the V5
training recipe verbatim:

    CT -> canonicalize LPS -> bone HU clip [-1000, 2000] -> bone-LUT normalize
      -> Dataset532 anatomy classifier (full CT, 4-channel softmax)
      -> for each anatomy in [Sacrum, LeftHip, RightHip]:
           bbox = pad(anatomy_prob >= 0.5, pad_vox=24)
           stack 3 channels = [CT_LUT_crop, prob_crop, SDF(prob_crop)]
           V291 predict (ABBC 4-channel)
           decode (core-seed watershed)
           relabel into PENGWIN range [lo..hi]
           paste into full label volume
      -> Femur: zeros (V0 still does not model femur)
      -> write uint8 label MHA

Layout (mirrors the model.tar.gz unpacked tree):
    /opt/ml/model/nnunet/results/Dataset532_PelvicAnatomyV2/
        PengwinTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth
    /opt/ml/model/nnunet/results/Dataset537_PelvicBICMFragmentV5/
        PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025
        StrongPeakNoContactABBCV291__nnUNetResEncUNetLPlans__3d_fullres/
        fold_0/checkpoint_best.pth
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
os.environ.setdefault("nnUNet_results", str(NN_RES))
os.environ.setdefault("nnUNet_preprocessed", str(NN_PREP))
os.environ.setdefault("nnUNet_raw", str(NN_RAW))
os.environ.setdefault("PENGWIN_ROOT", str(MODEL_ROOT))

# --- Dataset532: anatomy classifier (background/Sacrum/LeftHip/RightHip). ---
DS532_DATASET = "Dataset532_PelvicAnatomyV2"
DS532_TRAINER = "PengwinTrainer"
DS532_PLANS = "nnUNetResEncUNetLPlans"
DS532_CONFIG = "3d_fullres"
DS532_FOLD = 0
DS532_OUTPUT_CHANNELS = 4  # bg, Sacrum, LeftHip, RightHip

# --- Dataset537 V291: per-anatomy ABBC instance segmenter. ---
V291_DATASET = "Dataset537_PelvicBICMFragmentV5"
V291_TRAINER = (
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025"
    "StrongPeakNoContactABBCV291"
)
V291_PLANS = "nnUNetResEncUNetLPlans"
V291_CONFIG = "3d_fullres"
V291_FOLD = 0
V291_OUTPUT_CHANNELS = 4  # ABBC: bg, border, boundary, core
CHECKPOINT_NAME = "checkpoint_best.pth"

# --- V5 anatomy contract (matches dataset.json v5_contract). ---
V5_DATASET532_PROB_CHANNEL = {"Sacrum": 1, "LeftHip": 2, "RightHip": 3}
V5_ANATOMY_RANGES = {
    "Sacrum": (1, 50),
    "LeftHip": (51, 100),
    "RightHip": (101, 150),
}
ROI_PAD_VOX = 24
ANATOMY_PROB_THRESHOLD = 0.50
SDF_CLIP_MM = 40.0
BONE_HU_RANGE = (-1000.0, 2000.0)

# --- ABBC decode hyperparams (mirror V0.1 description.md "bw=10 + sr=0.05"). ---
BACKGROUND_THRESHOLD = 0.50
CORE_THRESHOLD = 0.50
ANATOMY_SIZE_RATIO_KEEP = 0.05
MIN_COMPONENT_VOXELS = 1820  # ~1 cm^3 at the V5 plan resolution.

# --- Anatomy routing (verbatim from PENGWIN 2026 Task 1 spec). ---
FEMUR_RANGE = (151, 200)


def log(msg: str) -> None:
    print(f"[pengwin_v0] {msg}", flush=True)


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
    return OUTPUT_DIR / input_path.name


def write_label_map(label_arr: np.ndarray, ref_img: sitk.Image, out_path: Path) -> None:
    """Write a label MHA preserving ref_img geometry; clip to [0,200] uint8."""
    if label_arr.ndim != 3:
        raise ValueError(f"expected [Z, Y, X] label map, got {label_arr.shape}")
    label_arr = np.clip(label_arr, 0, 200).astype(np.uint8, copy=False)
    out_img = sitk.GetImageFromArray(label_arr)
    out_img.SetSpacing(ref_img.GetSpacing())
    out_img.SetOrigin(ref_img.GetOrigin())
    out_img.SetDirection(ref_img.GetDirection())
    uniq = np.unique(label_arr)
    log(
        f"write_label_map: dtype={label_arr.dtype} shape={label_arr.shape} "
        f"n_unique={len(uniq)} unique_head={uniq[:10].tolist()} -> {out_path}"
    )
    sitk.WriteImage(out_img, str(out_path), useCompression=False)


def write_zero_output(input_path: Path, ref_img: sitk.Image, reason: str) -> None:
    out_path = output_path(input_path)
    zero = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint8)
    write_label_map(zero, ref_img, out_path)
    log(f"wrote ALL-ZERO output ({reason}) -> {out_path}")


# ---------------------------------------------------------------------------
# Image preprocessing helpers (copied verbatim from preprocessing.py / utils.py
# so the container has no dependency on /opt/app/code_task1 at runtime).
# ---------------------------------------------------------------------------
def orientation_code(img: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())


def canonicalize_sitk(img: sitk.Image, target: str = "LPS") -> sitk.Image:
    code = orientation_code(img)
    if code == target:
        return img
    return sitk.DICOMOrient(img, target)


def bone_window_clip(arr: np.ndarray,
                     hu_low: float = BONE_HU_RANGE[0],
                     hu_high: float = BONE_HU_RANGE[1]) -> np.ndarray:
    return np.clip(arr.astype(np.float32, copy=False), float(hu_low), float(hu_high))


def canonicalize_and_clip_image(img: sitk.Image) -> tuple[sitk.Image, np.ndarray]:
    img_lps = canonicalize_sitk(img, target="LPS")
    arr = bone_window_clip(sitk.GetArrayFromImage(img_lps))
    return img_lps, arr


def bone_lut_normalize(arr: np.ndarray) -> np.ndarray:
    """Deterministic bone-window LUT (verbatim from preprocessing._bone_lut_normalize)."""
    x = arr.astype(np.float32, copy=False)
    xp = np.asarray([-1000.0, -200.0, 150.0, 700.0, 1500.0, 2000.0], dtype=np.float32)
    fp = np.asarray([0.0, 0.05, 0.35, 0.70, 0.92, 1.0], dtype=np.float32)
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp).astype(np.float32)


def ndi_distance_transform(mask: np.ndarray,
                           spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.distance_transform_edt(mask, sampling=spacing_zyx).astype(np.float32, copy=False)


def selected_anatomy_sdf_from_prob(prob: np.ndarray,
                                   spacing_zyx: tuple[float, float, float],
                                   clip_mm: float = SDF_CLIP_MM) -> np.ndarray:
    """Signed distance to anatomy prob>=0.5 mask, clipped/scaled to [-1, 1]."""
    arr = np.asarray(prob, dtype=np.float32)
    mask = arr >= 0.5
    if not mask.any():
        return np.zeros_like(arr, dtype=np.float32)
    inside = ndi_distance_transform(mask, spacing_zyx)
    outside = ndi_distance_transform(~mask, spacing_zyx)
    sdf = np.clip(inside - outside, -float(clip_mm), float(clip_mm)) / float(clip_mm)
    return sdf.astype(np.float32, copy=False)


def bbox_from_mask(mask: np.ndarray, pad_vox: int = ROI_PAD_VOX) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    pad = int(max(0, pad_vox))
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, np.asarray(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# nnU-Net predictor utilities (generic over output channels).
# ---------------------------------------------------------------------------
def build_predictor(dataset: str, trainer: str, plans: str, config: str,
                    fold: int, checkpoint_name: str = CHECKPOINT_NAME):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
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
    model_dir = NN_RES / dataset / f"{trainer}__{plans}__{config}"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"trained model folder missing: {model_dir}")
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=(int(fold),),
        checkpoint_name=checkpoint_name,
    )
    log(f"build_predictor: {dataset} {trainer[-30:]} device={device}")
    return predictor


def predict_logits_full(predictor, image_4d: np.ndarray,
                        image_properties: dict, output_channels: int) -> tuple[np.ndarray, dict]:
    """Run sliding-window inference on a (C, Z, Y, X) array; return logits + data_props.

    The number of input channels in image_4d MUST match the trained model's
    plans.channel_names; otherwise the encoder will reject the input.
    """
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy

    if image_4d.ndim != 4:
        raise ValueError(f"expected (C, Z, Y, X) input, got {image_4d.shape}")
    image_4d = image_4d.astype(np.float32, copy=False)
    ppa = PreprocessAdapterFromNpy(
        [image_4d],
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
    import torch
    if torch.is_tensor(data):
        data_tensor = data.float()
    else:
        data_tensor = torch.from_numpy(np.asarray(data)).float()
    logits = _predict_custom_logits_from_preprocessed_data(
        predictor, data_tensor, output_channels=output_channels,
    )
    logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
    return logits_np, dct["data_properties"]


def _predict_custom_logits_from_preprocessed_data(predictor, data, output_channels: int):
    """Sliding-window inference that allocates `output_channels` for a custom head."""
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


def softmax_axis0(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    arr = arr - np.max(arr, axis=0, keepdims=True)
    exp = np.exp(arr).astype(np.float32, copy=False)
    return exp / np.maximum(np.sum(exp, axis=0, keepdims=True), np.float32(1e-8))


# ---------------------------------------------------------------------------
# ABBC -> instance fragment decoder (V288 core-seed watershed).
# ---------------------------------------------------------------------------
def decode_abbc_core_seed_watershed(
    probs: np.ndarray,
    *,
    background_threshold: float = BACKGROUND_THRESHOLD,
    core_threshold: float = CORE_THRESHOLD,
    min_component_voxels: int = MIN_COMPONENT_VOXELS,
    size_ratio_keep: float = ANATOMY_SIZE_RATIO_KEEP,
) -> np.ndarray:
    """Channels: 0=background, 1=border, 2=boundary, 3=core. Returns uint16 labels 1..N."""
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
        original_support = labels > 0
        refill_mask = original_support & (out == 0)
        if refill_mask.any():
            non_zero = out > 0
            if non_zero.any():
                nearest = ndi.distance_transform_edt(~non_zero, return_indices=True)[1]
                out[refill_mask] = out[tuple(idx[refill_mask] for idx in nearest)]
    return out


def _merge_by_size_ratio(labels: np.ndarray, *, size_ratio_keep: float) -> np.ndarray:
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
    from nnunetv2.preprocessing.resampling.default_resampling import (
        resample_data_or_seg_to_shape,
    )

    label_in = label_pp.astype(np.uint16, copy=False)[None]

    shape_after_cropping = tuple(
        int(s) for s in data_properties["shape_after_cropping_and_before_resampling"]
    )
    current_spacing = predictor.configuration_manager.spacing
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

    transpose_backward = predictor.plans_manager.transpose_backward
    full = np.transpose(full, tuple(transpose_backward))
    return full


# ---------------------------------------------------------------------------
# Per-anatomy V291 inference pipeline.
# ---------------------------------------------------------------------------
def run_per_anatomy_pelvic(image_path: Path, ref_img: sitk.Image) -> np.ndarray:
    """Full V0.2 pipeline: Ds532 anatomy classifier -> per-anatomy V291 ABBC inference.

    Femur (151-200) is still NOT modeled in V0; this returns a label map sized
    to ref_img with only Sacrum (1-50) / LeftHip (51-100) / RightHip (101-150)
    fragments populated.
    """
    # --- Step 1: canonicalize CT to LPS + bone-window clip + LUT-normalize. ---
    img_lps, arr_clipped = canonicalize_and_clip_image(ref_img)
    image_npy_lps = arr_clipped.astype(np.float32, copy=False)
    ct_lut_full = bone_lut_normalize(arr_clipped)

    spacing_xyz = img_lps.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    log(
        f"pelvic LPS preproc: shape={image_npy_lps.shape} "
        f"spacing_zyx={spacing_zyx} hu_clip=[{BONE_HU_RANGE[0]},{BONE_HU_RANGE[1]}]"
    )

    # --- Step 2: Dataset532 anatomy classifier (full CT, 1-channel HU input). ---
    ds532_predictor = build_predictor(
        DS532_DATASET, DS532_TRAINER, DS532_PLANS, DS532_CONFIG, DS532_FOLD,
    )
    ds532_image_4d = image_npy_lps[None]  # (1, Z, Y, X) raw HU; CTNormalization handles z-score.
    ds532_props = {"spacing": list(spacing_zyx)}
    log(f"ds532 predict: image (C,Z,Y,X)={ds532_image_4d.shape}")
    ds532_logits_pp, ds532_data_props = predict_logits_full(
        ds532_predictor, ds532_image_4d, ds532_props, output_channels=DS532_OUTPUT_CHANNELS,
    )
    ds532_probs_pp = softmax_axis0(ds532_logits_pp)
    log(f"ds532 predict: probs shape (preprocessed) = {ds532_probs_pp.shape}")

    # Resample each anatomy probability channel back to the original CT grid.
    # We re-use the integer label-map resampler with a (prob*255) trick: the
    # resampler does nearest for label_in but actually accepts a float buffer
    # and runs linear order=1 z-order=0 anyway. For probability, we want
    # linear interpolation, so we call a dedicated path that mirrors nnUNet's
    # export_prediction (probability resample is order=1 / order_z=0 linear).
    from nnunetv2.preprocessing.resampling.default_resampling import (
        resample_data_or_seg_to_shape,
    )
    shape_after_cropping = tuple(int(s) for s in ds532_data_props["shape_after_cropping_and_before_resampling"])
    current_spacing = ds532_predictor.configuration_manager.spacing
    transpose_forward = ds532_predictor.plans_manager.transpose_forward
    transpose_backward = ds532_predictor.plans_manager.transpose_backward
    original_spacing = [ds532_data_props["spacing"][i] for i in transpose_forward]
    probs_resampled_axes = resample_data_or_seg_to_shape(
        ds532_probs_pp.astype(np.float32, copy=False),
        shape_after_cropping,
        current_spacing,
        original_spacing,
        is_seg=False,
        order=1,
        order_z=0,
    )  # shape: (C, *shape_after_cropping) in plans axis order.

    # Paste into the full pre-cropping volume.
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    shape_before_cropping = tuple(int(s) for s in ds532_data_props["shape_before_cropping"])
    bbox_used = ds532_data_props.get("bbox_used_for_cropping", None)
    probs_full_axes = np.zeros(
        (DS532_OUTPUT_CHANNELS, *shape_before_cropping), dtype=np.float32,
    )
    if bbox_used is None:
        probs_full_axes = np.asarray(probs_resampled_axes, dtype=np.float32)
    else:
        slicer = bounding_box_to_slice(bbox_used)
        probs_full_axes[(slice(None), *slicer)] = np.asarray(probs_resampled_axes, dtype=np.float32)

    # Set background prob = 1 outside any anatomy where we filled zeros so
    # the channel sum stays ≈ 1 (not strictly required for the downstream
    # thresholding, but safer).
    fg_sum = probs_full_axes[1:].sum(axis=0)
    bg_mask = fg_sum < 1e-6
    if bg_mask.any():
        probs_full_axes[0][bg_mask] = 1.0

    # Undo plans.transpose_forward so axes match the SimpleITK (Z, Y, X) order.
    probs_full = np.transpose(
        probs_full_axes, (0, *(int(a) + 1 for a in transpose_backward)),
    )
    if probs_full.shape[1:] != image_npy_lps.shape:
        raise RuntimeError(
            f"ds532 probs shape {probs_full.shape[1:]} != CT shape {image_npy_lps.shape}"
        )

    # Free Ds532 predictor before loading V291 (T4 has 16 GiB).
    del ds532_predictor, ds532_logits_pp, ds532_probs_pp
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Step 3: V291 per-anatomy inference. ---
    v291_predictor = build_predictor(
        V291_DATASET, V291_TRAINER, V291_PLANS, V291_CONFIG, V291_FOLD,
    )

    full_label = np.zeros(image_npy_lps.shape, dtype=np.uint16)
    for anatomy in ("Sacrum", "LeftHip", "RightHip"):
        prob_channel = V5_DATASET532_PROB_CHANNEL[anatomy]
        anatomy_prob = probs_full[prob_channel]  # (Z, Y, X)
        anatomy_mask = anatomy_prob >= ANATOMY_PROB_THRESHOLD
        bbox = bbox_from_mask(anatomy_mask, pad_vox=ROI_PAD_VOX)
        if bbox is None:
            log(f"[{anatomy}] empty Ds532 anatomy mask (>={ANATOMY_PROB_THRESHOLD}); skipping")
            continue
        ct_lut_crop = ct_lut_full[bbox]
        prob_crop = anatomy_prob[bbox]
        sdf_crop = selected_anatomy_sdf_from_prob(prob_crop, spacing_zyx, clip_mm=SDF_CLIP_MM)
        v291_image_4d = np.stack([ct_lut_crop, prob_crop, sdf_crop], axis=0)
        v291_props = {"spacing": list(spacing_zyx)}
        log(
            f"[{anatomy}] V291 predict: bbox={[(s.start, s.stop) for s in bbox]} "
            f"crop_shape={ct_lut_crop.shape}"
        )
        v291_logits_pp, v291_data_props = predict_logits_full(
            v291_predictor, v291_image_4d, v291_props, output_channels=V291_OUTPUT_CHANNELS,
        )
        v291_probs = softmax_axis0(v291_logits_pp)
        decoded_pp = decode_abbc_core_seed_watershed(v291_probs)
        n_local = int(len([v for v in np.unique(decoded_pp) if int(v) > 0]))
        log(f"[{anatomy}] decode: {n_local} fragments at preprocessed grid")

        # Resample integer label map back to ORIGINAL CROP grid (not full CT).
        decoded_crop = resample_label_map_to_original(
            decoded_pp, v291_data_props, v291_predictor,
        )
        if decoded_crop.shape != ct_lut_crop.shape:
            log(
                f"[{anatomy}] WARNING: resampled crop {decoded_crop.shape} != "
                f"input crop {ct_lut_crop.shape}; using min-overlap paste"
            )

        # Relabel into the PENGWIN range for this anatomy.
        lo, hi = V5_ANATOMY_RANGES[anatomy]
        n_slots = hi - lo + 1
        local_ids = sorted(int(v) for v in np.unique(decoded_crop) if int(v) > 0)
        if not local_ids:
            log(f"[{anatomy}] no fragments after resample; skipping paste")
            continue
        remap = np.zeros(int(max(local_ids)) + 1, dtype=np.uint16)
        for i, old_id in enumerate(local_ids):
            slot = i if i < n_slots else (n_slots - 1)  # collapse overflow into last slot.
            remap[old_id] = np.uint16(lo + slot)
        remapped_crop = remap[decoded_crop]

        # Paste into full label, only over voxels that already 0.
        out_slot = full_label[bbox]
        write_mask = (remapped_crop > 0) & (out_slot == 0)
        out_slot[write_mask] = remapped_crop[write_mask]
        full_label[bbox] = out_slot
        log(
            f"[{anatomy}] painted {int(write_mask.sum())} voxels "
            f"into range [{lo},{hi}] ({len(local_ids)} fragments)"
        )

    # --- Step 4: undo LPS reorientation so output aligns with the ORIGINAL ref_img. ---
    if orientation_code(ref_img) != "LPS":
        # Wrap full_label as a SimpleITK image in the LPS frame, then reorient
        # back to the original orientation code.
        lps_label_img = sitk.GetImageFromArray(full_label.astype(np.uint16, copy=False))
        lps_label_img.SetSpacing(img_lps.GetSpacing())
        lps_label_img.SetOrigin(img_lps.GetOrigin())
        lps_label_img.SetDirection(img_lps.GetDirection())
        target_code = orientation_code(ref_img)
        oriented = sitk.DICOMOrient(lps_label_img, target_code)
        full_label = sitk.GetArrayFromImage(oriented).astype(np.uint16, copy=False)
        log(f"reoriented label map: LPS -> {target_code} (shape={full_label.shape})")
    return full_label


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
            write_zero_output(image_path, ref_img, "femur route unmodeled in V0")
        else:
            label_arr = run_per_anatomy_pelvic(image_path, ref_img)
            write_label_map(label_arr, ref_img, out_path)
            log(f"wrote pelvic prediction -> {out_path}")
    except Exception as exc:
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

"""PENGWIN 2026 Task 1 — V0.2 추론 진입점 (V0.3.1 견고화 버전).

V0.2 는 V0.1 의 contract mismatch 를 수정한 버전이다.
(V0.1 은 모델이 3-channel anatomy-aware 입력을 기대했지만 컨테이너가 1-channel
full CT 를 넣어 주는 문제가 있었다.)
새 파이프라인은 V5 학습 레시피를 그대로 재현한다:

    CT -> LPS 정규화 -> bone HU clip [-1000, 2000] -> bone-LUT 정규화
      -> Dataset532 anatomy classifier 추론 (full CT, 4-channel softmax)
      -> [Sacrum, LeftHip, RightHip] 각각에 대해:
           bbox = pad(anatomy_prob >= 0.5, pad_vox=24)
           3-channel 입력 구성 = [CT_LUT_crop, prob_crop, SDF(prob_crop)]
           V291 추론 (ABBC 4-channel)
           디코딩 (core-seed watershed)
           PENGWIN 라벨 범위 [lo..hi] 로 재매핑
           전체 라벨 볼륨에 paste
      -> Femur: 전부 zero (V0 에서는 femur 모델이 아직 없음)
      -> uint8 라벨 MHA 로 write

모델 파일 레이아웃 (model.tar.gz 해제된 트리 구조):
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
import time as _time  # V0.3 시간 예산 헬퍼에서 사용 (이름 충돌 회피용 alias)
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


INPUT_DIR = Path("/input/images/peripelvic-fracture-ct")
OUTPUT_DIR = Path("/output/images/peripelvic-fracture-ct-segmentation")

# Grand Challenge 의 모델 디렉터리 경로.
# model.tar.gz 는 trailing-dot convention (`tar -C model_payload -czf model.tar.gz .`)
# 으로 패킹되어 있어 내용물이 /opt/ml/model/ 바로 아래로 풀린다 (model_payload/ prefix 없음).
MODEL_ROOT = Path(os.environ.get("PENGWIN_MODEL_ROOT", "/opt/ml/model"))
NN_RES = Path(os.environ.get("nnUNet_results", str(MODEL_ROOT / "nnunet" / "results")))
NN_PREP = Path(os.environ.get("nnUNet_preprocessed", str(MODEL_ROOT / "nnunet" / "preprocessed")))
NN_RAW = Path(os.environ.get("nnUNet_raw", str(MODEL_ROOT / "nnunet" / "raw")))
os.environ.setdefault("nnUNet_results", str(NN_RES))
os.environ.setdefault("nnUNet_preprocessed", str(NN_PREP))
os.environ.setdefault("nnUNet_raw", str(NN_RAW))
os.environ.setdefault("PENGWIN_ROOT", str(MODEL_ROOT))

# --- Dataset532: anatomy classifier (background/Sacrum/LeftHip/RightHip 4-class). ---
DS532_DATASET = "Dataset532_PelvicAnatomyV2"
DS532_TRAINER = "PengwinTrainer"
DS532_PLANS = "nnUNetResEncUNetLPlans"
DS532_CONFIG = "3d_fullres"
DS532_FOLD = 0
DS532_OUTPUT_CHANNELS = 4  # background, Sacrum, LeftHip, RightHip

# --- Dataset537 V291: anatomy 별 ABBC instance segmenter. ---
V291_DATASET = "Dataset537_PelvicBICMFragmentV5"
V291_TRAINER = (
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025"
    "StrongPeakNoContactABBCV291"
)
V291_PLANS = "nnUNetResEncUNetLPlans"
V291_CONFIG = "3d_fullres"
V291_FOLD = 0
V291_OUTPUT_CHANNELS = 4  # ABBC: background, border, boundary, core
CHECKPOINT_NAME = "checkpoint_best.pth"

# --- V5 anatomy contract (dataset.json 의 v5_contract 와 일치). ---
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

# --- ABBC decode 하이퍼파라미터 (V0.1 description.md 의 "bw=10 + sr=0.05" 와 동일). ---
BACKGROUND_THRESHOLD = 0.50
CORE_THRESHOLD = 0.50
ANATOMY_SIZE_RATIO_KEEP = 0.05
MIN_COMPONENT_VOXELS = 1820  # V5 plan 해상도에서 약 1 cm^3 에 해당

# --- anatomy routing (PENGWIN 2026 Task 1 스펙 그대로). ---
FEMUR_RANGE = (151, 200)

# --- V0.3 견고화 관련 플래그 ---
V03_USE_BONE_SKELETON_FALLBACK = True  # Ds532 실패 시 bone HU mask fallback 사용
V03_USE_TIME_BUDGET = True             # 시간 예산 기반 안전망 활성화

# --- V0.3 견고화 관련 상수 ---
# 시간 예산 설정 (Grand Challenge 의 10분 제한을 안전하게 지키기 위함)
TIME_BUDGET_SECONDS = 480.0   # 8분: I/O / decode / paste 여유분 포함
TIME_BUDGET_HARD_LIMIT = 540.0   # 9분: 이 시점 이후로는 남은 anatomy 모두 skip
BONE_HU_THRESHOLD = 200.0
BONE_MIN_COMPONENT_VOXELS = 10000
SANITY_MAX_BBOX_FRACTION = 0.35
SANITY_MAX_POSTPAD_BBOX_FRACTION = 0.50  # pad 적용 후 bbox 가 전체 볼륨의 50% 미만이어야 한다
LARGEST_CC_KEEP_ONLY = True  # 가장 큰 CC 만 유지해 25개 슬롯이 노이즈 fragment 로 채워지는 것 방지
MIN_DS532_CC_VOXELS = 500  # Ds532 mask 의 sparse outlier CC 제거 임계값 (voxels)
DS532_MORPH_OPENING_ITERS = 1  # Ds532 mask 에 적용할 binary_opening 반복 횟수


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


def classify_pelvic_femur_v2(arr_clipped, spacing_zyx,
                             spacing_x, spacing_y, spacing_z,
                             physical_x_mm, physical_z_mm,
                             hu_threshold=BONE_HU_THRESHOLD,
                             min_component_voxels=BONE_MIN_COMPONENT_VOXELS,
                             min_bone_size_voxels=BONE_MIN_COMPONENT_VOXELS):
    """Bone-skeleton 기반 anatomy 라우팅 (V0.4).

    spacing 기반 classify_pelvic_femur 가 case 163/189 처럼 spacing_z=0.80 인
    pelvic small-FOV 케이스를 'femur' 로 잘못 라우팅해 zero output 을 내는 문제를
    해결하기 위해, bone HU mask 로부터 Sacrum/LeftHip/RightHip 가 충분히 추출되면
    'pelvic' 으로 강제한다. 그 외엔 기존 spacing 룰로 fallback.

    Returns:
        tuple[str, dict|None]:
            ('pelvic', bone_masks dict)  — bone-skeleton 분해 성공시
            (spacing-rule-result, None)  — fallback
    """
    try:
        bone_masks = bone_skeleton_anatomy_decomposition(
            arr_clipped, spacing_zyx,
            hu_threshold=hu_threshold,
            min_component_voxels=min_component_voxels,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"classify_pelvic_femur_v2: bone decomposition failed ({exc}); falling back to spacing rule")
        bone_masks = {}

    # 충분한 크기의 hip/sacrum 가 2개 이상 추출됐는지 검사
    candidate_keys = ('Sacrum', 'LeftHip', 'RightHip')
    valid = 0
    for k in candidate_keys:
        m = bone_masks.get(k)
        if m is not None and int(m.sum()) >= int(min_bone_size_voxels):
            valid += 1
    if valid >= 2:
        log(f"classify_pelvic_femur_v2: bone-skeleton detected {valid} anatomies -> route=pelvic")
        return "pelvic", bone_masks

    fallback = classify_pelvic_femur(
        spacing_x, spacing_y, spacing_z, physical_x_mm, physical_z_mm,
    )
    log(f"classify_pelvic_femur_v2: bone-skeleton insufficient ({valid} valid) -> spacing fallback={fallback}")
    return fallback, None


# ---------------------------------------------------------------------------
# 입출력 관련 헬퍼.
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
    """ref_img 의 geometry (spacing/origin/direction) 를 유지하며 라벨 MHA 를 저장한다.

    값은 [0, 200] 으로 clip 후 uint8 로 캐스팅한다 (PENGWIN 라벨 범위 보장).
    """
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
# 이미지 전처리 헬퍼들. preprocessing.py / utils.py 에서 그대로 복사해 왔다.
# 이렇게 함으로써 런타임에 /opt/app/code_task1 에 의존하지 않아도 되도록 한다.
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
    """결정론적 bone-window LUT (preprocessing._bone_lut_normalize 의 동일 구현).

    학습 시 사용한 piecewise-linear HU → [0,1] 매핑을 그대로 재현해 train/inference
    contract 가 어긋나지 않도록 한다.
    """
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
    """anatomy 확률 prob>=0.5 mask 로부터 signed distance 를 계산해 [-1, 1] 범위로 스케일링한다.

    mask 내부는 양수, 외부는 음수가 되어 V291 모델이 위치 정보를 학습된 형태로 받을 수 있다.
    """
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
# nnU-Net predictor 관련 헬퍼 (출력 채널 수에 일반화된 형태).
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
    """(C, Z, Y, X) 배열에 sliding-window 추론을 수행해 logit 과 data_properties 를 반환한다.

    image_4d 의 입력 채널 수는 학습된 모델의 plans.channel_names 와 정확히 일치해야 한다.
    그렇지 않으면 encoder 가 입력을 거부한다.
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
    """custom head 의 출력 채널 수에 맞춰 logit 버퍼를 할당하는 sliding-window 추론.

    nnUNet 기본 구현은 plans.json 의 num_classes 에 의존하지만, 본 프로젝트는
    ABBC 4-channel 헤드처럼 비표준 출력 채널을 사용하므로 output_channels 를
    명시적으로 주입한다.
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


def softmax_axis0(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    arr = arr - np.max(arr, axis=0, keepdims=True)
    exp = np.exp(arr).astype(np.float32, copy=False)
    return exp / np.maximum(np.sum(exp, axis=0, keepdims=True), np.float32(1e-8))


# ---------------------------------------------------------------------------
# ABBC 확률 → fragment instance 라벨 디코더 (V288 core-seed watershed).
# ---------------------------------------------------------------------------
def decode_abbc_core_seed_watershed(
    probs: np.ndarray,
    *,
    background_threshold: float = BACKGROUND_THRESHOLD,
    core_threshold: float = CORE_THRESHOLD,
    min_component_voxels: int = MIN_COMPONENT_VOXELS,
    size_ratio_keep: float = ANATOMY_SIZE_RATIO_KEEP,
) -> np.ndarray:
    """V288 core-seed watershed 로 ABBC 확률맵을 fragment instance 라벨로 디코딩한다.

    ABBC 채널 의미: 0=background, 1=border, 2=boundary, 3=core.
    core 영역을 seed 로 두고 background 가 아닌 영역(support) 전체로 watershed 를 확장한다.
    반환값은 uint16 라벨 1..N (배경은 0).
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
# fragment 단위 라벨 맵을 preprocessed grid 에서 원본 CT 격자로 resampling.
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
# V0.3 견고화된 anatomy ROI 추출 헬퍼 (4-layer 파이프라인용).
# ---------------------------------------------------------------------------
def bone_skeleton_anatomy_decomposition(arr_clipped, spacing_zyx,
                                        hu_threshold=200.0,
                                        min_component_voxels=10000):
    """Layer 1: 학습 데이터 분포에 의존하지 않는 anatomy ROI 분해 (bone HU mask 기반).

    PENGWIN CT 의 골격 부위는 HU > 200 으로 안정적으로 분리 가능하다.
    Ds532 가 OOD (out-of-distribution) 케이스에서 실패하더라도, 본 함수는 항상
    합리적인 Sacrum/LeftHip/RightHip ROI 를 추출할 수 있게 해 주는 안전망이다.

    Args:
        arr_clipped: (Z, Y, X) HU 배열 (이미 [-1000, 2000] 로 clip 된 상태)
        spacing_zyx: physical spacing (z, y, x) mm
        hu_threshold: bone 으로 간주할 최소 HU 값
        min_component_voxels: 너무 작은 connected component 는 무시할 임계값

    Returns:
        dict[str, np.ndarray] — anatomy 별 mask (LPS frame 기준).
        Sacrum 은 중앙 CC, LeftHip 은 +x 쪽, RightHip 은 -x 쪽으로 배정한다.
        추출 실패 시 빈 dict 반환.
    """
    import scipy.ndimage as ndi
    bone_mask = arr_clipped > float(hu_threshold)
    # Morphology 후처리: opening 으로 노이즈 제거하면서 작은 gap 도 함께 정리
    bone_mask = ndi.binary_opening(bone_mask, structure=np.ones((2, 2, 2), dtype=bool), iterations=1)
    labels, n_cc = ndi.label(bone_mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n_cc == 0:
        return {}
    # connected component 를 크기 내림차순으로 정렬
    sizes = ndi.sum(bone_mask, labels, index=np.arange(1, n_cc + 1))
    sorted_idx = np.argsort(sizes)[::-1]
    top_cc_ids = [int(sorted_idx[i]) + 1 for i in range(min(5, n_cc)) if sizes[sorted_idx[i]] >= min_component_voxels]
    if not top_cc_ids:
        return {}
    # 각 CC 의 centroid (z, y, x) 좌표 계산
    centroids_raw = ndi.center_of_mass(bone_mask, labels, top_cc_ids)
    # scipy 버전/입력 개수에 따라 반환 형식이 달라지므로 3-tuple 의 list 로 정규화한다.
    if len(top_cc_ids) == 1:
        # index 가 1개일 때 scipy 가 단일 tuple 을 반환하는 경우가 있어 list 로 감싼다
        if isinstance(centroids_raw, tuple) and len(centroids_raw) == 3 and not isinstance(centroids_raw[0], tuple):
            centroids = [centroids_raw]
        else:
            centroids = list(centroids_raw)
    else:
        centroids = list(centroids_raw)
    # x-centroid 로 좌/중/우 분류 (LPS 좌표계에서는 +x 가 patient left)
    x_dim = float(arr_clipped.shape[2])
    cc_info = []
    for i, cc_id in enumerate(top_cc_ids):
        c = centroids[i]
        if not (hasattr(c, '__len__') and len(c) == 3):
            log(f"bone-skeleton: CC {cc_id} centroid malformed ({c}), skipping")
            continue
        cz, cy, cx = float(c[0]), float(c[1]), float(c[2])
        cc_info.append({
            'cc_id': int(cc_id),
            'cx': float(cx),
            'cy': float(cy),
            'cz': float(cz),
            'size': int(sizes[cc_id - 1]),
        })
    # 크기 내림차순으로 정렬
    cc_info.sort(key=lambda d: -d['size'])

    # 가장 큰 3개 CC 를 사용해 x-centroid 로 anatomy 를 분류한다:
    #   가장 중앙에 가까운 CC = Sacrum
    #   +x 쪽 (이미지 좌측 절반) = LeftHip
    #   -x 쪽 (이미지 우측 절반) = RightHip
    if len(cc_info) >= 3:
        top3 = cc_info[:3]
    else:
        top3 = cc_info  # 3개 미만이면 가능한 만큼만 사용 (best effort)

    # 이미지 중앙 x 좌표
    cx_center = x_dim / 2.0
    # 중앙성 점수 = abs(cx - cx_center). 작을수록 중앙에 가깝다.
    top3_sorted_by_centrality = sorted(top3, key=lambda d: abs(d['cx'] - cx_center))
    if len(top3_sorted_by_centrality) >= 1:
        sacrum_cc = top3_sorted_by_centrality[0]
    else:
        sacrum_cc = None

    remaining = [d for d in top3 if d['cc_id'] != (sacrum_cc['cc_id'] if sacrum_cc else None)]
    if len(remaining) >= 2:
        remaining.sort(key=lambda d: d['cx'])  # x 좌표 오름차순 정렬
        righthip_cc = remaining[0]   # x 가 작은 쪽 = patient right (LPS)
        lefthip_cc = remaining[-1]   # x 가 큰 쪽 = patient left (LPS)
    elif len(remaining) == 1:
        # hip CC 가 1개만 보일 때 centroid 부호를 보고 좌/우 결정
        only_hip = remaining[0]
        if only_hip['cx'] < cx_center:
            righthip_cc = only_hip
            lefthip_cc = None
        else:
            lefthip_cc = only_hip
            righthip_cc = None
    else:
        lefthip_cc = righthip_cc = None

    masks = {}
    if sacrum_cc is not None:
        masks['Sacrum'] = (labels == sacrum_cc['cc_id'])
    if lefthip_cc is not None:
        masks['LeftHip'] = (labels == lefthip_cc['cc_id'])
    if righthip_cc is not None:
        masks['RightHip'] = (labels == righthip_cc['cc_id'])
    log(f"bone-skeleton: {len(masks)} anatomies extracted ({list(masks.keys())})")
    return masks


def ds532_argmax_masks(probs_full):
    """Ds532 4-channel softmax → mutually-exclusive anatomy mask + morphology cleanup.

    Layer 2 일부: Ds532 4-channel softmax 의 argmax 로 서로 배타적인 anatomy mask 를
    추출하고, sparse outlier 잡음을 제거하기 위해 binary_opening + small CC drop 을
    적용한다. 이는 V0.3.1 GC failure (LeftHip bbox 98.6% / RightHip bbox 100%) 처럼
    largest-CC keep 만으로는 막을 수 없는 sparse fragment 연결 문제를 차단한다.

    Args:
        probs_full: (4, Z, Y, X) softmax 확률 텐서.

    Returns:
        dict[str, np.ndarray] — anatomy 별 mask (서로 disjoint 임이 보장됨).
    """
    import scipy.ndimage as ndi
    argmax_map = np.argmax(probs_full, axis=0)  # (Z, Y, X), 0=bg, 1=sacrum, 2=lh, 3=rh
    masks = {}
    structure = np.ones((2, 2, 2), dtype=bool)
    cc_struct = np.ones((3, 3, 3), dtype=bool)
    for anatomy, ch_idx in V5_DATASET532_PROB_CHANNEL.items():
        raw_mask = (argmax_map == ch_idx)
        # Morphology cleanup: opening removes thin / isolated noise
        if DS532_MORPH_OPENING_ITERS > 0:
            cleaned = ndi.binary_opening(raw_mask, structure=structure,
                                          iterations=DS532_MORPH_OPENING_ITERS)
        else:
            cleaned = raw_mask
        # Drop small CCs (Ds532 sparse outliers)
        cc_labels, n_cc = ndi.label(cleaned, structure=cc_struct)
        if n_cc > 1:
            sizes = ndi.sum(cleaned, cc_labels, index=np.arange(1, n_cc + 1))
            keep_ids = np.where(sizes >= MIN_DS532_CC_VOXELS)[0] + 1
            if len(keep_ids) > 0:
                cleaned = np.isin(cc_labels, keep_ids)
            else:
                cleaned = np.zeros_like(cleaned)
        elif n_cc == 1:
            if int(cleaned.sum()) < MIN_DS532_CC_VOXELS:
                cleaned = np.zeros_like(cleaned)
        masks[anatomy] = cleaned
        log(f"Ds532 argmax: {anatomy} raw={int(raw_mask.sum())} cleaned={int(cleaned.sum())} "
            f"({n_cc} CCs, drop<{MIN_DS532_CC_VOXELS})")
    return masks


def merge_masks_with_sanity(ds532_masks, bone_masks, image_shape,
                            sanity_max_fraction=0.35,
                            bone_fallback_when_sanity_fails=True):
    """Layer 2+3: Ds532 mask 와 bone-skeleton mask 를 sanity check 와 함께 병합한다.

    각 anatomy 별 정책:
      1) Ds532 mask 의 voxel 비율이 sanity_max_fraction 미만이면 Ds532 사용 (정확도 우선)
      2) sanity 실패 시 bone skeleton mask 로 fallback (가용한 경우)
      3) 둘 다 실패 → 해당 anatomy 는 skip 하고 라벨을 0 으로 둔다

    반환:
        dict[str, dict] — { anatomy: { 'mask': arr, 'source': 'ds532' | 'bone_fallback' | 'none' } }
    """
    img_voxels = float(np.prod(image_shape))
    out = {}
    for anatomy in ('Sacrum', 'LeftHip', 'RightHip'):
        ds_mask = ds532_masks.get(anatomy)
        bone_mask = bone_masks.get(anatomy)
        ds_fraction = float(ds_mask.sum()) / img_voxels if ds_mask is not None else 1.0
        if ds_mask is not None and ds_fraction < sanity_max_fraction:
            out[anatomy] = {'mask': ds_mask, 'source': 'ds532', 'fraction': ds_fraction}
        elif bone_mask is not None and bone_fallback_when_sanity_fails:
            bone_fraction = float(bone_mask.sum()) / img_voxels
            out[anatomy] = {'mask': bone_mask, 'source': 'bone_fallback', 'fraction': bone_fraction}
            log(f"[{anatomy}] Ds532 sanity fail (covers {ds_fraction*100:.1f}%), bone-skeleton fallback ({bone_fraction*100:.1f}%)")
        else:
            out[anatomy] = {'mask': None, 'source': 'none', 'fraction': ds_fraction}
            log(f"[{anatomy}] both Ds532 ({ds_fraction*100:.1f}%) and bone skeleton failed — anatomy will be zero")
    return out


def estimate_v291_inference_seconds(bbox, patch_size=(224, 160, 192), seconds_per_patch=2.5):
    """Layer 3: V291 sliding-window 추론 소요 시간 대략 추정.

    nnUNet 의 sliding window 는 tile_step_size=0.5 이므로 patch 의 50% 씩 이동한다.
    축별 patch 개수 = ceil((dim - patch) / (patch * 0.5)) + 1.
    이를 곱한 총 patch 수에 patch 당 평균 추론 시간을 곱해 ETA 를 구한다.
    """
    if bbox is None:
        return 0.0
    crop_z = bbox[0].stop - bbox[0].start
    crop_y = bbox[1].stop - bbox[1].start
    crop_x = bbox[2].stop - bbox[2].start
    pz, py, px = patch_size
    n_z = max(1, int(np.ceil(max(0, crop_z - pz) / (pz * 0.5)) + 1))
    n_y = max(1, int(np.ceil(max(0, crop_y - py) / (py * 0.5)) + 1))
    n_x = max(1, int(np.ceil(max(0, crop_x - px) / (px * 0.5)) + 1))
    n_patches = n_z * n_y * n_x
    return float(n_patches) * float(seconds_per_patch)


# ---------------------------------------------------------------------------
# anatomy 별 V291 추론 파이프라인 (메인 로직).
# ---------------------------------------------------------------------------
def run_per_anatomy_pelvic(image_path: Path, ref_img: sitk.Image,
                           prerouted_bone_masks: dict | None = None) -> np.ndarray:
    """V0.3 견고화 파이프라인 (4-layer 구조):
       Layer 1: bone-skeleton anatomy 분해 (HU>200) — 항상 실행 (fallback 확보)
       Layer 2: Ds532 argmax refinement              — Ds532 가 합리적이면 우선 사용
       Layer 3: bbox sanity + 시간 예산 점검         — 안전망 역할
       Layer 4: V291 per-anatomy ABBC 추론 + decode + paste

    Femur (151-200) 은 V0.2 와 동일하게 zero 출력 (V0 에서는 femur 모델 미포함).
    """
    import time
    t_start = time.time()

    # === Step 1: LPS 정규화 + bone window HU clip ===
    img_lps, arr_clipped = canonicalize_and_clip_image(ref_img)
    image_npy_lps = arr_clipped.astype(np.float32, copy=False)
    ct_lut_full = bone_lut_normalize(arr_clipped)
    spacing_xyz = img_lps.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    img_shape = image_npy_lps.shape
    log(f"V0.3 preproc: shape={img_shape} spacing_zyx={spacing_zyx}")

    # === Layer 1: bone HU mask 기반 anatomy 분해 (항상 실행, fallback 용) ===
    if prerouted_bone_masks is not None:
        bone_masks = prerouted_bone_masks
        log(f"L1 bone-skeleton: reusing prerouted masks {list(bone_masks.keys())}")
    else:
        bone_masks = bone_skeleton_anatomy_decomposition(
            arr_clipped, spacing_zyx,
            hu_threshold=BONE_HU_THRESHOLD,
            min_component_voxels=BONE_MIN_COMPONENT_VOXELS,
        )
        log(f"L1 bone-skeleton: {list(bone_masks.keys())}")

    # === Step 2: Ds532 anatomy classifier 추론 (전체 CT, 1-channel HU 입력) ===
    ds532_predictor = build_predictor(
        DS532_DATASET, DS532_TRAINER, DS532_PLANS, DS532_CONFIG, DS532_FOLD,
    )
    ds532_image_4d = image_npy_lps[None]
    ds532_props = {"spacing": list(spacing_zyx)}
    log(f"L2 Ds532 predict: image (C,Z,Y,X)={ds532_image_4d.shape}")
    ds532_logits_pp, ds532_data_props = predict_logits_full(
        ds532_predictor, ds532_image_4d, ds532_props, output_channels=DS532_OUTPUT_CHANNELS,
    )
    ds532_probs_pp = softmax_axis0(ds532_logits_pp)
    # nnUNet preprocessed grid 의 prob 를 원본 CT 격자로 다시 resampling (linear interp)
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    shape_after_cropping = tuple(int(s) for s in ds532_data_props["shape_after_cropping_and_before_resampling"])
    current_spacing = ds532_predictor.configuration_manager.spacing
    transpose_forward = ds532_predictor.plans_manager.transpose_forward
    transpose_backward = ds532_predictor.plans_manager.transpose_backward
    original_spacing = [ds532_data_props["spacing"][i] for i in transpose_forward]
    probs_resampled_axes = resample_data_or_seg_to_shape(
        ds532_probs_pp.astype(np.float32, copy=False),
        shape_after_cropping, current_spacing, original_spacing,
        is_seg=False, order=1, order_z=0,
    )
    shape_before_cropping = tuple(int(s) for s in ds532_data_props["shape_before_cropping"])
    bbox_used = ds532_data_props.get("bbox_used_for_cropping", None)
    probs_full_axes = np.zeros((DS532_OUTPUT_CHANNELS, *shape_before_cropping), dtype=np.float32)
    if bbox_used is None:
        probs_full_axes = np.asarray(probs_resampled_axes, dtype=np.float32)
    else:
        slicer = bounding_box_to_slice(bbox_used)
        probs_full_axes[(slice(None), *slicer)] = np.asarray(probs_resampled_axes, dtype=np.float32)
    # cropping bbox 바깥 영역은 fg 가 비어 있으므로 background prob 를 1 로 채워 정상화
    fg_sum = probs_full_axes[1:].sum(axis=0)
    bg_mask = fg_sum < 1e-6
    if bg_mask.any():
        probs_full_axes[0][bg_mask] = 1.0
    probs_full = np.transpose(probs_full_axes, (0, *(int(a) + 1 for a in transpose_backward)))
    assert probs_full.shape[1:] == img_shape, f"shape mismatch: {probs_full.shape[1:]} vs {img_shape}"

    # === Layer 2: Ds532 argmax 기반 anatomy mask 추출 ===
    ds532_masks = ds532_argmax_masks(probs_full)
    log(f"L2 Ds532 argmax: " + ", ".join(f"{a}={ds532_masks[a].sum()}" for a in ['Sacrum','LeftHip','RightHip']))

    # === Layer 2+3: Ds532 mask 와 bone fallback 을 sanity check 와 함께 병합 ===
    merged = merge_masks_with_sanity(
        ds532_masks, bone_masks, img_shape,
        sanity_max_fraction=SANITY_MAX_BBOX_FRACTION,
        bone_fallback_when_sanity_fails=V03_USE_BONE_SKELETON_FALLBACK,
    )

    # V291 모델 로딩 전에 Ds532 관련 텐서를 해제해 GPU 메모리 확보
    del ds532_predictor, ds532_logits_pp, ds532_probs_pp, probs_resampled_axes, probs_full_axes
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Layer 4: anatomy 별 V291 추론 → ABBC decode → 전체 라벨에 paste ===
    v291_predictor = build_predictor(
        V291_DATASET, V291_TRAINER, V291_PLANS, V291_CONFIG, V291_FOLD,
    )

    full_label = np.zeros(img_shape, dtype=np.uint16)
    for anatomy in ("Sacrum", "LeftHip", "RightHip"):
        elapsed = time.time() - t_start
        if V03_USE_TIME_BUDGET and elapsed >= TIME_BUDGET_HARD_LIMIT:
            log(f"[{anatomy}] HARD time budget hit at {elapsed:.0f}s, emit zero for this and remaining anatomies")
            break

        info = merged[anatomy]
        if info['mask'] is None:
            log(f"[{anatomy}] no usable mask, emit zero")
            continue

        mask = info['mask']
        # 가장 큰 connected component 만 유지 (작은 노이즈 CC 가 fragment 슬롯을 낭비하는 것 방지)
        if LARGEST_CC_KEEP_ONLY:
            import scipy.ndimage as ndi
            cc_labels, n_cc = ndi.label(mask, structure=np.ones((3,3,3), dtype=bool))
            if n_cc > 1:
                sizes = ndi.sum(mask, cc_labels, index=np.arange(1, n_cc+1))
                keep_idx = int(np.argmax(sizes)) + 1
                mask = (cc_labels == keep_idx)

        bbox = bbox_from_mask(mask, pad_vox=ROI_PAD_VOX)
        if bbox is None:
            log(f"[{anatomy}] mask empty after CC filter, emit zero")
            continue

        # NEW (FIX #2): pad 적용 후 bbox sanity check.
        # mask 가 작더라도 tight-FOV 입력에서는 pad 가 추가되며 bbox 가 폭증할 수 있다.
        # bbox 가 볼륨의 50% 를 넘으면 V291 추론이 매우 느려지고 결과 신뢰도가 떨어진다.
        bbox_fraction = float(np.prod([bbox[i].stop - bbox[i].start for i in range(3)])) / float(np.prod(img_shape))
        if bbox_fraction > SANITY_MAX_POSTPAD_BBOX_FRACTION:
            log(f"[{anatomy}] bbox covers {bbox_fraction*100:.1f}% (>{SANITY_MAX_POSTPAD_BBOX_FRACTION*100:.0f}%); "
                f"shrinking pad_vox to keep bbox under threshold")
            # pad_vox 를 단계적으로 줄여가며 임계값 밑으로 떨어뜨린다
            bbox_found = False
            for shrunk_pad in (12, 6, 0):
                bbox_try = bbox_from_mask(mask, pad_vox=shrunk_pad)
                if bbox_try is None:
                    continue
                bbox_fraction_try = float(np.prod([bbox_try[i].stop - bbox_try[i].start for i in range(3)])) / float(np.prod(img_shape))
                if bbox_fraction_try <= SANITY_MAX_POSTPAD_BBOX_FRACTION:
                    log(f"[{anatomy}] shrunk pad_vox={shrunk_pad}, new bbox fraction={bbox_fraction_try*100:.1f}%")
                    bbox = bbox_try
                    bbox_fraction = bbox_fraction_try
                    bbox_found = True
                    break
            if not bbox_found:
                # NEW V0.3.2 fix: try bone-skeleton mask before giving up
                bone_mask = bone_masks.get(anatomy)
                if bone_mask is not None and bone_mask.any():
                    # Largest CC only
                    if LARGEST_CC_KEEP_ONLY:
                        import scipy.ndimage as _ndi
                        bcc_labels, bcc_n = _ndi.label(bone_mask, structure=np.ones((3, 3, 3), dtype=bool))
                        if bcc_n > 1:
                            bcc_sizes = _ndi.sum(bone_mask, bcc_labels, index=np.arange(1, bcc_n + 1))
                            keep_idx = int(np.argmax(bcc_sizes)) + 1
                            bone_mask = (bcc_labels == keep_idx)
                    bbox_bone = bbox_from_mask(bone_mask, pad_vox=ROI_PAD_VOX)
                    if bbox_bone is not None:
                        bbox_bone_fraction = float(np.prod([bbox_bone[i].stop - bbox_bone[i].start for i in range(3)])) / float(np.prod(img_shape))
                        if bbox_bone_fraction <= SANITY_MAX_POSTPAD_BBOX_FRACTION:
                            log(f"[{anatomy}] bone-skeleton fallback after bbox-sanity fail: bbox fraction {bbox_bone_fraction*100:.1f}%")
                            mask = bone_mask
                            bbox = bbox_bone
                            bbox_fraction = bbox_bone_fraction
                            info['source'] = 'bone_after_bbox_sanity_fail'
                            # Recompute mask probability channel for V291 input
                            bbox_found = True
                if not bbox_found:
                    # pad=0 으로도 너무 크고 bone fallback 도 실패 — zero 로 처리
                    log(f"[{anatomy}] even bone-skeleton bbox too large or absent; emit zero")
                    continue

        # 시간 예산 기반 sanity check (Grand Challenge 10분 제한 보호)
        eta = estimate_v291_inference_seconds(bbox)
        bbox_fraction = float(np.prod([bbox[i].stop - bbox[i].start for i in range(3)])) / float(np.prod(img_shape))
        log(f"[{anatomy}] bbox crop={tuple(bbox[i].stop - bbox[i].start for i in range(3))} "
            f"fraction={bbox_fraction*100:.1f}% source={info['source']} ETA={eta:.0f}s elapsed={elapsed:.0f}s")

        if V03_USE_TIME_BUDGET and (elapsed + eta) > TIME_BUDGET_SECONDS:
            log(f"[{anatomy}] elapsed+ETA={elapsed+eta:.0f}s > budget {TIME_BUDGET_SECONDS}s, emit zero")
            continue

        # V291 입력: 3-channel anatomy-aware 텐서 구성
        # (V5 학습 레시피와 동일 — CT_LUT / anatomy prob / SDF)
        ct_lut_crop = ct_lut_full[bbox]
        # Channel 1: Ds532 anatomy 확률 ([0,1] clip). bone fallback 시에는 mask 자체를 0/1 로 사용
        prob_crop = probs_full[V5_DATASET532_PROB_CHANNEL[anatomy]][bbox] if info['source'] == 'ds532' else np.clip(mask[bbox].astype(np.float32), 0.0, 1.0)
        # Channel 2: prob>=0.5 mask 의 signed distance field
        sdf_crop = selected_anatomy_sdf_from_prob(prob_crop, spacing_zyx, clip_mm=SDF_CLIP_MM)
        v291_image_4d = np.stack([ct_lut_crop, prob_crop, sdf_crop], axis=0)
        v291_props = {"spacing": list(spacing_zyx)}

        # V291 sliding-window 추론 실행
        v291_logits_pp, v291_data_props = predict_logits_full(
            v291_predictor, v291_image_4d, v291_props, output_channels=V291_OUTPUT_CHANNELS,
        )
        v291_probs = softmax_axis0(v291_logits_pp)
        decoded_pp = decode_abbc_core_seed_watershed(v291_probs)
        decoded_crop = resample_label_map_to_original(decoded_pp, v291_data_props, v291_predictor)

        # 라벨 재매핑 후 전체 볼륨에 paste
        # PENGWIN 스펙: Sacrum [1..50], LeftHip [51..100], RightHip [101..150]
        lo, hi = V5_ANATOMY_RANGES[anatomy]
        n_slots = hi - lo + 1
        local_ids = sorted(int(v) for v in np.unique(decoded_crop) if int(v) > 0)
        if not local_ids:
            log(f"[{anatomy}] no fragments after decode")
            continue
        # 슬롯이 부족하면 fragment 크기 순으로 상위 n_slots 개만 살린다
        if len(local_ids) > n_slots:
            sizes_local = [(lid, int((decoded_crop == lid).sum())) for lid in local_ids]
            sizes_local.sort(key=lambda x: -x[1])
            local_ids = [t[0] for t in sizes_local[:n_slots]]
        remap = np.zeros(max(local_ids) + 1, dtype=np.uint16)
        for i, old_id in enumerate(local_ids):
            remap[old_id] = np.uint16(lo + i)
        # 슬롯 초과 ID 는 0 으로 마스킹 (PENGWIN 라벨 범위 보호)
        decoded_crop_filtered = np.where(np.isin(decoded_crop, local_ids), decoded_crop, 0).astype(np.int32, copy=False)
        # remap 인덱스를 안전하게 사용하기 위해 valid 범위로 clip
        decoded_crop_filtered = np.clip(decoded_crop_filtered, 0, max(local_ids))
        remapped_crop = remap[decoded_crop_filtered]

        out_slot = full_label[bbox]
        write_mask = (remapped_crop > 0) & (out_slot == 0)
        out_slot[write_mask] = remapped_crop[write_mask]
        full_label[bbox] = out_slot
        log(f"[{anatomy}] painted {int(write_mask.sum())} voxels, {len(local_ids)} fragments in range [{lo},{hi}]")

    # === Step 5: 원본이 LPS 가 아니었다면 원래 방향으로 되돌린다 ===
    # nnUNet 은 LPS 로 추론하므로, 원본 orientation 으로 복구해야
    # Grand Challenge 가 기대하는 좌표계와 일치한다.
    if orientation_code(ref_img) != "LPS":
        lps_label_img = sitk.GetImageFromArray(full_label.astype(np.uint16, copy=False))
        lps_label_img.SetSpacing(img_lps.GetSpacing())
        lps_label_img.SetOrigin(img_lps.GetOrigin())
        lps_label_img.SetDirection(img_lps.GetDirection())
        target_code = orientation_code(ref_img)
        oriented = sitk.DICOMOrient(lps_label_img, target_code)
        full_label = sitk.GetArrayFromImage(oriented).astype(np.uint16, copy=False)
        log(f"reoriented label map: LPS -> {target_code} (shape={full_label.shape})")

    log(f"V0.3 pipeline complete in {time.time() - t_start:.1f}s, unique={sorted(int(v) for v in np.unique(full_label)[:20])}")
    return full_label


# ---------------------------------------------------------------------------
# main() — 컨테이너 진입점
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    log(f"start: model_root={MODEL_ROOT} nn_results={NN_RES}")
    try:
        image_path = find_input_image()
        log(f"input image: {image_path}")
        ref_img = sitk.ReadImage(str(image_path))

        size = ref_img.GetSize()       # SimpleITK 순서 (X, Y, Z)
        spacing = ref_img.GetSpacing() # SimpleITK 순서 (sx, sy, sz)
        spacing_x, spacing_y, spacing_z = (float(s) for s in spacing)
        physical_x_mm = float(size[0]) * spacing_x
        physical_z_mm = float(size[2]) * spacing_z
        log(
            "spatial: "
            f"size={size} spacing=({spacing_x:.4f},{spacing_y:.4f},{spacing_z:.4f}) "
            f"physical_x_mm={physical_x_mm:.2f} physical_z_mm={physical_z_mm:.2f}"
        )
        # Bone-skeleton-aware routing: spacing 룰 단독으로는 case 163/189 처럼
        # spacing_z=0.80 인 pelvic small-FOV 를 'femur' 로 잘못 라우팅한다.
        # 먼저 LPS 정규화 + HU clip 한 배열을 만들고, bone HU mask 로부터
        # Sacrum/LeftHip/RightHip 가 2개 이상 확인되면 pelvic 으로 강제.
        img_lps_for_route, arr_clipped_for_route = canonicalize_and_clip_image(ref_img)
        spacing_xyz_lps = img_lps_for_route.GetSpacing()
        spacing_zyx_lps = (
            float(spacing_xyz_lps[2]),
            float(spacing_xyz_lps[1]),
            float(spacing_xyz_lps[0]),
        )
        route, prerouted_bone_masks = classify_pelvic_femur_v2(
            arr_clipped_for_route, spacing_zyx_lps,
            spacing_x, spacing_y, spacing_z, physical_x_mm, physical_z_mm,
        )
        log(f"anatomy routing: {route}")

        out_path = output_path(image_path)
        if route == "femur":
            write_zero_output(image_path, ref_img, "femur route unmodeled in V0")
        else:
            label_arr = run_per_anatomy_pelvic(image_path, ref_img, prerouted_bone_masks)
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

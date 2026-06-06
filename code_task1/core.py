"""PENGWIN 2026 Task 1 — Core constants and dataset registry.

Single source of truth for:
    - PENGWIN label scheme (1-200 instance IDs)
    - Anatomy classes + ranges
    - Pelvic / Femur split (case ID ranges)
    - Active split Pelvic/Femur anatomy datasets
    - Active pelvic per-anatomy BICM V5 fragment dataset
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import sys
import time
from typing import Literal

# =============================================================================
# Paths
# =============================================================================
# Keep /workspace as the default because this challenge workstation and the
# current scripts are laid out there, but allow Docker/local runs to relocate the
# tree by exporting PENGWIN_ROOT before importing any code_task1 modules.
ROOT = Path(os.environ.get("PENGWIN_ROOT", "/workspace"))
DATA_RAW = ROOT / "data/task1_2/extracted"
# nnUNet 출력 root 를 code_task1/result/ 아래로 통합 (2026-05-31 정리).
# 이전: ROOT/nnunet/{raw,preprocessed,results} — workspace 루트에 분산.
# 현재: ROOT/code_task1/result/{raw,preprocessed,results} — code_task1 내부에서 자급자족.
# 변경 이유: code_task1 이 독립 작업 디렉토리가 되도록 — 모든 산출물이 한 곳에 모이게.
NN_RAW = ROOT / "code_task1/result/raw"
NN_PREP = ROOT / "code_task1/result/preprocessed"
NN_RES = ROOT / "code_task1/result/results"
RESULT = ROOT / "code_task1/result"
# Result artifacts 의 평면화 (2026-05-31 정리).
# 이전: task1_active/<UTC YYYYMMDD>/{reports,weights,visualize} — date partition.
# 현재: reports/ 단일 폴더 — 시간 정보는 파일명의 RESULT_DATE 접미사로 표현.
# 변경 이유: code_task1/ 내부 중복 prefix (task1_active) 제거, day-partition 으로 인한
# 디스크 분산 및 GC 부담 회피.
RESULT_DATE = os.environ.get("PENGWIN_RESULT_DATE", time.strftime("%Y%m%d", time.gmtime()))
RESULT_REPORT = RESULT / "reports"
RESULT_WEIGHT = RESULT / "weights"
RESULT_VISUALIZE = RESULT / "visualize"


def configure_nnunet_env(force: bool = False) -> dict[str, str]:
    """Set nnU-Net path environment variables from this repo layout.

    nnU-Net still reads these paths from environment variables at import and
    inference time. Keeping the setup here makes every Python entry point
    self-contained and prevents the noisy "nnUNet_raw is not defined" warnings
    during smoke tests.
    """
    values = {
        "nnUNet_raw": str(NN_RAW),
        "nnUNet_preprocessed": str(NN_PREP),
        "nnUNet_results": str(NN_RES),
        # nnU-Net v1-style compatibility variable used by a few helpers.
        "nnUNet_raw_data_base": str(NN_RAW),
    }
    for key, value in values.items():
        if force or not os.environ.get(key):
            os.environ[key] = value
    return values


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger without requiring a separate local logging module."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
    return log


def setup_cli_logging(level: int = logging.INFO) -> None:
    """Configure stdout logging once for command-line entry points."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        root.addHandler(handler)


configure_nnunet_env()


# =============================================================================
# PENGWIN label scheme (대회 공식) — single source of truth in anatomy_registry
# =============================================================================
# Instance IDs encode anatomy via fixed 50-wide blocks (Sacrum 1-50, LeftHip
# 51-100, RightHip 101-150, Femur 151-200). The canonical registry AND every
# instance-ID arithmetic helper now live in `anatomy_registry`, re-exported here
# so existing `from core import ANATOMY_RANGES` / `core.<x>` references keep
# resolving unchanged. Add or modify an anatomy THERE (one row), never here.
#
# DECOY WARNING: the case-ID helpers below (is_pelvic/is_femur, 1-120/151-200/
# 251-420) and HU/probability thresholds elsewhere are a DIFFERENT namespace that
# happens to share the digits 150/200/50 — they must NOT route through the registry.
from anatomy_registry import (  # noqa: E402  (beside the constants it replaces)
    Anatomy,
    ANATOMY_REGISTRY,
    ANATOMY_RANGES,
    ANATOMY_NAMES,
    ANATOMY_RANGE_DICT,
    ANATOMY_TO_INDEX,
    PELVIC_ANATOMY_INDICES,
    NUM_ANATOMIES,
    MIN_INSTANCE_ID,
    MAX_INSTANCE_ID,
    PELVIC_MAX_INSTANCE_ID,
    INSTANCE_CAPACITY,
    anatomy_by_index,
    anatomy_by_name,
    anatomy_of_id,
    id_range,
    all_anatomy_ranges,
    anatomy_start_ids,
    anatomy_ranges_by_name,
    global_to_local,
    local_to_global,
    valid_instance_mask,
    clip_to_valid_instances,
    anatomy_index_array,
    same_anatomy,
)


# =============================================================================
# Pelvic / Femur split (active source case ID ranges)
# =============================================================================
def is_pelvic(cid: int) -> bool:
    """True if case ID belongs to Pelvic train set (001-120, 151-200)."""
    return (1 <= cid <= 120) or (151 <= cid <= 200)


def is_femur(cid: int) -> bool:
    """True if case ID belongs to Femur train set (251-420)."""
    return 251 <= cid <= 420


def case_subject_type(cid: int) -> str:
    """Return 'Pelvic' / 'Femur' / 'Unknown' based on case ID range."""
    if is_pelvic(cid):
        return "Pelvic"
    if is_femur(cid):
        return "Femur"
    return "Unknown"


# =============================================================================
# Dataset registry — active anatomy V2 + fragment V5 datasets only
# =============================================================================
ABBC_BOUNDARY_LABEL = 1
ABBC_CORE_LABEL = 2
ABBC_BORDER_LABEL = 3
ABBC_CONTACT_LABEL = ABBC_BORDER_LABEL
ABBC_HARD_NEGATIVE_LABEL = 4

# [AUDIT][Risk:Medium][Scope:legacy_constant]
# The constants below remain only so old JSON/probe readers can import this
# module while we delete heavy V3/V4 artifacts. Active Dataset537 V5 does not
# use these heads or sidecars; registry and CLI must point to `bicm_v5` only.
FACTOR_INSTANCE_OUTPUT_CHANNELS = 15
FACTOR_SUPPORT_CH = 0
FACTOR_CONTACT_CH = 1
FACTOR_CORE_CH = 2
FACTOR_HARD_NEGATIVE_CH = 3
FACTOR_OFFSET_CHS = (4, 5, 6)
FACTOR_EMBEDDING_CHS = tuple(range(7, 15))
FACTOR_INSTANCE_CHANNEL_NAMES = [
    "support_logit",
    "contact_energy_logit",
    "core_seed_logit",
    "hard_negative_logit",
    "offset_z",
    "offset_y",
    "offset_x",
    "embedding_0",
    "embedding_1",
    "embedding_2",
    "embedding_3",
    "embedding_4",
    "embedding_5",
    "embedding_6",
    "embedding_7",
]
FACTOR_INSTANCE_SIDECAR_DIR = "factorized_instance_targets"
BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR = "boundary_fragment_v3_targets"

# Retained for old JSON/probe readers only. This 19-channel morphology/edge
# contract is not the active Dataset537 training contract.
CONTACT_INSTANCE_OUTPUT_CHANNELS = 19
CONTACT_INSTANCE_MORPH_BG_CH = 0
CONTACT_INSTANCE_MORPH_SUPPORT_CH = 1
CONTACT_INSTANCE_MORPH_CONTACT_CH = 2
CONTACT_INSTANCE_MORPH_HARD_NEGATIVE_CH = 3
CONTACT_INSTANCE_CORE_CH = 4
CONTACT_INSTANCE_OFFSET_CHS = (5, 6, 7)
CONTACT_INSTANCE_EMBEDDING_CHS = tuple(range(8, 16))
CONTACT_INSTANCE_EDGE_BREAK_CHS = (16, 17, 18)
CONTACT_INSTANCE_CHANNEL_NAMES = [
    "morph_background_logit",
    "morph_support_noncontact_logit",
    "morph_contact_surface_logit",
    "morph_hard_negative_logit",
    "core_center_logit",
    "offset_z",
    "offset_y",
    "offset_x",
    "embedding_0",
    "embedding_1",
    "embedding_2",
    "embedding_3",
    "embedding_4",
    "embedding_5",
    "embedding_6",
    "embedding_7",
        "edge_break_z_logit",
        "edge_break_y_logit",
        "edge_break_x_logit",
    ]

BICM_V38_OUTPUT_CHANNELS = 8
BICM_V38_CHANNEL_NAMES = [
    "support_logit",
    "contact_edge_ridge_logit",
    "core_marker_logit",
    "exterior_context_logit",
    "roi_contact_present_logit",
    "edge_break_z_logit",
    "edge_break_y_logit",
    "edge_break_x_logit",
]

BICM_V68_OUTPUT_CHANNELS = 10
BICM_V68_CHANNEL_NAMES = [
    "semantic_background_logit",
    "semantic_exterior_context_logit",
    "semantic_interior_shell_logit",
    "semantic_core_logit",
    "semantic_contact_surface_logit",
    "contact_topology_logit",
    "core_marker_logit",
    "edge_break_z_logit",
    "edge_break_y_logit",
    "edge_break_x_logit",
]

BFV3_BINARY_BARRIER_OUTPUT_CHANNELS = 6
BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS = 7
BFV3_XYZ_AFFINITY_OUTPUT_CHANNELS = 8
BFV3_XYZ_AFFINITY_CHANNEL_NAMES = [
    "semantic_background_logit",
    "semantic_exterior_context_logit",
    "semantic_contact_label2_logit",
    "semantic_shell_label3_logit",
    "semantic_core_label4_logit",
    "same_fragment_affinity_z_logit",
    "same_fragment_affinity_y_logit",
    "same_fragment_affinity_x_logit",
]
BFV3_AFFINITY13_SEED_OUTPUT_CHANNELS = 18
BFV3_AFFINITY13_SEED_CHANNEL_NAMES = [
    "support_logit",
    "exterior_context_logit",
    "fracture_surface_logit",
    "seed_center_logit",
    "seed_body_logit",
    "same_fragment_affinity_0_0_1_logit",
    "same_fragment_affinity_0_1_-1_logit",
    "same_fragment_affinity_0_1_0_logit",
    "same_fragment_affinity_0_1_1_logit",
    "same_fragment_affinity_1_-1_-1_logit",
    "same_fragment_affinity_1_-1_0_logit",
    "same_fragment_affinity_1_-1_1_logit",
    "same_fragment_affinity_1_0_-1_logit",
    "same_fragment_affinity_1_0_0_logit",
    "same_fragment_affinity_1_0_1_logit",
    "same_fragment_affinity_1_1_-1_logit",
    "same_fragment_affinity_1_1_0_logit",
    "same_fragment_affinity_1_1_1_logit",
]
BFV3_MUTEX13_SEED_OUTPUT_CHANNELS = 31
BFV3_MUTEX13_SEED_CHANNEL_NAMES = BFV3_AFFINITY13_SEED_CHANNEL_NAMES + [
    "different_fragment_mutex_0_0_1_logit",
    "different_fragment_mutex_0_1_-1_logit",
    "different_fragment_mutex_0_1_0_logit",
    "different_fragment_mutex_0_1_1_logit",
    "different_fragment_mutex_1_-1_-1_logit",
    "different_fragment_mutex_1_-1_0_logit",
    "different_fragment_mutex_1_-1_1_logit",
    "different_fragment_mutex_1_0_-1_logit",
    "different_fragment_mutex_1_0_0_logit",
    "different_fragment_mutex_1_0_1_logit",
    "different_fragment_mutex_1_1_-1_logit",
    "different_fragment_mutex_1_1_0_logit",
    "different_fragment_mutex_1_1_1_logit",
]
BFV3_NO_CONTACT_PAIRWISE_V273_OUTPUT_CHANNELS = 28
BFV3_NO_CONTACT_PAIRWISE_V273_CHANNEL_NAMES = [
    "support_logit",
    "seed_body_logit",
    "join_0_0_1_logit",
    "join_0_1_-1_logit",
    "join_0_1_0_logit",
    "join_0_1_1_logit",
    "join_1_-1_-1_logit",
    "join_1_-1_0_logit",
    "join_1_-1_1_logit",
    "join_1_0_-1_logit",
    "join_1_0_0_logit",
    "join_1_0_1_logit",
    "join_1_1_-1_logit",
    "join_1_1_0_logit",
    "join_1_1_1_logit",
    "cut_0_0_1_logit",
    "cut_0_1_-1_logit",
    "cut_0_1_0_logit",
    "cut_0_1_1_logit",
    "cut_1_-1_-1_logit",
    "cut_1_-1_0_logit",
    "cut_1_-1_1_logit",
    "cut_1_0_-1_logit",
    "cut_1_0_0_logit",
    "cut_1_0_1_logit",
    "cut_1_1_-1_logit",
    "cut_1_1_0_logit",
    "cut_1_1_1_logit",
]
BFV3_FRAGMENT_POSITION_V275_OUTPUT_CHANNELS = 51
BFV3_FRAGMENT_POSITION_V275_CHANNEL_NAMES = [
    "background_logit",
] + [f"fragment_position_{idx:02d}_logit" for idx in range(1, 51)]
BFV3_SEPARATOR_GAP_V277_OUTPUT_CHANNELS = 2
BFV3_SEPARATOR_GAP_V277_CHANNEL_NAMES = [
    "support_logit",
    "separator_gap_logit",
]
BFV3_SEPARATOR_ENERGY_V278_OUTPUT_CHANNELS = 2
BFV3_SEPARATOR_ENERGY_V278_CHANNEL_NAMES = [
    "support_logit",
    "separator_energy_logit",
]
BFV3_SEPARATOR_SOFTMAX_V287_OUTPUT_CHANNELS = 3
BFV3_SEPARATOR_SOFTMAX_V287_CHANNEL_NAMES = [
    "background_logit",
    "support_body_logit",
    "separator_gap_logit",
]
BFV3_ABBC_V288_OUTPUT_CHANNELS = 4
BFV3_ABBC_V288_CHANNEL_NAMES = [
    "background_logit",
    "border_logit",
    "boundary_logit",
    "core_logit",
]
BFV3_ABBC_SDF_V289_OUTPUT_CHANNELS = 4
BFV3_ABBC_SDF_V289_CHANNEL_NAMES = [
    "background_logit",
    "border_logit",
    "boundary_sdf",
    "core_logit",
]
BFV3_ABBC_SDF_FDM_V290_OUTPUT_CHANNELS = 4
BFV3_ABBC_SDF_FDM_V290_CHANNEL_NAMES = BFV3_ABBC_SDF_V289_CHANNEL_NAMES
BFV3_ABBC_BWEIGHT_V291_OUTPUT_CHANNELS = 4
BFV3_ABBC_BWEIGHT_V291_CHANNEL_NAMES = BFV3_ABBC_V288_CHANNEL_NAMES
BFV3_CENTER_FLOW_OUTPUT_CHANNELS = 8
BFV3_CENTER_FLOW_CHANNEL_NAMES = [
    "support_logit",
    "exterior_context_logit",
    "fracture_surface_logit",
    "center_heatmap_logit",
    "distance_field_logit",
    "offset_to_center_z",
    "offset_to_center_y",
    "offset_to_center_x",
]
BFV3_NO_CONTACT_CENTER_FLOW_OUTPUT_CHANNELS = 5
BFV3_NO_CONTACT_CENTER_FLOW_CHANNEL_NAMES = [
    "support_logit",
    "center_heatmap_logit",
    "offset_to_center_z",
    "offset_to_center_y",
    "offset_to_center_x",
]
BFV3_SPATIAL_EMBEDDING_OUTPUT_CHANNELS = 4
BFV3_SPATIAL_EMBEDDING_CHANNEL_NAMES = [
    "support_logit",
    "sink_z_logit",
    "sink_y_logit",
    "sink_x_logit",
]
BFV3_QUERY_MASK_V280_OUTPUT_CHANNELS = 1
BFV3_QUERY_MASK_V280_CHANNEL_NAMES = [
    "query_fragment_mask_logit",
]
BFV3_QUERY_MASK_PN_V281_OUTPUT_CHANNELS = 1
BFV3_QUERY_MASK_PN_V281_CHANNEL_NAMES = [
    "query_fragment_mask_logit",
]
BFV3_GLOBAL_COORD_QUERY_MASK_V285_OUTPUT_CHANNELS = 1
BFV3_GLOBAL_COORD_QUERY_MASK_V285_CHANNEL_NAMES = BFV3_QUERY_MASK_V280_CHANNEL_NAMES
BFV3_GLOBAL_COORD_QUERY_MASK_PN_V286_OUTPUT_CHANNELS = 1
BFV3_GLOBAL_COORD_QUERY_MASK_PN_V286_CHANNEL_NAMES = BFV3_QUERY_MASK_PN_V281_CHANNEL_NAMES
BFV3_FREE_EMBEDDING_V282_OUTPUT_CHANNELS = 5
BFV3_FREE_EMBEDDING_V282_CHANNEL_NAMES = [
    "support_logit",
    "free_embedding_0",
    "free_embedding_1",
    "free_embedding_2",
    "free_embedding_3",
]
BFV3_GLOBAL_COORD_FREE_EMBEDDING_V283_OUTPUT_CHANNELS = 5
BFV3_GLOBAL_COORD_FREE_EMBEDDING_V283_CHANNEL_NAMES = BFV3_FREE_EMBEDDING_V282_CHANNEL_NAMES

BICM_V105_CHANNEL_NAMES = [
    "semantic_background_logit",
    "semantic_exterior_context_logit",
    "semantic_interior_shell_logit",
    "semantic_core_logit",
    "semantic_contact_surface_logit",
    "contact_metric_logit",
    "core_marker_logit",
    "offset_to_center_z",
    "offset_to_center_y",
    "offset_to_center_x",
]


@dataclass(frozen=True)
class DatasetCfg:
    """Single dataset definition. Heterogeneous fields by `kind`.

    Active training uses split semantic anatomy targets plus one pelvic
    per-anatomy fragment target:
    - Dataset532_PelvicAnatomyV2: background/Sacrum/LeftHip/RightHip.
    - Dataset533_FemurAnatomyV2: background/Femur.
    - Dataset537_PelvicBICMFragmentV5: CT-LUT per-anatomy ROI samples with V5
      target geometry. Active training uses V68 hybrid semantic/topology heads
      because V67 support succeeded while contact/core topology failed.

    """
    name: str
    kind: Literal["anatomy_semantic", "bicm_v5"]
    filter: Literal["all", "pelvic", "femur"]
    n_classes: int
    trainer: Literal[
        "PengwinTrainer",
        "PengwinTrainerBICMFactorizedV6",
        "PengwinTrainerBICMContactV8",
        "PengwinTrainerBICMContactV9",
        "PengwinTrainerBICMContactV10",
        "PengwinTrainerBICMContactV11",
        "PengwinTrainerBICMContactV12",
        "PengwinTrainerBICMContactV13",
        "PengwinTrainerBICMContactV14",
        "PengwinTrainerBICMContactV15",
        "PengwinTrainerBICMContactV16",
        "PengwinTrainerBICMContactV17",
        "PengwinTrainerBICMContactV18",
        "PengwinTrainerBICMContactV19",
        "PengwinTrainerBICMContactV20",
        "PengwinTrainerBICMContactV22",
        "PengwinTrainerBICMContactV23",
        "PengwinTrainerBICMContactV24",
        "PengwinTrainerBICMContactV25",
        "PengwinTrainerBICMContactV26",
        "PengwinTrainerBICMContactV27",
        "PengwinTrainerBICMContactV28",
        "PengwinTrainerBICMContactV29",
        "PengwinTrainerBICMContactV30",
        "PengwinTrainerBICMContactV31",
        "PengwinTrainerBICMContactV32",
        "PengwinTrainerBICMContactV34",
        "PengwinTrainerBICMContactV35",
        "PengwinTrainerBICMContactV36",
        "PengwinTrainerBICMContactV37",
        "PengwinTrainerBICMEdgeAffinityV38",
        "PengwinTrainerBICMEdgePrimaryV39",
        "PengwinTrainerBICMEdgeCoreV40",
        "PengwinTrainerBICMInstanceCoreV41",
        "PengwinTrainerBICMInstanceCoreV42",
        "PengwinTrainerBICMEdgeContactV43",
        "PengwinTrainerBICMEdgeLocalRankV44",
        "PengwinTrainerBICMEdgeCandidateV45",
        "PengwinTrainerBICMEdgeCurriculumV46",
        "PengwinTrainerBICMSeparatedContactV47",
        "PengwinTrainerBICMSupportAwareContactV48",
        "PengwinTrainerBICMDecoderFeatureContactV49",
        "PengwinTrainerBICMDecoderFeaturePhaseV50",
        "PengwinTrainerBICMDenseEdgeCostV51",
        "PengwinTrainerBICMDenseEdgeCoreV52",
        "PengwinTrainerBICMFragmentMarkerCoreV53",
        "PengwinTrainerBICMSparseHeadBalancedV54",
        "PengwinTrainerBICMCalibratedSparseHeadV55",
        "PengwinTrainerBICMPhasedMarkerContactV56",
        "PengwinTrainerBICMDecoderFeatureMarkerPhaseV57",
        "PengwinTrainerBICMHeatmapMarkerContactV58",
        "PengwinTrainerBICMCorePreservingContactV59",
        "PengwinTrainerBICMStrictCoreTopologyV60",
        "PengwinTrainerBICMPeakSeedV61",
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
        "PengwinTrainerBICMEvalBandEdgePrimaryV86",
        "PengwinTrainerBICMEvalBandSemanticGateV87",
        "PengwinTrainerBICMCorePreservingBandPrecisionV88",
        "PengwinTrainerBICMBandFalseOnlyPrecisionV89",
        "PengwinTrainerBICMSemanticBandProductV90",
        "PengwinTrainerBICMTopologyAwareEdgeBalanceV91",
        "PengwinTrainerBICMEvalAlignedSupportContactV93",
        "PengwinTrainerBICMFinalRowCalibratedV94",
        "PengwinTrainerBICMDecoderFeatureContactV95",
        "PengwinTrainerBICMDenseBandGateV96",
        "PengwinTrainerBICMPositiveDenseBandGateV97",
        "PengwinTrainerBICMTeacherDistilledDenseGateV98",
        "PengwinTrainerBICMAdaptiveDualMarginDenseGateV99",
        "PengwinTrainerBICMNegativeBalancedDenseGateV100",
        "PengwinTrainerBICMDualFieldProductGateV101",
        "PengwinTrainerBICMDistanceRankDenseGateV102",
        "PengwinTrainerBICMSemanticDistanceContactV103",
        "PengwinTrainerBICMTeacherSemanticContactV104",
        "PengwinTrainerBICMOffsetAssignmentV105",
        "PengwinTrainerBICMOffsetAttractorV106",
        "PengwinTrainerBICMRadialSupportOffsetV107",
        "PengwinTrainerBICMWatershedBarrierV108",
        "PengwinTrainerBICMPrecisionLockedRecallV109",
        "PengwinTrainerBICMSemanticGeometryBridgeV110",
    ]
    anatomies: list[str] = field(default_factory=list)
    anatomy: str | None = None
    global_label_range: tuple[int, int] | None = None
    foundation_dataset: int | None = None

    def __getitem__(self, key: str):
        """Dict-style access for existing utility code."""
        return getattr(self, key)


DATASETS: dict[int, DatasetCfg] = {
    # Active anatomy learning is split by subject region so each model sees one
    # consistent target distribution and label set.
    532: DatasetCfg(
        "Dataset532_PelvicAnatomyV2",
        "anatomy_semantic",
        "pelvic",
        4,
        "PengwinTrainer",
        anatomies=["Sacrum", "LeftHip", "RightHip"],
    ),
    533: DatasetCfg(
        "Dataset533_FemurAnatomyV2",
        "anatomy_semantic",
        "femur",
        2,
        "PengwinTrainer",
        anatomies=["Femur"],
    ),
    # Dataset537 keeps the V5 materialized ROI/label contract. V81 remains the
    # active locked diagnostic: V82/V83/V84 improved isolated precision or
    # patch metrics but lost full-volume recall/support/core balance.
    # Fulltrain remains blocked until a fixed full-volume gate passes.
    537: DatasetCfg(
        "Dataset537_PelvicBICMFragmentV5",
        "bicm_v5",
        "pelvic",
        5,
        "PengwinTrainerBICMCoreStableEdgePrecisionV81",
        anatomies=["Sacrum", "LeftHip", "RightHip"],
        anatomy=None,
        global_label_range=(1, 150),
        foundation_dataset=532,
    ),
    # [V0.x][FIX:B2][2026-05-31] Dataset538 — Dataset537 의 4-anatomy 버전.
    # filter="all" 로 170 pelvic + 170 femur (총 340) 케이스 모두 포함.
    # global_label_range=(1, 200) 로 femur fragment ID (151-200) 까지 커버.
    # anatomies=[..., "Femur"] 로 빌드 시점에 4 ROI 생성.
    # foundation_dataset=539 — Stage E inference 시 Dataset539 의 5-class
    # anatomy probability 를 입력 채널로 사용.
    538: DatasetCfg(
        "Dataset538_PelvicFemurBICMFragmentV5",
        "bicm_v5",
        "all",
        5,
        "PengwinTrainerBICMCoreStableEdgePrecisionV81",
        anatomies=["Sacrum", "LeftHip", "RightHip", "Femur"],
        anatomy=None,
        global_label_range=(1, 200),
        foundation_dataset=539,
    ),
    # [V0.x][FIX:B1][2026-05-31] Dataset539 — Dataset532 의 5-class 버전.
    # 베이스라인 Dataset001 과 호환되는 5-class semantic anatomy (Femur 포함).
    # filter="all" 로 340 케이스 모두 학습 — 170 femur-only 케이스 구제.
    539: DatasetCfg(
        "Dataset539_PelvicFemurAnatomyV3",
        "anatomy_semantic",
        "all",
        5,
        "PengwinTrainer",
        anatomies=["Sacrum", "LeftHip", "RightHip", "Femur"],
    ),
}


RESERVED_DATASETS = {
    534: "retired_deleted_20260519",
    535: "retired_deleted_20260519",
    536: "retired_deleted_20260519",
}


def dataset_cfg(ds_id: int) -> DatasetCfg:
    """Return the active dataset config."""
    if ds_id not in DATASETS:
        raise KeyError(f"unknown active dataset {ds_id}. active={sorted(DATASETS)}")
    return DATASETS[ds_id]


# =============================================================================
# SOTA reference (PENGWIN 2024 official, MIC-DKFZ 1st)
# =============================================================================
SOTA = {
    "iou_a": 0.9810,
    "iou_f": 0.9296,
    "hd95_f": 5.866,
    "assd_f": 1.843,
}


# =============================================================================
# Geometry defaults used by preprocessing diagnostics
# =============================================================================
PAD = 15              # v8: 10→15. bbox crop 시 small fragment (~100 vox) 절단 방지.
                      #     PENGWIN spec drops <500mm³ ≈ 1000 vox at 0.8mm³, but our
                      #     GT contains down to 1 vox fragments — 5 voxel 추가 margin.


# =============================================================================
# Active nnU-Net trainers
# =============================================================================
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch import distributed as dist
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

# [V0.x][WARN-FIX:#3][2026-06-02] nnUNet 2.5.2 가 deprecated `torch.cuda.amp.GradScaler()`
# 를 nnUNetTrainer.initialize() 의 GradScaler() 호출(line 164)에서 생성하며 FutureWarning
# 을 띄운다. 경고는 *생성 시점*에 발생하므로 사후 교체로는 못 막는다. nnUNet 트레이너 모듈
# 네임스페이스의 GradScaler 심볼을 신 API(torch.amp.GradScaler('cuda')) 팩토리로 교체해
# 생성 자체를 modern API 로 바꾼다(억제 아님, 근본 수정). 모든 trainer 에 적용.
try:
    import torch.amp as _torch_amp
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _nnunet_trainer_mod

    def _modern_grad_scaler(*_a, **_k):
        return _torch_amp.GradScaler("cuda", *_a, **_k)

    _nnunet_trainer_mod.GradScaler = _modern_grad_scaler
except Exception:  # pragma: no cover
    pass

# Make helper modules importable when nnU-Net imports the native PengwinTrainer
# module from its supported `nnUNetTrainer/variants` package path.
_ROOT = Path(os.environ.get("PENGWIN_ROOT", "/workspace"))
_CODE_TASK1 = _ROOT / "code_task1"
if str(_CODE_TASK1) not in sys.path:
    sys.path.insert(0, str(_CODE_TASK1))
# [V0.x][2026-06-01] STU-Net 백본 (vendored, Apache-2.0) — TotalSegmentator 뼈 사전학습 전이용.
try:
    from stunet import STUNet, STUNET_VARIANTS
    _STUNET_AVAILABLE = True
except Exception:  # pragma: no cover
    STUNet = None
    STUNET_VARIANTS = {}
    _STUNET_AVAILABLE = False
try:
    from loss import (
        BoundaryFragmentV3Loss,
        BoundaryFragmentV3CoreRecallRidgeLoss,
        BoundaryFragmentV3CoreRecallRidgeMassCapLoss,
        BoundaryFragmentV3CoreRecallRidgeVerySoftPositiveMassCapLoss,
        BoundaryFragmentV3CoreRecallRidgeLogitCalibratedLoss,
        BoundaryFragmentV3CoreRecallRidgePrecisionLogitCalibratedLoss,
        BoundaryFragmentV3CoreRecallRidgeCandidateBinaryBarrierHeadLoss,
        BoundaryFragmentV3CoreRecallRidgeDenseCandidateCore025StrongPeakContactShellContrastBarrierHeadLoss,
        BoundaryFragmentV3Core025StrongPeakXYZAffinityHeadLoss,
        BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss,
        BoundaryFragmentV3Core025StrongPeakAffinity13ContactHardNegativeSeedHealedHeadLoss,
        BoundaryFragmentV3Core025StrongPeakMutex13SeedHealedHeadLoss,
        BoundaryFragmentV3Core025StrongPeakMutex13SupportLeakGuardSeedHealedHeadLoss,
        BoundaryFragmentV3Core025StrongPeakMutex13SeedPeakSupportLeakGuardSeedHealedHeadLoss,
        BoundaryFragmentV3Core025StrongPeakMutex13SeedPeakBodySupportLeakGuardSeedHealedHeadLoss,
        BoundaryFragmentV3Core025StrongPeakMutex13FractureSeedCalibratedSupportLeakGuardSeedHealedHeadLoss,
        BoundaryFragmentV3Core025StrongPeakPairwiseSoftmaxMutex13SeedHealedHeadLoss,
        BoundaryFragmentV3Core025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedHeadLoss,
        BoundaryFragmentV3Core025StrongPeakNoContactPairwiseSoftmaxSeedHealedV273HeadLoss,
        BoundaryFragmentV3Core025StrongPeakNoContactSeparatorGapV277HeadLoss,
        BoundaryFragmentV3Core025StrongPeakNoContactABBCV288HeadLoss,
        BoundaryFragmentV3Core025StrongPeakNoContactABBCV291HeadLoss,
        BoundaryFragmentV3RidgeFitDeepSupervisionWrapper,
        BoundaryFragmentV3RidgeFitLoss,
        BICMV5SparseDeepSupervisionWrapper,
        BICMV5SupportGeometryLoss,
        BICMV5SparseLoss,
        BICMV62SemanticBoundaryLoss,
        BICMV64TopologyPrecisionLoss,
        BICMV65BalancedContactCoreLoss,
        BICMV67SemanticEdgePairLoss,
        ContactEnergyLoss3D,
        ContactFuseLoss3D,
        DC_CE_BD_loss,
        DC_CE_TV_BD_loss,
        compute_median_frequency_class_weights,
    )
    _BOUNDARY_LOSS_AVAILABLE = True
except ImportError:
    _BOUNDARY_LOSS_AVAILABLE = False

try:
    from fuseformer import ContactLegacyFuseUNet
    _FUSEFORMER_AVAILABLE = True
except ImportError:
    _FUSEFORMER_AVAILABLE = False


def _pelvic_same_anatomy_contact_mask(inst: np.ndarray) -> np.ndarray:
    """Narrow contact target: 6-neighbor different fragments in the same bone.

    [QC][Performance][Scope:dataloader_target]
    This runs for every sampled crop. The previous implementation dilated every
    fragment separately and made V4.2 CPU-bound before epoch 0 finished. The
    adjacency formulation is the same local topology contract but scales with
    three axis comparisons rather than number_of_fragments * crop_volume.
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


def _instance_core_heatmap(inst: np.ndarray,
                           centers_zyx: np.ndarray | None = None,
                           crop_lbs: np.ndarray | None = None) -> np.ndarray:
    """Per-fragment core target for the current crop.

    [REPRO][Risk:Major][Scope:target_profile]
    The function is called at dataloader time, not dataset-build time. Therefore
    the core target must be tied to an explicit loss/profile environment setting
    so old V4 evidence remains reproducible. Defaulting to `edt_heatmap` keeps
    the original V4 baseline stable; V4.1/V4.2 profiles opt into
    `compact_marker` before dataloaders are constructed.
    """
    core = np.zeros(inst.shape, dtype=np.float32)
    profile = os.environ.get("PENGWIN_FACTOR_CORE_TARGET", "edt_heatmap").strip().lower()
    radius = max(0, int(os.environ.get("PENGWIN_FACTOR_CORE_RADIUS_VOX", "2")))
    sigma = max(1e-3, float(os.environ.get("PENGWIN_FACTOR_CORE_SIGMA_VOX", "6.0")))
    for fragment_id in [int(v) for v in np.unique(inst) if 1 <= int(v) <= MAX_INSTANCE_ID]:
        mask = inst == fragment_id
        if not mask.any():
            continue
        coords = np.argwhere(mask)
        center: np.ndarray
        if centers_zyx is not None and fragment_id < len(centers_zyx) and np.isfinite(centers_zyx[fragment_id]).all():
            origin = np.asarray(crop_lbs if crop_lbs is not None else (0, 0, 0), dtype=np.float32)
            requested = np.rint(np.asarray(centers_zyx[fragment_id], dtype=np.float32) - origin).astype(int)
            if np.all(requested >= 0) and np.all(requested < np.asarray(mask.shape)) and mask[tuple(requested)]:
                center = requested
            else:
                delta = coords.astype(np.float32) - requested.reshape(1, 3).astype(np.float32)
                center = coords[int(np.argmin(np.sum(delta * delta, axis=1)))]
        else:
            center = coords[len(coords) // 2]
        if profile in {"center_gaussian", "gaussian_heatmap", "fast_heatmap"}:
            # [QC][Performance][Scope:dataloader_target]
            # V4.3 keeps a graded center target but avoids per-crop EDT. The
            # sidecar center is deterministic and the Gaussian is masked to the
            # fragment, so every fragment has a smooth peak without CPU-heavy
            # distance transforms.
            delta = coords.astype(np.float32) - center.reshape(1, 3).astype(np.float32)
            values = np.exp(-np.sum(delta * delta, axis=1) / (2.0 * sigma * sigma)).astype(np.float32)
            core[tuple(coords.T)] = np.maximum(core[tuple(coords.T)], values)
            continue
        if profile in {"edt_heatmap", "legacy_heatmap"}:
            from scipy import ndimage as ndi

            dist = ndi.distance_transform_edt(mask)
            max_dist = float(dist.max())
            if max_dist <= 0:
                core[mask] = 1.0
                continue
            core[mask] = np.maximum(core[mask], (dist[mask] / max_dist).astype(np.float32))
            continue
        marker = np.zeros_like(mask, dtype=bool)
        marker[tuple(center)] = True
        if radius > 0:
            # [QC][Performance]
            # Build the compact marker from a local ball around the sidecar center.
            # This is O(radius^3) instead of O(crop volume) and keeps the training
            # loop from spending minutes in distance transforms.
            lbs = np.maximum(center - radius, 0)
            ubs = np.minimum(center + radius + 1, np.asarray(mask.shape))
            local_slices = tuple(slice(int(a), int(b)) for a, b in zip(lbs, ubs))
            local_shape = tuple(int(b - a) for a, b in zip(lbs, ubs))
            zz, yy, xx = np.indices(local_shape)
            dz = zz + int(lbs[0]) - int(center[0])
            dy = yy + int(lbs[1]) - int(center[1])
            dx = xx + int(lbs[2]) - int(center[2])
            local_ball = (dz * dz + dy * dy + dx * dx) <= radius * radius
            marker[local_slices] |= local_ball & mask[local_slices]
        if not marker.any():
            marker[tuple(center)] = True
        core[marker] = 1.0
    return core


def _instance_offset_target(inst: np.ndarray,
                            centers_zyx: np.ndarray,
                            crop_lbs: np.ndarray,
                            offset_scale: float = 64.0) -> np.ndarray:
    """Offset from crop voxel to the preprocessed fragment center."""
    offset = np.zeros((3, *inst.shape), dtype=np.float32)
    if not (inst > 0).any():
        return offset
    grid = np.indices(inst.shape, dtype=np.float32)
    origin = np.asarray(crop_lbs, dtype=np.float32).reshape(3, 1, 1, 1)
    abs_grid = grid + origin
    scale = float(offset_scale)
    for fragment_id in [int(v) for v in np.unique(inst) if 1 <= int(v) <= MAX_INSTANCE_ID]:
        if fragment_id >= len(centers_zyx):
            continue
        center = centers_zyx[fragment_id]
        if not np.isfinite(center).all():
            continue
        mask = inst == fragment_id
        offset[:, mask] = ((center.reshape(3, 1) - abs_grid[:, mask]) / scale).astype(np.float32)
    return offset


def _instance_edge_break_target(inst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 3-axis same-anatomy break-edge targets for fragment topology.

    `edge_break[a, z, y, x]` describes the edge from voxel `(z,y,x)` to the
    next voxel in axis `a`. Positive edges are only same-anatomy,
    different-fragment pelvic adjacencies. All in-volume adjacencies are valid
    negatives otherwise, including same-fragment interior edges, pelvis-to-
    background edges, external cortical surface, femur/vertebra-like hard
    negatives, and pure background. Keeping the background/external edges in
    the loss is important: when valid edges were limited to pelvic support, the
    edge head was unconstrained outside the support and lit up most of the CT.
    Raw adjacency positives were only about 7e-5 of all in-volume edges in the
    overfit set, so training uses a small support-limited dilation controlled
    by `PENGWIN_EDGE_BREAK_DILATION` (default: 1 voxel).
    """
    inst = np.where((inst >= 1) & (inst <= MAX_INSTANCE_ID), inst, 0).astype(np.int16, copy=False)
    edge_break = np.zeros((3, *inst.shape), dtype=np.float32)
    edge_valid = np.zeros((3, *inst.shape), dtype=np.float32)
    support_touch = np.zeros((3, *inst.shape), dtype=bool)
    for axis in range(3):
        a_sl = [slice(None)] * 3
        b_sl = [slice(None)] * 3
        a_sl[axis] = slice(0, -1)
        b_sl[axis] = slice(1, None)
        a = inst[tuple(a_sl)]
        b = inst[tuple(b_sl)]
        in_volume = np.ones(a.shape, dtype=bool)
        a_fg = a > 0
        b_fg = b > 0
        same_anatomy = a_fg & b_fg & (
            ((a.astype(np.int32) - 1) // 50) == ((b.astype(np.int32) - 1) // 50)
        )
        edge_valid[(axis, *a_sl)] = in_volume.astype(np.float32)
        support_touch[(axis, *a_sl)] = a_fg | b_fg
        edge_break[(axis, *a_sl)] = (same_anatomy & (a != b)).astype(np.float32)
    radius = int(os.environ.get("PENGWIN_EDGE_BREAK_DILATION", "1"))
    if radius > 0 and edge_break.any():
        from scipy.ndimage import binary_dilation

        structure = np.ones((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=bool)
        for axis in range(3):
            edge_break[axis] = (
                binary_dilation(edge_break[axis] > 0.5, structure=structure)
                & support_touch[axis]
            ).astype(np.float32)
    return edge_break, edge_valid


def _instance_hard_negative_mask(inst: np.ndarray,
                                 data_crop: np.ndarray,
                                 external_band_vox: int = 3) -> np.ndarray:
    """Return outside-support bone/context negatives for the factorized heads.

    V4 keeps hard-negative supervision outside pelvic fragment support. Contact
    false positives inside support are handled by the contact loss; marking them
    hard-negative would make the support and suppressor heads contradictory.
    """
    from scipy import ndimage as ndi

    support = (inst >= 1) & (inst <= MAX_INSTANCE_ID)
    outside = ~support
    if not support.any():
        return np.zeros_like(support, dtype=bool)
    distance_to_support = ndi.distance_transform_edt(outside)
    external_band = outside & (distance_to_support <= int(external_band_vox))
    bone_like = np.zeros_like(support, dtype=bool)
    if data_crop.ndim == 4 and data_crop.shape[0] >= 1:
        img = np.asarray(data_crop[0], dtype=np.float32)
        finite = np.isfinite(img)
        if finite.any():
            if float(np.nanmax(img)) <= 3.0 and float(np.nanmin(img)) >= -1.0:
                bone_like = finite & (img > 0.30)
            else:
                bone_like = finite & (img > 150.0)
    if data_crop.ndim == 4 and data_crop.shape[0] >= 6:
        bone_like |= np.asarray(data_crop[5] > 0.35, dtype=bool)
    return outside & (external_band | bone_like)


class PengwinLegacyTopologyDataLoader3D(nnUNetDataLoader3D):
    """Crop Dataset537 and build Contact-Instance targets inside nnU-Net.

    We do not store dense offset/contact/core sidecars because they duplicate
    the very large preprocessed volume. The sidecar contains centers and
    sampling coordinates; dense target tensors are derived after the exact crop
    is known, guaranteeing alignment with nnU-Net's resampled `_seg.npy`.
    """

    def __init__(self, *args,
                 sidecar_dir: Path,
                 edge_center_probability: float = 0.55,
                 support_negative_center_probability: float = 0.20,
                 tiny_center_probability: float = 0.15,
                 hard_negative_center_probability: float = 0.10,
                 core_center_probability: float = 0.0,
                 mined_center_probability: float = 0.0,
                 mined_centers_json: Path | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sidecar_dir = Path(sidecar_dir)
        self.edge_center_probability = float(edge_center_probability)
        self.support_negative_center_probability = float(support_negative_center_probability)
        self.tiny_center_probability = float(tiny_center_probability)
        self.hard_negative_center_probability = float(hard_negative_center_probability)
        self.core_center_probability = float(core_center_probability)
        self.mined_center_probability = float(mined_center_probability)
        self.mined_centers = self._load_mined_centers(mined_centers_json)

    @staticmethod
    def _load_mined_centers(path: Path | None) -> dict[str, np.ndarray]:
        if path is None or str(path).strip() == "":
            return {}
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"mined support center JSON missing: {path}")
        payload = json.loads(path.read_text())
        raw = payload.get("by_key") if isinstance(payload, dict) else None
        if raw is None and isinstance(payload, dict):
            raw = {
                str(row.get("key") or row.get("case") or ""): row.get("centers_zyx", [])
                for row in payload.get("cases", [])
            }
        if not isinstance(raw, dict):
            raise ValueError(f"invalid mined center JSON; expected dict with by_key/cases: {path}")
        centers: dict[str, np.ndarray] = {}
        for key, values in raw.items():
            if not key:
                continue
            arr = np.asarray(values, dtype=np.int32)
            if arr.size == 0:
                continue
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"invalid mined centers for {key}: expected [N,3], got {arr.shape}")
            normalized = key if str(key).startswith("PENGWIN_") else f"PENGWIN_{int(key):03d}"
            centers[normalized] = arr
        return centers

    def _sidecar_path(self, key: str) -> Path:
        return self.sidecar_dir / f"{key}.npz"

    def _load_sidecar(self, key: str):
        path = self._sidecar_path(key)
        if not path.exists():
            raise FileNotFoundError(
                f"Contact-Instance sidecar missing: {path}. Run "
                "`python code_task1/preprocessing.py build-instance-sidecars --dataset 537`."
            )
        return np.load(path)

    def _bbox_from_center(self, shape: tuple[int, ...], center: np.ndarray) -> tuple[list[int], list[int]]:
        need_to_pad = self.need_to_pad.copy()
        dim = len(shape)
        for d in range(dim):
            if need_to_pad[d] + shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - shape[d]
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]
        bbox_lbs = [
            int(min(max(lbs[i], int(center[i]) - self.patch_size[i] // 2), ubs[i]))
            for i in range(dim)
        ]
        bbox_ubs = [bbox_lbs[i] + int(self.patch_size[i]) for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def _choose_priority_center(self, sidecar, key: str | None = None) -> np.ndarray | None:
        draw = np.random.random()
        if (
            key is not None
            and self.mined_center_probability > 0
            and key in self.mined_centers
            and draw < self.mined_center_probability
        ):
            loc = self.mined_centers[key]
            if loc.shape[0] > 0:
                # [DATA][Risk:Major][Scope:patch_full_volume_mismatch]
                # These centers are mined from fixed full-volume support FP/FN
                # components. They are not a post-hoc threshold: they only decide
                # which training crop to show after a failed full-volume probe.
                return loc[np.random.choice(loc.shape[0])].astype(int, copy=False)
            draw = np.random.random()
        else:
            draw = np.random.random()
        pools: list[str]
        core_cut = float(np.clip(self.core_center_probability, 0.0, 1.0))
        if draw < core_cut:
            centers = (
                np.asarray(sidecar["centers_zyx"], dtype=np.float32)
                if "centers_zyx" in sidecar.files
                else np.zeros((0, 3), dtype=np.float32)
            )
            sizes = (
                np.asarray(sidecar["fragment_sizes"], dtype=np.int64)
                if "fragment_sizes" in sidecar.files
                else np.zeros((centers.shape[0],), dtype=np.int64)
            )
            # [DATA][Risk:Major][Scope:core_positive_crop_exposure]
            # V152/V153 showed the hard sacrum fragments can have no row6 seed
            # at inference. This branch changes only crop selection, not the
            # label tensor or decoder, so the sampler hypothesis is isolated.
            if centers.ndim == 2 and centers.shape[1] == 3 and sizes.shape[0] == centers.shape[0]:
                valid = np.isfinite(centers).all(axis=1) & (sizes > 0)
            else:
                valid = np.zeros((0,), dtype=bool)
            if np.any(valid):
                loc = np.rint(centers[np.flatnonzero(valid)]).astype(np.int32, copy=False)
                return loc[np.random.choice(loc.shape[0])].astype(int, copy=False)
            draw = np.random.random()

        residual = max(0.0, 1.0 - core_cut)
        edge_cut = core_cut + residual * self.edge_center_probability
        support_cut = edge_cut + residual * self.support_negative_center_probability
        tiny_cut = support_cut + residual * self.tiny_center_probability
        hard_cut = tiny_cut + residual * self.hard_negative_center_probability
        if draw < edge_cut:
            # `contact_locations` are same-anatomy different-fragment voxels.
            # They are also the best available crop centers for the relational
            # edge-break target without duplicating dense edge sidecars.
            pools = ["contact_locations", "tiny_locations", "support_locations", "hard_negative_locations"]
        elif draw < support_cut:
            # These are the near-negative samples that were missing in the
            # broad-contact experiments: ordinary fragment interior/support
            # where contact/edge should be suppressed.
            pools = ["support_locations", "contact_locations", "tiny_locations", "hard_negative_locations"]
        elif draw < tiny_cut:
            pools = ["tiny_locations", "contact_locations", "support_locations", "hard_negative_locations"]
        elif draw < hard_cut:
            pools = ["hard_negative_locations", "support_locations", "contact_locations"]
        else:
            pools = ["support_locations", "contact_locations", "tiny_locations"]
        for name in pools:
            loc = np.asarray(sidecar[name]) if name in sidecar.files else np.zeros((0, 3), dtype=np.int32)
            if loc.shape[0] > 0:
                return loc[np.random.choice(loc.shape[0])].astype(int, copy=False)
        return None

    def _crop_pad(self, arr: np.ndarray,
                  bbox_lbs: list[int],
                  bbox_ubs: list[int],
                  pad_value: float = 0) -> tuple[np.ndarray, np.ndarray]:
        shape = arr.shape[-3:]
        valid_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
        valid_ubs = np.minimum(shape, bbox_ubs)
        spatial_slice = tuple(slice(int(i), int(j)) for i, j in zip(valid_lbs, valid_ubs))
        if arr.ndim == 4:
            sliced = arr[(slice(None), *spatial_slice)]
            padding = ((0, 0), *[(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(3)])
        else:
            sliced = arr[spatial_slice]
            padding = tuple((-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(3))
        return np.pad(sliced, padding, "constant", constant_values=pad_value), np.asarray(bbox_lbs, dtype=np.int32)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        morphology_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
        support_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        core_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        contact_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        hard_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        offset_all = np.zeros((self.batch_size, 3, *self.patch_size), dtype=np.float32)
        edge_break_all = np.zeros((self.batch_size, 3, *self.patch_size), dtype=np.float32)
        edge_valid_all = np.zeros((self.batch_size, 3, *self.patch_size), dtype=np.float32)
        instance_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)

        for j, key in enumerate(selected_keys):
            data, seg, _properties = self._data.load_case(key)
            shape = data.shape[1:]
            force_fg = self.get_do_oversample(j)
            sidecar = self._load_sidecar(key)
            center = self._choose_priority_center(sidecar, key=key) if force_fg else None
            if center is None:
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=None)
            else:
                bbox_lbs, bbox_ubs = self._bbox_from_center(shape, center)
            data_crop, crop_lbs = self._crop_pad(data, bbox_lbs, bbox_ubs, pad_value=0)
            seg_crop, _ = self._crop_pad(seg[0], bbox_lbs, bbox_ubs, pad_value=0)
            instance_source = seg_crop
            if "instance" in sidecar.files:
                dense_instance = np.asarray(sidecar["instance"], dtype=np.uint16)
                # [DATA][Risk:High][Scope:instance_target_alignment]
                # Dataset537 V5 stores semantic labels 0..4 in `_seg`, not fragment
                # instance IDs. Factorized instance supervision must therefore use
                # the dense sidecar instance map when present; otherwise semantic
                # classes would be treated as fake fragments and embedding/contact
                # losses would optimize the wrong grouping problem.
                if tuple(dense_instance.shape[-3:]) != tuple(shape):
                    raise ValueError(
                        f"{key}: sidecar instance shape {dense_instance.shape[-3:]} "
                        f"does not match preprocessed shape {tuple(shape)}"
                    )
                instance_source, _ = self._crop_pad(dense_instance, bbox_lbs, bbox_ubs, pad_value=0)
            inst = np.where((instance_source >= 1) & (instance_source <= MAX_INSTANCE_ID), instance_source, 0).astype(np.uint16, copy=False)
            support = inst > 0
            contact = _pelvic_same_anatomy_contact_mask(inst)
            hard = _instance_hard_negative_mask(inst, data_crop) & ~contact
            morphology = np.zeros_like(inst, dtype=np.int16)
            morphology[support & ~contact] = CONTACT_INSTANCE_MORPH_SUPPORT_CH
            morphology[contact] = CONTACT_INSTANCE_MORPH_CONTACT_CH
            morphology[hard] = CONTACT_INSTANCE_MORPH_HARD_NEGATIVE_CH
            centers = np.asarray(sidecar["centers_zyx"], dtype=np.float32)
            edge_break, edge_valid = _instance_edge_break_target(inst)

            data_all[j] = data_crop
            morphology_all[j, 0] = morphology
            support_all[j, 0] = support.astype(np.float32)
            core_all[j, 0] = _instance_core_heatmap(inst, centers, crop_lbs)
            contact_all[j, 0] = contact.astype(np.float32)
            hard_all[j, 0] = hard.astype(np.float32)
            offset_all[j] = _instance_offset_target(inst, centers, crop_lbs)
            edge_break_all[j] = edge_break
            edge_valid_all[j] = edge_valid
            instance_all[j, 0] = inst.astype(np.int16, copy=False)

        return {
            "data": data_all,
            "target": {
                "morphology": morphology_all,
                "support": support_all,
                "core": core_all,
                "contact": contact_all,
                "hard_negative": hard_all,
                "offset": offset_all,
                "edge_break": edge_break_all,
                "edge_valid": edge_valid_all,
                "instance": instance_all,
            },
            "keys": selected_keys,
        }


def _present_fragment_slices(inst: np.ndarray) -> list:
    """[(fid, bbox_slices), ...] for valid fragment ids via ONE ndimage.find_objects pass.

    Hot-path optimization: the per-fragment target builders (`_fragment_center_core`,
    `_fragment_seed_centers`, `_fragment_spatial_embedding_targets`) used to do
    ``for fid in np.unique(inst): np.argwhere(inst == fid)`` — an O(n_fragments × patch
    volume) full-volume rescan costing ~0.5-1.0 s/sample and starving the GPU. This
    returns each fragment's bounding box in a single pass; callers then scan only the
    bbox. Voxel coords found inside the bbox plus the bbox offset are BIT-IDENTICAL
    (same C-order) to ``np.argwhere(inst == fid)``, so every target is unchanged — only
    the access pattern is faster (verified by the bit-exact audit).
    """
    from scipy import ndimage as ndi

    inst_i = np.ascontiguousarray(inst, dtype=np.int32)
    maxv = int(inst_i.max(initial=0))
    if maxv <= 0:
        return []
    objs = ndi.find_objects(inst_i, max_label=min(maxv, MAX_INSTANCE_ID))
    return [(idx + 1, sl) for idx, sl in enumerate(objs) if sl is not None]


class PengwinBICMV5DataLoader3D(nnUNetDataLoader3D):
    """V5 dataloader that centers crops on sparse decoder-critical labels.

    Dataset537 V5 keeps a simple 5-class semantic target. The first 003 overfit
    showed that default nnU-Net foreground crops learn shell/support but never
    see enough core/contact voxels to make a marker-based decoder work. This
    dataloader changes only the forced-foreground class choice; transforms,
    padding, batching, and validation remain native nnU-Net behavior.
    """

    SPARSE_CLASS_DISTRIBUTION = (
        (3, 0.55),  # core marker: required for every decoded fragment seed
        (4, 0.30),  # contact surface: required to split same-anatomy fragments
        (2, 0.10),  # shell: high-volume support near-negative
        (1, 0.05),  # exterior context: suppress support leakage
    )
    SUPPORT_MIXED_CLASS_DISTRIBUTION = (
        # [DATA][Risk:High][Scope:case003_support_fp]
        # V6.2 removed contact supervision and still produced support masks
        # roughly 2-3x larger than GT. That isolates the next question to
        # sampling: the sparse profile always forces a positive foreground crop
        # and mostly centers core/contact, so the support head rarely sees
        # exterior/context-dominated negatives. This profile keeps marker/core
        # exposure but deliberately spends much more forced-foreground budget on
        # class 1 and class 2 boundaries; the remaining non-forced batches come
        # from oversample_foreground_percent < 1.0 in the trainer.
        (1, 0.40),  # exterior context: support negative adjacent to bone
        (2, 0.30),  # shell/support: positive support geometry
        (3, 0.25),  # core marker: preserve seed coverage
        (4, 0.05),  # contact: kept only as rare support-positive context
    )
    CONTACT_MIXED_CLASS_DISTRIBUTION = (
        # [DATA][Risk:High][Scope:contact_recall]
        # V7 anatomy-context support passed full-volume HD95, but contact recall
        # stayed low when only 5% of forced crops were class-4 centered. This
        # profile changes only crop centers for the next ablation: contact gets
        # enough positive patches while exterior/shell/core remain represented so
        # broad support/contact FP does not become invisible to the loss.
        (4, 0.35),  # contact: decoder split ridge
        (1, 0.25),  # exterior context: contact/support negative
        (2, 0.25),  # shell/support: support non-contact near-negative
        (3, 0.15),  # core marker: seed stability
    )
    CONTACT15_MIXED_CLASS_DISTRIBUTION = (
        # [DATA][Risk:High][Scope:single_variable_sampling]
        # `bicm_v7_contact_mixed` at 35% contact centers destabilized support
        # and collapsed contact in patch validation. This middle profile tests
        # whether the original 5% contact exposure was simply too sparse while
        # keeping most crop budget on exterior/shell/core context.
        (1, 0.35),  # exterior context: suppress broad support/contact
        (2, 0.25),  # shell/support: support non-contact near-negative
        (3, 0.25),  # core marker: seed stability
        (4, 0.15),  # contact: moderate positive exposure
    )
    CONTACT_ROI_CLASS_DISTRIBUTION = (
        # [DATA][Risk:High][Scope:contact_positive_exposure]
        # V10 full-volume case003 had good support/core but predicted zero
        # contact even in LeftHip, the only ROI with GT contact. V11 changes
        # exposure, not target or decoder: when a contact-positive ROI is
        # sampled, most forced foreground crops are centered on class 4. The
        # trainer-level ROI weighting below still keeps no-contact ROIs in the
        # stream so absent-contact negatives remain visible.
        (4, 0.60),  # contact: make the positive ridge visible to the head
        (2, 0.20),  # shell/support: support-local non-contact near-negative
        (3, 0.15),  # core marker: seed stability
        (1, 0.05),  # exterior context: suppress support/contact leakage
    )

    def __init__(self, *args, forced_class_distribution=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_class_distribution = tuple(forced_class_distribution or self.SPARSE_CLASS_DISTRIBUTION)

    @staticmethod
    def _class_key(class_locations: dict, label: int):
        if not class_locations:
            return None
        for key, value in class_locations.items():
            try:
                if int(key) == int(label) and len(value) > 0:
                    return key
            except (TypeError, ValueError):
                continue
        return None

    def _pick_forced_class(self, class_locations: dict):
        available = [
            (label, prob)
            for label, prob in self.forced_class_distribution
            if self._class_key(class_locations, label) is not None
        ]
        if not available:
            return None
        labels = [label for label, _ in available]
        probs = np.asarray([prob for _, prob in available], dtype=np.float64)
        probs = probs / probs.sum()
        return self._class_key(class_locations, int(np.random.choice(labels, p=probs)))

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool,
                 class_locations: dict | None,
                 overwrite_class=None, verbose: bool = False):
        # [DATA][Risk:Major][Scope:sampling]
        # Core/contact targets can be single-digit voxels per ROI. When a crop
        # is already forced to foreground, choosing the rare class explicitly is
        # the least invasive way to test whether collapse was caused by sample
        # density rather than target topology or model architecture.
        if force_fg and overwrite_class is None and class_locations:
            overwrite_class = self._pick_forced_class(class_locations)
        return super().get_bbox(
            data_shape,
            force_fg,
            class_locations,
            overwrite_class=overwrite_class,
            verbose=verbose,
        )


BICM_V6_OUTPUT_CHANNELS = 4
BICM_V8_OUTPUT_CHANNELS = 5


class PengwinBICMV5EdgeAffinityDataLoader3D(PengwinLegacyTopologyDataLoader3D):
    """V38 dataloader: semantic V5 heads plus instance edge-affinity sidecars.

    [AUDIT][Risk:High][Scope:v38_target_alignment]
    V5/V37 training samples expose only semantic labels (`0..4`), which cannot
    tell the loss whether two adjacent support voxels belong to the same GT
    fragment or two different fragments. The V38 sidecar stores the original
    instance map in the exact preprocessed geometry; this loader uses it only
    for supervision targets, never as a model input.
    """

    def __init__(self, *args,
                 instance_center_core_radius_vox: float | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_center_core_radius_vox = (
            None if instance_center_core_radius_vox is None else float(instance_center_core_radius_vox)
        )

    @staticmethod
    def _fragment_center_core(inst: np.ndarray,
                              centers_zyx: np.ndarray,
                              bbox_lbs: list[int],
                              radius_vox: float) -> np.ndarray:
        """Create one compact core marker per fragment present in the crop.

        [DATA][Risk:High][Scope:fragment_seed_target]
        Semantic V5 core can connect multiple fragments into one anatomy-level
        marker, which V39/V40 measured as `seed_components=1` despite perfect
        fragment coverage. This target uses original instance IDs from the
        sidecar and therefore supervises fragment-scale markers directly.
        """
        core = np.zeros(inst.shape, dtype=bool)
        centers = np.asarray(centers_zyx, dtype=np.float32)
        radius = max(float(radius_vox), 0.0)
        rad_i = int(np.ceil(radius))
        bbox = np.asarray(bbox_lbs, dtype=np.float32)
        for fid, _sl in _present_fragment_slices(inst):
            coords = np.argwhere(inst[_sl] == fid)
            if coords.size == 0:
                continue
            coords = coords + np.asarray([s.start for s in _sl], dtype=coords.dtype)
            center_global = centers[fid] if fid < centers.shape[0] else np.asarray(coords.mean(axis=0), dtype=np.float32)
            if not np.isfinite(center_global).all() or np.allclose(center_global, 0.0):
                center_local = coords.mean(axis=0)
            else:
                center_local = center_global - bbox
            nearest = coords[np.argmin(((coords.astype(np.float32) - center_local.reshape(1, 3)) ** 2).sum(axis=1))]
            z, y, x = [int(v) for v in nearest]
            z0, z1 = max(0, z - rad_i), min(inst.shape[0], z + rad_i + 1)
            y0, y1 = max(0, y - rad_i), min(inst.shape[1], y + rad_i + 1)
            x0, x1 = max(0, x - rad_i), min(inst.shape[2], x + rad_i + 1)
            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            ball = ((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) <= (radius ** 2)
            patch = core[z0:z1, y0:y1, x0:x1]
            same_fragment = inst[z0:z1, y0:y1, x0:x1] == fid
            written = ball & same_fragment
            patch[written] = True
            if not written.any():
                core[z, y, x] = True
        return core

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        support_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        core_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        contact_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        exterior_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        edge_break_all = np.zeros((self.batch_size, 3, *self.patch_size), dtype=np.float32)
        edge_valid_all = np.zeros((self.batch_size, 3, *self.patch_size), dtype=np.float32)
        offset_all = np.zeros((self.batch_size, 3, *self.patch_size), dtype=np.float32)
        instance_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
        semantic_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)

        for j, key in enumerate(selected_keys):
            data, seg, _properties = self._data.load_case(key)
            shape = data.shape[1:]
            sidecar = self._load_sidecar(key)
            inst_full = np.asarray(sidecar["instance"], dtype=np.uint16)
            if tuple(inst_full.shape) != tuple(shape):
                raise RuntimeError(
                    f"BICM V38 sidecar shape mismatch for {key}: "
                    f"instance={tuple(inst_full.shape)} data={tuple(shape)}"
                )
            force_fg = self.get_do_oversample(j)
            center = self._choose_priority_center(sidecar, key=key) if force_fg else None
            if center is None:
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=None)
            else:
                bbox_lbs, bbox_ubs = self._bbox_from_center(shape, center)

            data_crop, _crop_lbs = self._crop_pad(data, bbox_lbs, bbox_ubs, pad_value=0)
            sem_crop, _ = self._crop_pad(seg[0], bbox_lbs, bbox_ubs, pad_value=0)
            inst_crop, _ = self._crop_pad(inst_full, bbox_lbs, bbox_ubs, pad_value=0)
            inst = np.where((inst_crop >= 1) & (inst_crop <= MAX_INSTANCE_ID), inst_crop, 0).astype(np.uint16, copy=False)
            support = inst > 0
            contact = _pelvic_same_anatomy_contact_mask(inst)
            edge_break, edge_valid = _instance_edge_break_target(inst)
            offset = _instance_offset_target(
                inst,
                np.asarray(sidecar["centers_zyx"], dtype=np.float32),
                np.asarray(bbox_lbs, dtype=np.float32),
            )
            if self.instance_center_core_radius_vox is None:
                core = (sem_crop == 3)
            else:
                core = self._fragment_center_core(
                    inst,
                    np.asarray(sidecar["centers_zyx"], dtype=np.float32),
                    bbox_lbs,
                    self.instance_center_core_radius_vox,
                )

            data_all[j] = data_crop
            # [DATA][Risk:High][Scope:v68_hybrid_target]
            # V68 keeps the semantic support/exterior branch because V67
            # overfit support reliably, but it trains contact/core from
            # instance sidecars. Store the exact cropped V5 semantic labels so
            # the hybrid loss cannot reconstruct support from a mismatched
            # coordinate space or from instance IDs after augmentation.
            semantic_all[j, 0] = sem_crop.astype(np.int16, copy=False)
            support_all[j, 0] = support.astype(np.float32)
            core_all[j, 0] = core.astype(np.float32)
            contact_all[j, 0] = contact.astype(np.float32)
            exterior_all[j, 0] = (sem_crop == 1).astype(np.float32)
            edge_break_all[j] = edge_break
            edge_valid_all[j] = edge_valid
            # [DATA][Risk:High][Scope:v105_offset_assignment]
            # Offset supervision is derived from the same cropped sidecar instance
            # map as edge/core targets. It is stored in preprocessed voxel units
            # normalized by 64; consumers that still use edge rows ignore this key.
            offset_all[j] = offset
            instance_all[j, 0] = inst.astype(np.int16, copy=False)

        return {
            "data": data_all,
            "target": {
                "support": support_all,
                "core": core_all,
                "contact": contact_all,
                "exterior": exterior_all,
                "edge_break": edge_break_all,
                "edge_valid": edge_valid_all,
                "offset": offset_all,
                "instance": instance_all,
                "semantic": semantic_all,
            },
            "keys": selected_keys,
        }


class _GroupedSplitMixin:
    """[2026-06-05] Source-case grouped K-fold split — single source of truth.

    Stock nnUNet do_split runs random KFold over per-ROI identifiers, so a source
    CT with several anatomy ROIs (Dataset538) leaks across train/val. This mixin
    regenerates splits_final.json grouped by source case (generate_crossval_split
    over group IDs, seed 12345) before stock do_split reads it. Single-ROI datasets
    (Ds539) are delegated unchanged (grouped == ungrouped). Both the anatomy trainer
    (via PengwinTrainer) and the fracture chain (PengwinTrainerBoundaryFragmentV3)
    inherit this so Stage A / Stage B share ONE patient-grouped split (same seed
    12345 over the same 340 source cases → identical fold partition in both).
    """

    @staticmethod
    def _source_case_from_identifier(identifier: str) -> str:
        """nnU-Net 샘플 identifier에서 zero-padding된 source case ID를 반환한다."""
        stem = str(identifier).replace("PENGWIN_", "")
        return stem.split("_", 1)[0].zfill(3)

    @staticmethod
    def _splits_group_consistent(splits: list, source_of) -> bool:
        """splits_final.json 의 각 fold 에서 같은 source case 가 train/val 에 동시에
        들어가지 않는지(=leakage 없음) 검사한다."""
        for fold in splits:
            tr_src = {source_of(k) for k in fold.get("train", [])}
            val_src = {source_of(k) for k in fold.get("val", [])}
            if tr_src & val_src:
                return False
        return True

    def _ensure_grouped_splits_file(self):
        """[FIX:C1] source-case 단위 grouped K-fold split 강제 (위 클래스 docstring 참조).

        stock 이 splits_final.json 을 읽기 전에 source case 단위로 묶은 grouped split 을
        미리 생성한다. 한 source 당 ROI 하나뿐인 데이터셋(Ds539)은 grouped==ungrouped 이라
        stock 에 위임한다. 기존 leaky split 은 `.leaky.bak` 로 백업 후 재생성한다.
        """
        if self.fold == "all":
            return
        from batchgenerators.utilities.file_and_folder_operations import (
            join, isfile, load_json, save_json,
        )
        from nnunetv2.training.dataloading.utils import get_case_identifiers
        from nnunetv2.utilities.crossval_split import generate_crossval_split

        splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        keys = sorted(get_case_identifiers(self.preprocessed_dataset_folder))
        groups: dict[str, list[str]] = {}
        for k in keys:
            groups.setdefault(self._source_case_from_identifier(k), []).append(k)
        multi_roi = any(len(v) > 1 for v in groups.values())

        if isfile(splits_file):
            existing = load_json(splits_file)
            if self._splits_group_consistent(existing, self._source_case_from_identifier):
                return  # 이미 group-consistent → 그대로 사용
            if not multi_roi:
                return  # 단일 ROI 데이터셋은 leakage 불가 → 그대로 사용
            backup = splits_file + ".leaky.bak"
            if not isfile(backup):
                os.replace(splits_file, backup)
            self.print_to_log_file(
                f"[FIX:C1] 기존 splits_final.json 이 source-case leakage 를 포함 → "
                f"백업({backup}) 후 grouped split 재생성"
            )
        elif not multi_roi:
            return  # 단일 ROI 데이터셋 → stock 의 ungrouped split 과 동일하므로 위임

        group_ids = sorted(groups.keys())
        group_splits = generate_crossval_split(group_ids, seed=12345, n_splits=5)
        splits = []
        for gs in group_splits:
            tr = sorted(k for g in gs["train"] for k in groups[g])
            val = sorted(k for g in gs["val"] for k in groups[g])
            splits.append({"train": tr, "val": val})
        save_json(splits, splits_file)
        self.print_to_log_file(
            f"[FIX:C1] source-case grouped {len(splits)}-fold split 생성: "
            f"{len(group_ids)} source cases → {len(keys)} ROI samples → {splits_file}"
        )

    def do_split(self):
        """Grouped split, then stock read. PengwinTrainer overrides to add case-lock."""
        self._ensure_grouped_splits_file()
        return super().do_split()


class PengwinTrainer(_GroupedSplitMixin, nnUNetTrainer):
    """현재 사용 중인 split-anatomy trainer — DC+CE baseline, SGD+Cosine, ES patience 50.

    V2 계약 사항:
        - Loss는 기본적으로 DC+CE. Boundary/Tversky 항은 명시적이고 환경 변수로
          선택되는 ablation이므로, baseline 실험은 비교 가능한 상태로 유지된다.
        - CE class weight는 명시적으로 주거나 `PENGWIN_CE_CLASS_WEIGHTS=auto`로 설정한다.
        - 단순 mirror가 해부학적 라벨을 오염시키는 Pelvic 데이터셋에서는 좌우 mirror를 끈다.
    """

    NUM_EPOCHS_DEFAULT = 1500
    ES_PATIENCE = 50
    ES_MIN_DELTA = 5e-3   # v6: 1e-3 → 5e-3 (무한 climb 방지)
    ES_MIN_EPOCHS = 100
    WARMUP_EPOCHS = 30

    # Dataset532는 작은 Sacrum/LH/RH 영역을 포함하기 때문에 foreground patch를 더 공격적으로
    # oversample한다. Contact-LegacyFuse는 sparse한 core/contact-surface class를 가지고 있어서,
    # class-3 / class-4를 명시적으로 추가 sampling한다.
    PELVIC_DATASETS_OVERSAMPLE_50 = {
        "Dataset532_PelvicAnatomyV2",
    }
    ABBC_OFFICIAL_DATASETS: set[str] = set()

    # BoundaryDoU는 기본 비활성. PENGWIN 2024 상위 CT 레시피는 DC+CE를 사용하면서
    # boundary 부담을 representation/postprocess가 떠안는 구조였다. 그래서 BD는
    # 조용히 켜지는 기본값이 아니라 "측정된 ablation"으로 다룬다.
    USE_BOUNDARY_LOSS = False
    BD_WEIGHT = 0.0
    USE_TVERSKY_LOSS = False
    TV_WEIGHT = 0.0
    TV_ALPHA = 0.3
    TV_BETA = 0.7
    TV_ACTIVE_CLASSES: tuple[int, ...] | None = None
    TV_CLASS_WEIGHTS: tuple[float, ...] | None = None
    # CE class weight는 기본 비활성이다. "auto"로 두면 labelsTr를 스캔한다.
    USE_CLASS_WEIGHTS = False
    CE_CLASS_WEIGHTS: dict | str | None = None
    CLASS_WEIGHT_CLIP = (0.5, 2.0)
    DEFAULT_LOSS_PROFILE = "dc_ce"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "default"

    # 좌우 비대칭이 중요한 데이터셋(LH 라벨과 RH 라벨이 따로 있는 경우) — x-mirror를 비활성화한다.
    # Dataset532는 LeftHip과 RightHip을 모두 포함하므로 단순 mirror를 하면 해부학적으로 좌우가
    # 뒤바뀐 이미지가 만들어지는데 라벨은 바뀌지 않은 채로 남는다.
    DISABLE_X_MIRROR_DATASETS = {
        "Dataset532_PelvicAnatomyV2",
        # [V0.x][FIX:LR][2026-06-02] Ds539 추가 — diag_hip_precision 조사 결과 hip 저조의
        # 87.6%가 좌우 hip 스왑(laterality 혼동)이고, axis-2(L/R) mirror augmentation 이
        # L↔R 교환을 학습시키는 직접 원인. Ds532 와 동일하게 axis-2 mirror 비활성화.
        "Dataset539_PelvicFemurAnatomyV3",
    }

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_NUM_EPOCHS", self.NUM_EPOCHS_DEFAULT))
        # [QA][Test:runtime_smoke]
        # 이 override들은 의도적으로 일반적인 형태로 두었다. 그래야 stock trainer로도
        # V5의 1-epoch / 40-epoch case-locked 테스트를 돌릴 수 있다. opt-in 방식이라
        # 환경 변수가 명시되지 않으면 평소 학습에 영향을 주지 않는다.
        if os.environ.get("PENGWIN_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_VAL_ITERS"])
        # [REPRO][Risk:Major][Scope:optimizer_ablation]
        # 실험에서 PENGWIN_INITIAL_LR을 명시하지 않는 한 nnU-Net 기본값인 1e-2 SGD LR을 유지한다.
        # V6 case003 진단에서 warmup이 LR을 올리면서 support가 붕괴하는 것을 봤기 때문에,
        # 이 opt-in override로 target/model/loss/decoder를 건드리지 않고 optimizer 안정성만
        # 단독으로 테스트할 수 있게 했다.
        self.initial_lr = float(os.environ.get("PENGWIN_INITIAL_LR", "1e-2"))
        self.weight_decay = 3e-5
        ds_name = self.plans_manager.dataset_name
        oversample_profile = os.environ.get(
            "PENGWIN_OVERSAMPLE_PROFILE", self.DEFAULT_OVERSAMPLE_PROFILE
        ).strip().lower()
        if ds_name in self.ABBC_OFFICIAL_DATASETS:
            if oversample_profile in {"abbc_contact_energy_v1", "contact_fuse_v1"}:
                self.oversample_foreground_percent = 1.00
            elif oversample_profile == "weak0123":
                self.oversample_foreground_percent = 0.66
            else:
                self.oversample_foreground_percent = 0.50
        elif ds_name == "Dataset537_PelvicBICMFragmentV5":
            # [QA][Experiment:V5_sparse_sampling]
            # nnU-Net 기본 foreground sampling은 case003에서 core/contact를 전혀 예측하지 못했다.
            # sparse profile은 opt-in이라, 기본 DC+CE 결과는 재현 가능한 상태로 남는다.
            if oversample_profile == "bicm_v5_sparse":
                self.oversample_foreground_percent = 1.00
            elif oversample_profile == "bicm_v6_support_mixed":
                # [DATA][Risk:High][Scope:support_negative_exposure]
                # V6.2에서 support/core만 학습해도 여전히 support가 과대예측되는 걸 확인한 뒤,
                # V6.3은 오로지 crop 분포만 바꾼다. foreground 비율을 1.0이 아니게 두면
                # 기본 random crop이 진짜 background/외부 negative를 공급할 수 있다.
                # 이걸로 HD95가 개선된다면 원인은 표현이 아니라 sampling bias였다는 뜻이다.
                self.oversample_foreground_percent = 0.65
            elif oversample_profile == "bicm_v7_contact_mixed":
                # [DATA][Risk:High][Scope:contact_positive_exposure]
                # support_mixed와 동일한 random-background 비율을 유지한다.
                # 강제 crop이 발생할 때 어떤 foreground class를 고를지만 달라진다.
                self.oversample_foreground_percent = 0.65
            elif oversample_profile == "bicm_v7_contact15_mixed":
                # [DATA][Risk:High][Scope:contact_positive_exposure]
                # foreground-vs-random 예산은 support_mixed와 동일. 차이는 forced-class
                # 분포에서 contact 비율이 5%에서 15%로 바뀐 것 하나뿐이다.
                self.oversample_foreground_percent = 0.65
            elif oversample_profile in {"bicm_v11_contact_roi", "bicm_v18_roi_balanced"}:
                # [DATA][Risk:High][Scope:contact_positive_exposure]
                # V10에서 V9/V9.1과는 다른 실패가 나타났다: support/core는 안정적이었지만
                # contact head가 0으로 붕괴했다. 이 프로파일은 positive contact crop 노출을 늘리되,
                # 아래의 ROI sampling 확률로 no-contact ROI도 배치 흐름에 남겨서
                # absent-contact 학습이 유지되도록 한다.
                self.oversample_foreground_percent = 0.80
            else:
                self.oversample_foreground_percent = 0.33
        elif ds_name in self.PELVIC_DATASETS_OVERSAMPLE_50:
            if oversample_profile == "weak0123":
                self.oversample_foreground_percent = 0.66
            else:
                self.oversample_foreground_percent = 0.50
        else:
            self.oversample_foreground_percent = 0.33
        self._pengwin_oversample_profile = oversample_profile
        self._configure_loss_ablation_from_env()
        # Early stop 상태 변수
        self._es_best_dice = -1.0
        self._es_no_improve = 0
        self._es_triggered = False

    def _pengwin_case_lock(self, train_keys: list[str], val_keys: list[str]) -> tuple[list[str], list[str]]:
        """V5 root-cause 테스트를 위해 source-case 단위 overfit split을 강제하는 옵션 메서드.

        [DATA][Leakage:case_lock]
        Dataset537 V5에서는 source CT 하나당 여러 ROI sample이 나온다. case-locked overfit을
        제대로 하려면 같은 source case의 ROI identifier들을 train과 val 양쪽에 모두 포함시켜야
        한다. 그래야 patch validation이 다른 source case를 health check에 끼워넣지 못한다.
        이 동작은 `PENGWIN_OVERFIT_CASES` 또는 `PENGWIN_BICM_V5_CASES`가 명시적으로 설정된
        경우에만 켜진다.
        """
        raw = (
            os.environ.get("PENGWIN_OVERFIT_CASES", "").strip()
            or os.environ.get("PENGWIN_BICM_V5_CASES", "").strip()
        )
        if not raw:
            return train_keys, val_keys
        wanted = {part.strip().zfill(3) for part in raw.replace(",", " ").split() if part.strip()}
        all_keys = sorted(set(train_keys) | set(val_keys))
        selected = [
            key for key in all_keys
            if self._source_case_from_identifier(key) in wanted
        ]
        if not selected:
            raise RuntimeError(
                f"PENGWIN_OVERFIT_CASES={sorted(wanted)} matched no identifiers in "
                f"{self.plans_manager.dataset_name}"
            )
        self.print_to_log_file(
            "[PengwinTrainer] case-lock overfit split active: "
            f"cases={sorted(wanted)} identifiers={selected}"
        )
        return selected, selected

    def do_split(self):
        # grouped split (via _GroupedSplitMixin) → stock read → optional case-lock overfit.
        self._ensure_grouped_splits_file()
        train_keys, val_keys = nnUNetTrainer.do_split(self)
        return self._pengwin_case_lock(list(train_keys), list(val_keys))

    def _build_loss(self):
        """DC+CE를 기본으로 하고, 명시적이고 감사(audit) 가능한 loss ablation을 얹어서 구성한다.

        알고리즘:
            1. 부모 클래스와 동일하게 base DC_and_CE_loss를 만든다
               (batch_dice / smooth / DDP smoothing / ignore_label 의미를 그대로 보존).
               CE_CLASS_WEIGHTS가 설정돼 있으면 CE 부분에 `weight` 인자를 넘긴다.
            2. boundary/Tversky 프로파일이 명시적으로 선택돼 있으면 해당 compound loss로 감싼다.
            3. nnU-Net의 torch.compile 경로는 loss.dc가 Dice 모듈일 거라고 기대하기 때문에,
               우리 wrapper는 속성 통과(attribute pass-through)로 이를 노출한다.
            4. 마지막으로 부모처럼 DeepSupervisionWrapper로 감싼다 (multi-resolution).

        MIC-DKFZ의 loss 코드 패스를 서브클래싱하지 않는 이유:
            DC_and_CE_loss는 ignore_label과 DDP edge case들을 내부에서 다루는데,
            이걸 다시 구현하면 미묘한 버그가 생길 위험이 있다. 그래서 대체가 아니라 구성(compose)한다.

        Fallback:
            `loss.py` import가 실패하거나, USE_BOUNDARY_LOSS=False이거나, label manager가
            region 기반(multi-label, 예: 중첩된 해부 구조)이면 부모의 DC+CE로 폴백한다.
            절대 조용히 학습을 망가뜨리지 않는다.
        """
        # Region 기반 라벨 (multi-label)이면 → 부모 구현으로 위임 (boundary loss 없이).
        if self.label_manager.has_regions:
            return super()._build_loss()

        ce_kwargs = {}
        cw_tensor = None
        bicm_v5_aux_loss = None
        n_classes = self.label_manager.num_segmentation_heads
        if self.USE_CLASS_WEIGHTS and self.CE_CLASS_WEIGHTS is not None:
            cw = self._resolve_class_weights(n_classes)
            if cw is not None:
                cw_tensor = torch.as_tensor(cw, dtype=torch.float32, device=self.device)
                ce_kwargs["weight"] = cw_tensor
                self.print_to_log_file(
                    f"[PengwinTrainer v7] CE class weights: "
                    f"{[f'{w:.2f}' for w in cw]}"
                )

        if getattr(self, "USE_BICM_V67_SEMANTIC_EDGE_PAIR_OBJECTIVE", False):
            # [AUDIT][Risk:High][Scope:v67_representation_diagnostic]
            # V66은 support geometry는 통과했지만 LeftHip contact가 여전히 넓게 퍼지고
            # seed topology가 실패했다. V67은 semantic head와 V62 support 항을 그대로 두고,
            # 거기에 local contact-vs-support edge ranking을 추가한다.
            # 이건 표현/loss 진단(diagnostic)이지 159/fold0/fulltrain을 풀어주는 시도가 아니다.
            base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            aux_base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            loss = BICMV67SemanticEdgePairLoss(base, fullres_terms=True)
            bicm_v5_aux_loss = BICMV67SemanticEdgePairLoss(aux_base, fullres_terms=False)
            self.print_to_log_file(
                "[PengwinTrainer] Loss = DC+CE + V67 semantic edge-pair ranking "
                "(V62 support terms + local contact-vs-support rank + mild absent-ROI suppression)"
            )
        elif getattr(self, "USE_BICM_V65_BALANCED_CONTACT_CORE_OBJECTIVE", False):
            # [AUDIT][Risk:High][Scope:v65_loss_rebalance]
            # V64는 precision 압력이 과해서 contact가 붕괴하는 실패를 측정했다.
            # V65는 V62/V64의 "loss만 바꾸는 단일 변수" 원칙은 그대로 두면서,
            # class-4 contact를 우선 recall 쪽으로 재균형 잡는다.
            base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            aux_base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            loss = BICMV65BalancedContactCoreLoss(base, fullres_terms=True)
            bicm_v5_aux_loss = BICMV65BalancedContactCoreLoss(aux_base, fullres_terms=False)
            self.print_to_log_file(
                "[PengwinTrainer] Loss = DC+CE + V65 balanced contact/core terms "
                "(contact recall restored + mild precision/no-contact suppression)"
            )
        elif getattr(self, "USE_BICM_V64_TOPOLOGY_PRECISION_OBJECTIVE", False):
            # [AUDIT][Risk:High][Scope:v64_single_variable_loss]
            # V62 40-epoch 실험은 전체 support geometry는 통과했지만 contact precision,
            # tiny-fragment core coverage, IoU-F에서 실패했다. V64는 V5 semantic
            # head/target/decoder는 그대로 두고, 측정된 topology 오류를 잡는 쪽으로
            # loss 압력만 변경한다.
            base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            aux_base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            loss = BICMV64TopologyPrecisionLoss(base, fullres_terms=True)
            bicm_v5_aux_loss = BICMV64TopologyPrecisionLoss(aux_base, fullres_terms=False)
            self.print_to_log_file(
                "[PengwinTrainer] Loss = DC+CE + V64 topology-precision terms "
                "(no-contact suppression + contact precision + core seed balance)"
            )
        elif getattr(self, "USE_BICM_V62_SEMANTIC_BOUNDARY_OBJECTIVE", False):
            # [AUDIT][Risk:High][Scope:v62_methodology_pivot]
            # V38-V61에 걸쳐 factorized BICM head 계열을 모두 시도했지만 case003을 통과시키지 못했다.
            # V62는 의도적으로 더 단순한 V5 semantic head로 돌아가서, loss만 support boundary,
            # contact, core 항으로 바꾼다. 또 다른 숨겨진 branch를 추가하지 않으면서,
            # oracle을 통과하는 target이 boundary/HD95-aware 목적함수로 학습될 수 있는지 확인하려는 시도다.
            base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            aux_base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            loss = BICMV62SemanticBoundaryLoss(base, fullres_terms=True)
            bicm_v5_aux_loss = BICMV62SemanticBoundaryLoss(aux_base, fullres_terms=False)
            self.print_to_log_file(
                "[PengwinTrainer] Loss = DC+CE + V62 semantic-boundary terms "
                "(support Dice/boundary proxy + core/contact Tversky/Focal + sparse margin)"
            )
        elif getattr(self, "USE_BICM_V5_SUPPORT_GEOMETRY_OBJECTIVE", False):
            # [AUDIT][Risk:Major][Scope:v5_support_geometry]
            # V5.4 hybrid-oracle 분석에서, 예측된 support에 target의 contact/core 정보가
            # 빠져 있음을 확인했다. 이 분기는 target/input/decoder는 그대로 두고,
            # support geometry 목적함수만 변경한다. 따라서 새 아키텍처가 아니라
            # 한 변수만 바꿔보는 root-cause 실험이다.
            base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            aux_base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            loss = BICMV5SupportGeometryLoss(base, fullres_geometry_terms=True)
            bicm_v5_aux_loss = BICMV5SupportGeometryLoss(aux_base, fullres_geometry_terms=False)
            self.print_to_log_file(
                "[PengwinTrainer] Loss = DC+CE + V5 binary support Dice/Tversky "
                "+ target-core/contact support coverage + top-k support FP penalty"
            )
        elif getattr(self, "USE_BICM_V5_SPARSE_OBJECTIVE", False):
            # [AUDIT][Risk:Major][Scope:loss_contract]
            # V5는 class 3을 core로, class 4를 contact로 매핑한다. 예전 V3의 class-2 ridge loss를
            # 그대로 가져다 쓰면 의미가 어긋난 class를 학습하게 된다.
            # 그래서 이 분기는 `bicm_v5_sparse` loss profile일 때만 명시적으로 활성화된다.
            base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            aux_base = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            loss = BICMV5SparseLoss(base, fullres_sparse_terms=True)
            bicm_v5_aux_loss = BICMV5SparseLoss(aux_base, fullres_sparse_terms=False)
            self.print_to_log_file(
                "[PengwinTrainer] Loss = DC+CE + V5 support Dice + fullres "
                "core/contact Tversky/Focal/Margin sparse terms"
            )
        elif getattr(self, "USE_CONTACT_FUSE_OBJECTIVE", False):
            loss = ContactFuseLoss3D(
                n_classes=n_classes,
                ce_weight=cw_tensor,
                ignore_label=self.label_manager.ignore_label,
                contact_weight=getattr(self, "CONTACT_LOSS_WEIGHT", 2.5),
                contact_fp_weight=getattr(self, "CONTACT_FP_WEIGHT", 5.0),
                contact_pos_weight=getattr(self, "CONTACT_POS_WEIGHT", 0.75),
                core_weight=getattr(self, "CORE_LOSS_WEIGHT", 0.80),
                core_fp_weight=getattr(self, "CORE_FP_WEIGHT", 1.0),
                core_pos_weight=getattr(self, "CORE_POS_WEIGHT", 1.35),
                hard_negative_weight=getattr(self, "HARD_NEGATIVE_LOSS_WEIGHT", 1.25),
                hard_negative_contact_penalty=getattr(self, "HARD_NEGATIVE_CONTACT_PENALTY", 2.5),
                topk_false_contact_weight=getattr(self, "TOPK_FALSE_CONTACT_WEIGHT", 1.5),
                topk_fraction=getattr(self, "TOPK_FALSE_CONTACT_FRACTION", 0.02),
            )
            self.print_to_log_file(
                "[PengwinTrainer] Loss = CE + contact/core focal + class-4 "
                "hard-negative penalty + top-k false-contact mining (no Dice/Tversky)"
            )
        elif getattr(self, "USE_CONTACT_ENERGY_OBJECTIVE", False):
            loss = ContactEnergyLoss3D(
                n_classes=n_classes,
                ce_weight=cw_tensor,
                ignore_label=self.label_manager.ignore_label,
                contact_weight=getattr(self, "CONTACT_LOSS_WEIGHT", 2.0),
                contact_fp_weight=getattr(self, "CONTACT_FP_WEIGHT", 3.0),
                contact_pos_weight=getattr(self, "CONTACT_POS_WEIGHT", 1.0),
                core_weight=getattr(self, "CORE_LOSS_WEIGHT", 0.75),
                core_fp_weight=getattr(self, "CORE_FP_WEIGHT", 1.0),
                core_pos_weight=getattr(self, "CORE_POS_WEIGHT", 1.25),
            )
            self.print_to_log_file(
                "[PengwinTrainer] Loss = weighted CE + contact/core focal energy "
                "(no Dice/Tversky objective)"
            )
        else:
            loss = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs,
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

            if (
                self.USE_TVERSKY_LOSS
                and _BOUNDARY_LOSS_AVAILABLE
                and self.TV_WEIGHT > 0
            ):
                loss = DC_CE_TV_BD_loss(
                    dc_ce_loss=loss,
                    n_classes=n_classes,
                    weight_dc_ce=1.0,
                    weight_tversky=self.TV_WEIGHT,
                    tversky_alpha=self.TV_ALPHA,
                    tversky_beta=self.TV_BETA,
                    weight_bd=self.BD_WEIGHT if self.USE_BOUNDARY_LOSS else 0.0,
                    tversky_active_classes=self.TV_ACTIVE_CLASSES,
                    tversky_class_weights=self.TV_CLASS_WEIGHTS,
                )
                self.print_to_log_file(
                    f"[PengwinTrainer] Loss = DC+CE + {self.TV_WEIGHT}·Tversky"
                    f"(alpha={self.TV_ALPHA}, beta={self.TV_BETA})"
                    f" active={self.TV_ACTIVE_CLASSES or 'fg'}"
                    f" class_weights={self.TV_CLASS_WEIGHTS or 'uniform'}"
                    f" + {self.BD_WEIGHT if self.USE_BOUNDARY_LOSS else 0.0}·BoundaryDoU3D"
                    f"  (n_classes={n_classes})"
                )
            elif self.USE_BOUNDARY_LOSS and _BOUNDARY_LOSS_AVAILABLE and self.BD_WEIGHT > 0:
                loss = DC_CE_BD_loss(
                    dc_ce_loss=loss,
                    n_classes=n_classes,
                    weight_dc_ce=1.0,
                    weight_bd=self.BD_WEIGHT,
                )
                self.print_to_log_file(
                    f"[PengwinTrainer] Loss = DC+CE + {self.BD_WEIGHT}·BoundaryDoU3D"
                    f"  (n_classes={n_classes})"
                )
            elif (self.USE_BOUNDARY_LOSS or self.USE_TVERSKY_LOSS) and not _BOUNDARY_LOSS_AVAILABLE:
                self.print_to_log_file(
                    "[PengwinTrainer] WARN: custom loss requested but "
                    "code_task1/loss.py import failed — falling back to DC+CE."
                )

            if self._do_i_compile():
                loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            if bicm_v5_aux_loss is not None:
                loss = BICMV5SparseDeepSupervisionWrapper(loss, bicm_v5_aux_loss, weights)
            else:
                loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _configure_loss_ablation_from_env(self) -> None:
        """환경 변수로 선택된 loss ablation을 적용한다.

        지원하는 값:
            PENGWIN_LOSS_PROFILE=
                dc_ce|bd_dou_005|bd_dou_01|bd_dou_03|
                tversky_07|tversky_08|combo_tversky_bd005|
                abbc_contact_energy_v1|contact_fuse_v1|factorized_instance_v4
            PENGWIN_CE_CLASS_WEIGHTS=off|auto

        기본 학습은 그대로 비교 가능한 상태로 두면서, 코드 수정 없이도 재현 가능한
        loss 실험을 돌릴 수 있게 해준다.
        """
        profile = os.environ.get("PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE).strip().lower()
        self.USE_BOUNDARY_LOSS = False
        self.BD_WEIGHT = 0.0
        self.USE_TVERSKY_LOSS = False
        self.TV_WEIGHT = 0.0
        self.USE_CONTACT_FUSE_OBJECTIVE = False
        self.USE_CONTACT_ENERGY_OBJECTIVE = False
        self.USE_CONTACT_INSTANCE_OBJECTIVE = False
        self.USE_BICM_V5_SPARSE_OBJECTIVE = False
        self.USE_BICM_V5_SUPPORT_GEOMETRY_OBJECTIVE = False
        self.USE_BICM_V62_SEMANTIC_BOUNDARY_OBJECTIVE = False
        self.USE_BICM_V64_TOPOLOGY_PRECISION_OBJECTIVE = False
        self.USE_BICM_V65_BALANCED_CONTACT_CORE_OBJECTIVE = False
        self.USE_BICM_V67_SEMANTIC_EDGE_PAIR_OBJECTIVE = False
        self.TV_ALPHA = 0.3
        self.TV_BETA = 0.7
        self.TV_ACTIVE_CLASSES = None
        self.TV_CLASS_WEIGHTS = None
        if profile in ("", "dc_ce", "baseline"):
            pass
        elif profile == "bd_dou_005":
            self.USE_BOUNDARY_LOSS = True
            self.BD_WEIGHT = 0.05
        elif profile == "bd_dou_01":
            self.USE_BOUNDARY_LOSS = True
            self.BD_WEIGHT = 0.1
        elif profile == "bd_dou_03":
            self.USE_BOUNDARY_LOSS = True
            self.BD_WEIGHT = 0.3
        elif profile == "tversky_07":
            self.USE_TVERSKY_LOSS = True
            self.TV_WEIGHT = 0.5
            self.TV_ALPHA = 0.3
            self.TV_BETA = 0.7
        elif profile == "tversky_08":
            self.USE_TVERSKY_LOSS = True
            self.TV_WEIGHT = 0.5
            self.TV_ALPHA = 0.2
            self.TV_BETA = 0.8
        elif profile == "combo_tversky_bd005":
            self.USE_TVERSKY_LOSS = True
            self.TV_WEIGHT = 0.35
            self.TV_ALPHA = 0.3
            self.TV_BETA = 0.7
            self.USE_BOUNDARY_LOSS = True
            self.BD_WEIGHT = 0.05
        elif profile == "abbc_contact_energy_v1":
            # Contact-energy V2: 더 이상 class 3을 폭넓은 recall 대상으로 보지 않는다.
            # 후처리에서 energy ridge로 쓰는 좁은 contact/fracture surface이기 때문에,
            # 약간의 false negative보다 false positive가 훨씬 더 해롭다.
            # marker seed의 안정성을 위해 core는 그대로 활성 상태로 둔다.
            self.USE_TVERSKY_LOSS = True
            self.TV_WEIGHT = 0.55
            self.TV_ALPHA = 0.70
            self.TV_BETA = 0.30
            self.TV_ACTIVE_CLASSES = (2, 3)
            self.TV_CLASS_WEIGHTS = (0.0, 0.0, 1.75, 2.25)
        elif profile == "contact_fuse_v1":
            self.USE_CONTACT_FUSE_OBJECTIVE = True
            self.USE_CONTACT_ENERGY_OBJECTIVE = False
        elif profile == "bicm_v5_sparse":
            self.USE_BICM_V5_SPARSE_OBJECTIVE = True
        elif profile == "bicm_v5_support_geometry":
            self.USE_BICM_V5_SUPPORT_GEOMETRY_OBJECTIVE = True
        elif profile == "bicm_v62_semantic_boundary":
            self.USE_BICM_V62_SEMANTIC_BOUNDARY_OBJECTIVE = True
        elif profile == "bicm_v64_topology_precision":
            self.USE_BICM_V64_TOPOLOGY_PRECISION_OBJECTIVE = True
        elif profile == "bicm_v65_balanced_contact_core":
            self.USE_BICM_V65_BALANCED_CONTACT_CORE_OBJECTIVE = True
        elif profile == "bicm_v67_semantic_edge_pair":
            self.USE_BICM_V67_SEMANTIC_EDGE_PAIR_OBJECTIVE = True
        elif profile in {
            "bicm_v6_factorized",
            "bicm_v250_oracle_aligned_direct",
            "bicm_v6_precision",
            "bicm_v6_support_core",
            "bicm_v6_support_precision",
            "bicm_v6_support_surface",
            "bicm_v7_contact_balanced",
            "bicm_v7_contact_ranked",
            "bicm_v7_contact_presence",
            "bicm_v7_contact_ratio",
            "bicm_v7_contact_dense",
            "bicm_v8_contact_contour",
            "bicm_v9_contact_energy_pair",
            "bicm_v9_contact_energy_precision",
            "bicm_v10_adaptive_topology",
            "bicm_v12_contact_persistent",
            "bicm_v13_contact_precision",
            "bicm_v14_contact_curriculum",
            "bicm_v15_contact_presence_gate",
            "bicm_v16_roi_presence_classifier",
            "bicm_v19_roi_calibrated_presence",
            "bicm_v22_core_marker_separation",
            "bicm_v23_staged_core_marker",
            "bicm_v24_contact_preserved_core_marker",
            "bicm_v25_coupled_contact_core_marker",
            "bicm_v26_contact_tolerant_precision",
            "bicm_v28_isolated_contact_precision",
            "bicm_v29_gentle_isolated_contact_precision",
            "bicm_v30_soft_contact_ridge",
            "bicm_v31_local_contrastive_contact",
            "bicm_v32_memory_efficient_local_contrast",
            "bicm_v34_compact_core_marker",
            "bicm_v35_contact_precision_compact_core",
            "bicm_v36_staged_contact_precision_compact_core",
            "bicm_v37_asymmetric_contact_compact_core",
            "bicm_v38_edge_affinity",
            "bicm_v39_edge_primary",
            "bicm_v40_edge_core_separation",
            "bicm_v41_instance_core_edge",
            "bicm_v42_instance_core_edge_primary",
            "bicm_v43_edge_contact_viability",
            "bicm_v44_edge_local_rank",
            "bicm_v45_edge_candidate_save",
            "bicm_v46_staged_edge_contact",
            "bicm_v47_separated_contact_head",
            "bicm_v48_support_aware_contact_branch",
            "bicm_v49_decoder_feature_contact_branch",
            "bicm_v50_decoder_feature_contact_phase",
            "bicm_v51_dense_edge_cost",
            "bicm_v52_dense_edge_core_topology",
            "bicm_v53_fragment_marker_core",
            "bicm_v54_sparse_head_balanced",
            "bicm_v55_calibrated_sparse_head",
            "bicm_v56_phased_marker_contact",
            "bicm_v57_decoder_feature_marker_phase",
            "bicm_v58_heatmap_marker_contact",
            "bicm_v59_core_preserving_contact",
            "bicm_v60_strict_core_topology",
            "bicm_v61_peak_seed",
            "bicm_v68_semantic_topology",
            "bicm_v69_topology_calibrated",
            "bicm_v70_topology_consistency",
            "bicm_v71_edge_cut_primary",
            "bicm_v72_logit_calibrated",
            "bicm_v73_instance_topology",
            "bicm_v74_adaptive_instance_topology",
            "bicm_v75_edge_precision_seed_topology",
            "bicm_v76_gentle_edge_precision",
            "bicm_v77_edge_recall_precision_curriculum",
            "bicm_v78_core_anchored_edge_curriculum",
            "bicm_v79_duplicate_seed_edge",
            "bicm_v80_topology_state_adaptive",
            "bicm_v81_core_stable_edge_precision",
            "bicm_v82_seed_recall_edge_separation",
            "bicm_v83_seed_safe_edge_calibration",
            "bicm_v84_dual_head_contact_calibration",
            "bicm_v85_semantic_gate_contact",
            "bicm_v86_eval_band_edge_primary",
            "bicm_v87_eval_band_semantic_gate",
            "bicm_v88_core_preserving_band_precision",
            "bicm_v89_band_false_only_precision",
            "bicm_v90_semantic_band_product",
            "bicm_v91_topology_aware_edge_balance",
            "bicm_v93_eval_aligned_support_contact",
            "bicm_v94_final_row_support_contact",
            "bicm_v95_decoder_feature_contact",
            "bicm_v96_dense_band_gate",
            "bicm_v97_positive_dense_band_gate",
            "bicm_v98_teacher_distilled_dense_gate",
            "bicm_v99_adaptive_dual_margin_dense_gate",
            "bicm_v100_negative_balanced_dense_gate",
            "bicm_v101_dual_field_product_gate",
            "bicm_v102_distance_rank_dense_gate",
            "bicm_v103_semantic_distance_contact",
            "bicm_v104_teacher_semantic_contact",
            "bicm_v105_offset_assignment",
            "bicm_v106_offset_attractor",
            "bicm_v107_radial_support_offset",
            "bicm_v108_watershed_barrier",
            "bicm_v109_precision_locked_recall",
            "bicm_v110_semantic_geometry_bridge",
            "bicm_v111_joint_support_product",
            "bicm_v112_edge_graph_assignment",
            "bicm_v113_adaptive_boundary_product",
            "bicm_v114_encoder_adapter_boundary_product",
            "bicm_v115_semantic_oracle_adapter",
            "bicm_v116_graph_cost_separator",
            "bicm_v117_support_gate_semantic_contact",
            "bicm_v118_warm_support_gate_semantic_contact",
            "bicm_v119_all_network_adaptive_boundary",
            "bicm_v120_same_fragment_affinity",
            "bicm_v121_warm_same_fragment_affinity",
            "bicm_v122_warm_edge_product_precision",
            "bicm_v123_high_recall_gate_cleanup",
            "bicm_v124_support_conditioned_edge_precision",
            "bicm_v125_saturated_edge_dense_gate",
            "bicm_v126_local_adjacency_product",
            "bicm_v128_support_bridge_suppression",
            "bicm_v129_affinity_sharpening",
            "bicm_v130_support_topology_repair",
            "bicm_v131_all_network_support_topology_repair",
            "bicm_v132_support_veto_gate",
            "bicm_v133_support_veto_semantic",
            "bicm_v134_support_veto_gate_highlr",
            "bicm_v135_support_veto_gate_ultralr",
            "bicm_v136_support_topology_highlr",
            "bicm_v137_support_topology_ultralr",
            "bicm_v138_support_topology_midlr",
            "bicm_v139_support_topology_uppermidlr",
            "bicm_v140_support_topology_lowbracket",
            "bicm_v141_support_topology_highbracket",
            "bicm_v142_support_topology_coreseed_lowlr",
            "bicm_v143_support_topology_coreseed_midlr",
            "bicm_v144_strong_coreseed_lowlr",
            "bicm_v145_strong_coreseed_midlr",
            "bicm_v146_fragment_seed_presence_lowlr",
            "bicm_v147_fragment_seed_presence_midlr",
            "bicm_v148_core_heatmap_seed_lowlr",
            "bicm_v149_core_heatmap_seed_midlr",
            "bicm_v150_core_only_heatmap_seed_midlr",
            "bicm_v151_core_only_heatmap_seed_highlr",
            "bicm_v152_core_only_center_seed_midlr",
            "bicm_v153_core_heatmap_center_seed_midlr",
            "bicm_v154_core_only_center_seed_sampler",
            "bicm_v155_core_heatmap_center_seed_sampler",
            "bicm_v158_core_only_center_seed_affinity_sampler",
            "bicm_v159_core_heatmap_center_seed_affinity_sampler",
        }:
            # [QC][Invariant:custom_trainer_loss_dispatch]
            # V6 이후 BICM trainer들은 `_build_loss`를 override해서 custom sigmoid 목적함수를 쓴다.
            # base trainer는 단지 이 프로파일 이름들을 알아만 보면 된다 — 그래야 audit log에
            # 서브클래스가 진짜 loss를 만들기 전에 "DC+CE로 fallback"이라는 잘못된 메시지가 안 찍힌다.
            pass
        elif profile in {
            "contact_instance_v1",
            "contact_instance_exclusive_v1",
            "contact_instance_sigmoid_v1",
            "factorized_instance_v4",
            "factorized_instance_v4_seedfit",
            "factorized_instance_v4_staged",
            "factorized_instance_v4_v42",
            "factorized_instance_v4_heatseed",
            "factorized_instance_v4_v43",
        }:
            self.USE_CONTACT_INSTANCE_OBJECTIVE = True
            self.USE_CONTACT_FUSE_OBJECTIVE = False
            self.USE_CONTACT_ENERGY_OBJECTIVE = False
        else:
            self.print_to_log_file(
                f"[PengwinTrainer] unknown PENGWIN_LOSS_PROFILE={profile!r}; using DC+CE"
            )

        cw = os.environ.get("PENGWIN_CE_CLASS_WEIGHTS", self.DEFAULT_CE_CLASS_WEIGHTS).strip().lower()
        if cw == "auto":
            self.USE_CLASS_WEIGHTS = True
            self.CE_CLASS_WEIGHTS = "auto"
        elif cw in ("", "off", "none", "false", "0"):
            self.USE_CLASS_WEIGHTS = False
            self.CE_CLASS_WEIGHTS = None
        self.print_to_log_file(
            f"[PengwinTrainer] loss_profile={profile or 'dc_ce'} "
            f"boundary={self.BD_WEIGHT} "
            f"tversky={self.TV_WEIGHT}@({self.TV_ALPHA},{self.TV_BETA}) "
            f"ce_weights={self.CE_CLASS_WEIGHTS or 'off'} "
            f"oversample_profile={self._pengwin_oversample_profile} "
            f"oversample_fg={self.oversample_foreground_percent:.2f}"
        )

    def _resolve_class_weights(self, n_classes: int) -> np.ndarray | None:
        """CE_CLASS_WEIGHTS 속성을 (n_classes,) shape의 numpy 배열로 변환한다.

        받을 수 있는 형식:
            - None: None을 반환한다 (호출자가 weight 적용을 건너뛴다)
            - dict {class_idx: weight}: dense array로 변환하며, 누락된 키는 1.0으로 채운다
            - "auto": 생성된 nnU-Net raw label을 스캔해서 median-frequency 기반의
              clip된 weight를 계산한다.
            - np.ndarray / list: 복사 후 길이를 검증한다

        반환:
            (n_classes,) shape의 np.ndarray 또는 None.
        """
        cw = self.CE_CLASS_WEIGHTS
        if cw is None:
            return None
        if isinstance(cw, str):
            if cw == "auto":
                return self._compute_auto_class_weights(n_classes)
            return None
        if isinstance(cw, dict):
            arr = np.ones(n_classes, dtype=np.float32)
            for k, v in cw.items():
                if 0 <= int(k) < n_classes:
                    arr[int(k)] = float(v)
            return arr
        # list/array의 경우 — 길이를 검증하고 복사한다
        arr = np.asarray(cw, dtype=np.float32)
        if arr.shape != (n_classes,):
            self.print_to_log_file(
                f"[PengwinTrainer v7] CE_CLASS_WEIGHTS length {arr.shape} != "
                f"n_classes {n_classes} — ignoring."
            )
            return None
        return arr

    def _compute_auto_class_weights(self, n_classes: int) -> np.ndarray | None:
        """생성된 raw label로부터 class weight를 계산하고 audit JSON을 함께 기록한다."""
        try:
            import SimpleITK as sitk
        except ImportError:
            self.print_to_log_file("[PengwinTrainer] SimpleITK missing; CE auto weights disabled")
            return None
        dataset_name = self.plans_manager.dataset_name
        raw_root = Path(os.environ.get("nnUNet_raw", str(_ROOT / "nnunet/raw")))
        label_dir = raw_root / dataset_name / "labelsTr"
        label_paths = sorted(label_dir.glob("*.mha"))
        if not label_paths:
            self.print_to_log_file(f"[PengwinTrainer] no labelsTr found at {label_dir}; CE weights disabled")
            return None
        counts = np.zeros(n_classes, dtype=np.float64)
        for p in label_paths:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            bc = np.bincount(arr.ravel(), minlength=n_classes).astype(np.float64)
            counts += bc[:n_classes]
        mn, mx = self.CLASS_WEIGHT_CLIP
        counts_smooth = counts + 1.0
        freq = counts_smooth / counts_smooth.sum()
        present = counts > 0
        median = np.median(freq[present]) if present.any() else np.median(freq)
        weights = np.clip(median / freq, mn, mx).astype(np.float32)
        audit_dir = RESULT_WEIGHT
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"class_weights_{dataset_name}.json"
        audit_path.write_text(json.dumps({
            "dataset": dataset_name,
            "n_classes": n_classes,
            "n_label_files": len(label_paths),
            "method": "median_frequency",
            "clip": [mn, mx],
            "counts": [int(x) for x in counts.tolist()],
            "weights": [float(x) for x in weights.tolist()],
        }, indent=2))
        self.print_to_log_file(f"[PengwinTrainer] CE auto weights audit: {audit_path}")
        return weights

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """Pelvic 데이터셋에서 LH↔RH 라벨 오염을 막기 위해 axis-2 mirror(L/R)를 비활성화한다.

        transpose_forward [1,0,2]를 거치면 patch 공간의 axis 2는 원본의 x축, 즉 L/R 방향이다.
        이걸 mirror하면 LH 모양 voxel이 오른쪽에 생기는데 라벨은 여전히 LH라서 노이즈가 된다.
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        ds_name = self.plans_manager.dataset_name
        if ds_name in self.DISABLE_X_MIRROR_DATASETS:
            # mirror_axes에서 axis 2를 제거한다
            new_axes = tuple(a for a in mirror_axes if a != 2)
            self.print_to_log_file(
                f"[PengwinTrainer] {ds_name} → disable axis-2 mirror (L/R asymmetry). "
                f"mirror_axes: {mirror_axes} → {new_axes}"
            )
            self.inference_allowed_mirroring_axes = new_axes
            mirror_axes = new_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def _bicm_contact_roi_sampling_probabilities(self, dataset: nnUNetDataset) -> np.ndarray | None:
        """V11 contact-positive exposure 실험을 위한 train-case 확률을 반환한다.

        [AUDIT][Risk:High][Scope:single_variable_ablation]
        V10에서는 support와 core가 full-volume gate를 통과했음에도 contact head가
        0으로 붕괴(collapse)했다. 이 메서드는 오직 sample 노출 빈도만 바꿔서,
        contact-positive ROI 샘플의 선택 확률을 높이되 no-contact ROI도 분포에 남겨두어
        absent-contact negative까지 학습되도록 한다.

        [QC][Invariant:nnunet_sampling_probabilities]
        nnU-Net은 `list(dataset.keys())` 순서에 맞춰 정렬된 확률을 기대한다.
        만약 class-4 contact location을 가진 ROI가 하나도 없다면 `None`을 반환해
        기본 uniform sampling으로 폴백한다.
        """
        keys = list(dataset.keys())
        if not keys:
            return None
        weights: list[float] = []
        contact_keys: list[str] = []
        for key in keys:
            try:
                _, _, props = dataset.load_case(key)
            except Exception as exc:  # pragma: no cover - 전처리 데이터가 손상된 경우를 위한 runtime guard
                raise RuntimeError(f"failed to load Dataset537 BICM properties for {key!r}") from exc
            has_contact = PengwinBICMV5DataLoader3D._class_key(
                props.get("class_locations", {}),
                4,
            ) is not None
            if has_contact:
                contact_keys.append(str(key))
            weights.append(6.0 if has_contact else 1.0)
        if not contact_keys:
            self.print_to_log_file(
                "[PengwinTrainer] BICM contact ROI sampling disabled: no class-4 contact ROI in train split"
            )
            return None
        probs = np.asarray(weights, dtype=np.float64)
        probs = probs / probs.sum()
        self.print_to_log_file(
            "[PengwinTrainer] BICM contact ROI sampling",
            {
                "contact_positive_keys": contact_keys,
                "contact_positive_key_count": len(contact_keys),
                "total_key_count": len(keys),
                "max_probability": float(probs.max()),
                "min_probability": float(probs.min()),
            },
        )
        return probs

    def get_dataloaders(self):
        """기본 dataloader를 반환하되, opt-in으로 V5 sparse crop steering을 적용한다.

        [QA][Experiment:V5_sparse_sampling]
        이 메서드는 의도적으로 `Dataset537_PelvicBICMFragmentV5` 데이터셋 +
        `PENGWIN_OVERSAMPLE_PROFILE=bicm_v5_sparse` 조합에서만 동작한다.
        그 외 해부학 데이터셋과 V5 plain-DC+CE baseline은 부모 nnU-Net의 dataloader를 그대로 쓴다.
        """
        if (
            self.plans_manager.dataset_name != "Dataset537_PelvicBICMFragmentV5"
            or self._pengwin_oversample_profile not in {
                "bicm_v5_sparse",
                "bicm_v6_support_mixed",
                "bicm_v7_contact_mixed",
                "bicm_v7_contact15_mixed",
                "bicm_v11_contact_roi",
                "bicm_v18_roi_balanced",
            }
        ):
            return super().get_dataloaders()
        patch_size = self.configuration_manager.patch_size
        if len(patch_size) != 3:
            raise RuntimeError("BICM V5 sparse sampling supports only 3D fullres training.")
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        tr_keys, val_keys = self.do_split()
        dataset_tr = nnUNetDataset(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        dataset_val = nnUNetDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        if self._pengwin_oversample_profile == "bicm_v6_support_mixed":
            forced_distribution = PengwinBICMV5DataLoader3D.SUPPORT_MIXED_CLASS_DISTRIBUTION
        elif self._pengwin_oversample_profile == "bicm_v7_contact_mixed":
            forced_distribution = PengwinBICMV5DataLoader3D.CONTACT_MIXED_CLASS_DISTRIBUTION
        elif self._pengwin_oversample_profile == "bicm_v7_contact15_mixed":
            forced_distribution = PengwinBICMV5DataLoader3D.CONTACT15_MIXED_CLASS_DISTRIBUTION
        elif self._pengwin_oversample_profile in {"bicm_v11_contact_roi", "bicm_v18_roi_balanced"}:
            forced_distribution = PengwinBICMV5DataLoader3D.CONTACT_ROI_CLASS_DISTRIBUTION
        else:
            forced_distribution = PengwinBICMV5DataLoader3D.SPARSE_CLASS_DISTRIBUTION
        self.print_to_log_file(
            "[PengwinTrainer] BICM train forced class distribution",
            [(int(k), float(v)) for k, v in forced_distribution],
        )
        sampling_probabilities = (
            self._bicm_contact_roi_sampling_probabilities(dataset_tr)
            if self._pengwin_oversample_profile == "bicm_v11_contact_roi"
            else None
        )
        if self._pengwin_oversample_profile == "bicm_v18_roi_balanced":
            # [AUDIT][Risk:High][Scope:single_variable_ablation]
            # V17은 V11 sampler를 그대로 썼는데, 그 sampler는 contact-positive ROI에
            # 6배의 case weight를 부여한다. case003에서는 LeftHip contact ROI 하나가
            # Sacrum/RightHip의 absent-contact 학습을 지배하게 된다.
            # V18은 ROI case sampling만 uniform으로 되돌리고, V17 분기·V16 loss·
            # 고정된 target/decoder/threshold/forced class distribution은 그대로 유지한다.
            #
            # [QA][Gate:003_roi_absence]
            # 통과 근거는 full-volume eval에서 와야 한다. 즉 부재해야 할 Sacrum/RightHip
            # ROI 점수는 0.5 미만으로 떨어지면서 LeftHip contact recall은 무너지지 않아야 한다.
            self.print_to_log_file(
                "[PengwinTrainer] BICM V18 ROI sampling = uniform over case/anatomy keys "
                "(contact-positive ROI weighting disabled)"
            )
        dl_tr = PengwinBICMV5DataLoader3D(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=sampling_probabilities,
            pad_sides=None,
            transforms=tr_transforms,
            forced_class_distribution=forced_distribution,
        )
        dl_val = nnUNetDataLoader3D(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            # [DATA][Validation]
            # Validation은 편향 없이 유지한다. Patch validation은 단순 health check일 뿐,
            # 진짜 판단 기준은 full-volume bicm-v5-eval JSON이다.
            oversample_foreground_percent=0.0,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
        )
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def configure_optimizers(self):
        """SGD + Cosine annealing + warmup (PENGWIN 2024 표준)."""
        optimizer = SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                        momentum=0.99, nesterov=True)
        # Warmup 후 cosine 스케줄 적용
        warmup = self.WARMUP_EPOCHS

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            # 남은 epoch 동안 peak에서 0까지 cosine 감소
            import math
            progress = (epoch - warmup) / max(1, self.num_epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        return optimizer, lr_scheduler

    def run_training(self):
        """학습 중간에 ES(Early Stop)가 num_epochs를 줄일 수 있도록 while-loop로 오버라이드한다.

        부모 클래스의 `for epoch in range(start, num_epochs)`는 루프 생성 시점에 num_epochs를
        한 번만 평가하기 때문에, 이후 self.num_epochs를 바꿔도 반영되지 않는다.
        그래서 매 반복마다 self.num_epochs를 다시 확인하는 while-loop를 사용한다.
        """
        self.on_train_start()
        while self.current_epoch < self.num_epochs:
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        # EMA pseudo Dice 기준 Early stop 판단
        completed = self.current_epoch + 1
        if completed < self.ES_MIN_EPOCHS:
            return
        ema = float(self.logger.my_fantastic_logging["ema_fg_dice"][-1])
        if ema > self._es_best_dice + self.ES_MIN_DELTA:
            self._es_best_dice = ema
            self._es_no_improve = 0
        else:
            self._es_no_improve += 1
            if self._es_no_improve >= self.ES_PATIENCE and not self._es_triggered:
                self._es_triggered = True
                self.print_to_log_file(
                    f"Early stopping: no EMA-Dice improvement for "
                    f"{self.ES_PATIENCE} epochs. Best EMA Dice = {self._es_best_dice:.4f} "
                    f"at epoch {completed - self.ES_PATIENCE}. "
                    f"Stopping at epoch {completed}."
                )
                self.num_epochs = completed  # 메인 루프에 종료 신호를 보낸다


























































































































































































































































































































class PengwinBoundaryFragmentDataLoader3D(nnUNetDataLoader3D):
    """V3 dataloader that oversamples separator/context/core classes.

    The target is still a normal 5-class segmentation, so we keep nnU-Net's
    native crop/augmentation path and only change which class is used when a
    forced foreground crop is requested. This is deliberately less invasive
    than the retired multi-head topology dataloader.
    """

    FORCED_CLASS_DISTRIBUTION = (
        (2, 0.50),  # fracture/contact barrier
        (1, 0.20),  # external context hard negative
        (3, 0.20),  # fragment shell
        (4, 0.10),  # deep/tiny core
    )

    RIDGE80_CLASS_DISTRIBUTION = (
        (2, 0.80),  # force the thin ridge until crop density is disproved
        (1, 0.10),  # external context hard negative
        (3, 0.08),  # support near-negative shell
        (4, 0.02),  # marker sanity
    )

    def __init__(self, *args, forced_class_distribution=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_class_distribution = tuple(forced_class_distribution or self.FORCED_CLASS_DISTRIBUTION)

    @staticmethod
    def _has_locations(class_locations: dict, label: int) -> bool:
        if not class_locations:
            return False
        for key, value in class_locations.items():
            try:
                if int(key) == int(label) and len(value) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _class_key(class_locations: dict, label: int):
        for key, value in class_locations.items():
            try:
                if int(key) == int(label) and len(value) > 0:
                    return key
            except (TypeError, ValueError):
                continue
        return None

    def _pick_forced_class(self, class_locations: dict):
        available = [
            (label, prob)
            for label, prob in self.forced_class_distribution
            if self._has_locations(class_locations, label)
        ]
        if not available:
            return None
        labels = [x[0] for x in available]
        probs = np.asarray([x[1] for x in available], dtype=np.float64)
        probs = probs / probs.sum()
        chosen = int(np.random.choice(labels, p=probs))
        return self._class_key(class_locations, chosen)

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool,
                 class_locations: dict | None,
                 overwrite_class=None, verbose: bool = False):
        if force_fg and overwrite_class is None and class_locations:
            overwrite_class = self._pick_forced_class(class_locations)
        return super().get_bbox(
            data_shape,
            force_fg,
            class_locations,
            overwrite_class=overwrite_class,
            verbose=verbose,
        )


class PengwinBoundaryFragmentTargetSidecarDataLoader3D(PengwinLegacyTopologyDataLoader3D):
    """Crop Dataset537 ROI data with precomputed BFV3 semantic targets."""

    def __init__(self, *args, target_sidecar_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_sidecar_dir = Path(target_sidecar_dir)

    def _load_target(self, key: str) -> np.ndarray:
        path = self.target_sidecar_dir / f"{key}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"BoundaryFragment target sidecar missing: {path}. "
                "Build boundary_fragment_v3_targets from dense instance sidecars before training."
            )
        with np.load(path) as payload:
            if "target" not in payload:
                raise KeyError(f"{path} does not contain `target`")
            return np.asarray(payload["target"], dtype=np.uint8)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        target_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)

        for j, key in enumerate(selected_keys):
            data, _seg, _properties = self._data.load_case(key)
            shape = data.shape[1:]
            force_fg = self.get_do_oversample(j)
            sidecar = self._load_sidecar(key)
            center = self._choose_priority_center(sidecar, key=key) if force_fg else None
            if center is None:
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=None)
            else:
                bbox_lbs, bbox_ubs = self._bbox_from_center(shape, center)
            target = self._load_target(key)
            # [DATA][Risk:High][Scope:bfv3_target_alignment]
            # V164/V165 intentionally bypass Dataset537 V5 semantic labels.
            # The BFV3 target must share the exact preprocessed ROI grid with
            # the image crop; otherwise barrier/core metrics would be meaningless.
            if tuple(target.shape[-3:]) != tuple(shape):
                raise ValueError(
                    f"{key}: BFV3 target shape {target.shape[-3:]} does not match image shape {tuple(shape)}"
                )
            data_crop, _crop_lbs = self._crop_pad(data, bbox_lbs, bbox_ubs, pad_value=0)
            target_crop, _ = self._crop_pad(target, bbox_lbs, bbox_ubs, pad_value=0)
            data_all[j] = data_crop
            target_all[j, 0] = target_crop.astype(np.int16, copy=False)

        return {"data": data_all, "target": target_all}


class PengwinBoundaryFragmentSeedTargetSidecarDataLoader3D(PengwinBoundaryFragmentTargetSidecarDataLoader3D):
    """BFV3 sidecar loader with an additional fragment-scale seed target."""

    def __init__(
        self,
        *args,
        seed_radius_vox: float = 2.0,
        decoder_contact_center_probability: float = 0.0,
        decoder_contact_negative_center_probability: float = 0.0,
        decoder_contact_center_source: str = "v5",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seed_radius_vox = float(seed_radius_vox)
        if self.seed_radius_vox <= 0:
            raise ValueError(f"seed_radius_vox must be > 0, got {self.seed_radius_vox}")
        self.decoder_contact_center_probability = float(decoder_contact_center_probability)
        if not 0.0 <= self.decoder_contact_center_probability <= 1.0:
            raise ValueError(
                "decoder_contact_center_probability must be in [0, 1], "
                f"got {self.decoder_contact_center_probability}"
            )
        self.decoder_contact_negative_center_probability = float(decoder_contact_negative_center_probability)
        if not 0.0 <= self.decoder_contact_negative_center_probability <= 1.0:
            raise ValueError(
                "decoder_contact_negative_center_probability must be in [0, 1], "
                f"got {self.decoder_contact_negative_center_probability}"
            )
        if self.decoder_contact_center_probability + self.decoder_contact_negative_center_probability > 1.0:
            raise ValueError(
                "decoder contact center probabilities must sum to <= 1, got "
                f"positive={self.decoder_contact_center_probability} "
                f"negative={self.decoder_contact_negative_center_probability}"
            )
        self.decoder_contact_center_source = str(decoder_contact_center_source).strip().lower()
        if self.decoder_contact_center_source not in {"v5", "bfv3", "hybrid"}:
            raise ValueError(
                "decoder_contact_center_source must be one of {'v5', 'bfv3', 'hybrid'}, "
                f"got {decoder_contact_center_source!r}"
            )

    @staticmethod
    def _fragment_seed_centers(inst: np.ndarray,
                               centers_zyx: np.ndarray,
                               bbox_lbs: list[int]) -> np.ndarray:
        """Create one center voxel per fragment present in the crop.

        [DATA][Risk:High][Scope:v253_seed_contract]
        V253 seed-healing assigns seedless affinity components to the nearest
        predicted seed. The train batch must therefore expose both a compact
        seed body and a sparse seed center target derived from the same instance
        sidecar and crop coordinates.
        """
        centers = np.asarray(centers_zyx, dtype=np.float32)
        bbox = np.asarray(bbox_lbs, dtype=np.float32)
        seed_center = np.zeros(inst.shape, dtype=np.float32)
        for fid, _sl in _present_fragment_slices(inst):
            coords = np.argwhere(inst[_sl] == fid)
            if coords.size == 0:
                continue
            coords = coords + np.asarray([s.start for s in _sl], dtype=coords.dtype)
            center_global = centers[fid] if fid < centers.shape[0] else np.asarray(coords.mean(axis=0), dtype=np.float32)
            center_local = center_global - bbox if np.isfinite(center_global).all() else coords.mean(axis=0)
            nearest = coords[np.argmin(((coords.astype(np.float32) - center_local.reshape(1, 3)) ** 2).sum(axis=1))]
            seed_center[tuple(int(v) for v in nearest)] = 1.0
        return seed_center

    @staticmethod
    def _fragment_spatial_embedding_targets(inst: np.ndarray,
                                            centers_zyx: np.ndarray,
                                            bbox_lbs: list[int],
                                            full_shape: tuple[int, int, int]) -> np.ndarray:
        """Return full-ROI normalized fragment sink coordinates for one crop."""
        # [QC][Invariant:coordinate_space]
        # V270 learned target must be invariant to random crop position. Therefore
        # sink coordinates are normalized by the full preprocessed ROI shape, not
        # by the current patch shape. The dataloader owns bbox_lbs/centers_zyx, so
        # it is the only safe place to build this target.
        centers = np.asarray(centers_zyx, dtype=np.float32)
        bbox = np.asarray(bbox_lbs, dtype=np.float32)
        denom = np.maximum(np.asarray(full_shape, dtype=np.float32) - 1.0, 1.0)
        embedding = np.zeros((3, *inst.shape), dtype=np.float32)
        for fid, _sl in _present_fragment_slices(inst):
            sub = inst[_sl] == fid
            coords = np.argwhere(sub)
            if coords.size == 0:
                continue
            coords = coords + np.asarray([s.start for s in _sl], dtype=coords.dtype)
            center_global = (
                centers[fid]
                if fid < centers.shape[0]
                else coords.astype(np.float32).mean(axis=0) + bbox
            )
            if not np.isfinite(center_global).all():
                center_global = coords.astype(np.float32).mean(axis=0) + bbox
            sink_norm = np.clip(center_global.astype(np.float32) / denom, 0.0, 1.0)
            emb_sub = embedding[(slice(None), *_sl)]  # (3, bz, by, bx) view into embedding
            for axis in range(3):
                emb_sub[axis][sub] = float(sink_norm[axis])
        return embedding

    def _choose_decoder_contact_center(
        self,
        decoder_contact_full: np.ndarray,
        bfv3_full: np.ndarray | None = None,
    ) -> np.ndarray | None:
        draw = np.random.random()
        if draw < self.decoder_contact_center_probability:
            if self.decoder_contact_center_source == "hybrid":
                if bfv3_full is None:
                    raise ValueError("hybrid decoder contact center source requires BFV3 target")
                label_mask = (decoder_contact_full == 4) | (bfv3_full == 2)
            else:
                positive_label = 2 if self.decoder_contact_center_source == "bfv3" else 4
                label_mask = decoder_contact_full == positive_label
        elif draw < self.decoder_contact_center_probability + self.decoder_contact_negative_center_probability:
            if self.decoder_contact_center_source == "hybrid":
                if bfv3_full is None:
                    raise ValueError("hybrid decoder contact center source requires BFV3 target")
                positive_mask = (decoder_contact_full == 4) | (bfv3_full == 2)
                label_mask = (
                    (decoder_contact_full == 1)
                    | (decoder_contact_full == 2)
                    | (decoder_contact_full == 3)
                ) & ~positive_mask
            elif self.decoder_contact_center_source == "bfv3":
                label_mask = (
                    (decoder_contact_full == 1)
                    | (decoder_contact_full == 3)
                    | (decoder_contact_full == 4)
                )
            else:
                label_mask = (decoder_contact_full == 2) | (decoder_contact_full == 3)
        else:
            return None
        contact_flat = np.flatnonzero(label_mask)
        if contact_flat.size == 0:
            return None
        chosen = int(contact_flat[np.random.choice(contact_flat.size)])
        # [DATA][Risk:High][Scope:v247_contact_positive_exposure]
        # V247은 decoder contact positive를 BFV3 label2와 full-eval V5 label4의
        # union으로 둔다. sampler가 한쪽 label 공간만 보면 mass budget이 켜진 뒤
        # positive anchor가 부족해 contact recall이 닫힌다. source는 trainer별로
        # 고정해 old V5-centered trainer 재현성은 유지하고, V247만 새 decoder-facing
        # target과 같은 hybrid label 공간에서 crop exposure를 만든다.
        return np.asarray(np.unravel_index(chosen, decoder_contact_full.shape), dtype=np.int32)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        semantic_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
        decoder_contact_semantic_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
        seed_center_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        seed_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        instance_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
        spatial_embedding_all = np.zeros((self.batch_size, 3, *self.patch_size), dtype=np.float32)

        for j, key in enumerate(selected_keys):
            data, seg, _properties = self._data.load_case(key)
            shape = data.shape[1:]
            force_fg = self.get_do_oversample(j)
            sidecar = self._load_sidecar(key)

            target = self._load_target(key)
            inst_full = np.asarray(sidecar["instance"], dtype=np.uint16)
            # [DATA][Risk:High][Scope:bfv3_seed_alignment]
            # V201/V202 compare semantic BFV3 labels and fragment-scale seeds
            # in the same crop. Both sidecars must already be on the exact
            # preprocessed ROI grid; otherwise topology validation is invalid.
            if tuple(target.shape[-3:]) != tuple(shape):
                raise ValueError(
                    f"{key}: BFV3 target shape {target.shape[-3:]} does not match image shape {tuple(shape)}"
                )
            if target.size:
                target_min = int(np.min(target))
                target_max = int(np.max(target))
                if target_min < 0 or target_max > 4:
                    raise ValueError(
                        f"{key}: BFV3 semantic labels must be in [0, 4], "
                        f"got min={target_min} max={target_max}"
                    )
            if tuple(inst_full.shape) != tuple(shape):
                raise ValueError(
                    f"{key}: instance sidecar shape {tuple(inst_full.shape)} does not match image shape {tuple(shape)}"
                )
            if seg is None:
                raise ValueError(
                    f"{key}: decoder-contact supervision requires Dataset537 V5 segmentation, got None"
                )
            seg_full = np.asarray(seg)
            if seg_full.ndim == 4 and int(seg_full.shape[0]) == 1:
                decoder_contact_full = seg_full[0]
            elif seg_full.ndim == 3:
                decoder_contact_full = seg_full
            else:
                raise ValueError(
                    f"{key}: expected V5 segmentation shape [1, Z, Y, X] or [Z, Y, X], "
                    f"got {tuple(seg_full.shape)}"
                )
            # [DATA][Risk:High][Scope:decoder_contact_target]
            # BFV3 semantic label2는 thin fracture/barrier ridge이고, hard-case full eval의
            # contact GT는 Dataset537 V5 label4(contact_surface)다. `decoder_contact_semantic`
            # key는 crop exposure와 forensic audit용으로 보존하되, V247 기본 loss는
            # BFV3 semantic label2 positive / label1·3·4 hard-negative contract를 사용한다.
            if tuple(decoder_contact_full.shape) != tuple(shape):
                raise ValueError(
                    f"{key}: V5 segmentation shape {tuple(decoder_contact_full.shape)} "
                    f"does not match image shape {tuple(shape)}"
                )
            if decoder_contact_full.size:
                label_min = int(np.min(decoder_contact_full))
                label_max = int(np.max(decoder_contact_full))
                if label_min < 0 or label_max > 4:
                    raise ValueError(
                        f"{key}: V5 decoder-contact semantic labels must be in [0, 4], "
                        f"got min={label_min} max={label_max}"
                    )
            center = None
            if force_fg:
                decoder_contact_center_full = (
                    target
                    if self.decoder_contact_center_source == "bfv3"
                    else decoder_contact_full
                )
                center = self._choose_decoder_contact_center(
                    decoder_contact_center_full,
                    bfv3_full=target,
                )
                if center is None:
                    center = self._choose_priority_center(sidecar, key=key)
            if center is None:
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=None)
            else:
                bbox_lbs, bbox_ubs = self._bbox_from_center(shape, center)

            data_crop, _crop_lbs = self._crop_pad(data, bbox_lbs, bbox_ubs, pad_value=0)
            target_crop, _ = self._crop_pad(target, bbox_lbs, bbox_ubs, pad_value=0)
            decoder_contact_crop, _ = self._crop_pad(
                decoder_contact_full,
                bbox_lbs,
                bbox_ubs,
                pad_value=0,
            )
            inst_crop, _ = self._crop_pad(inst_full, bbox_lbs, bbox_ubs, pad_value=0)
            inst = np.where((inst_crop >= 1) & (inst_crop <= MAX_INSTANCE_ID), inst_crop, 0).astype(np.uint16, copy=False)
            centers_zyx = np.asarray(sidecar["centers_zyx"], dtype=np.float32)
            seed_center = self._fragment_seed_centers(
                inst,
                centers_zyx,
                bbox_lbs,
            )
            seed = PengwinBICMV5EdgeAffinityDataLoader3D._fragment_center_core(
                inst,
                centers_zyx,
                bbox_lbs,
                self.seed_radius_vox,
            )
            spatial_embedding = self._fragment_spatial_embedding_targets(
                inst,
                centers_zyx,
                bbox_lbs,
                tuple(int(v) for v in shape),
            )

            data_all[j] = data_crop
            semantic_all[j, 0] = target_crop.astype(np.int16, copy=False)
            decoder_contact_semantic_all[j, 0] = decoder_contact_crop.astype(np.int16, copy=False)
            seed_center_all[j, 0] = seed_center.astype(np.float32, copy=False)
            seed_all[j, 0] = seed.astype(np.float32, copy=False)
            instance_all[j, 0] = inst.astype(np.int16, copy=False)
            spatial_embedding_all[j] = spatial_embedding.astype(np.float32, copy=False)

        return {
            "data": data_all,
            "target": {
                "semantic": semantic_all,
                "decoder_contact_semantic": decoder_contact_semantic_all,
                "seed_center": seed_center_all,
                "seed": seed_all,
                "instance": instance_all,
                "spatial_embedding": spatial_embedding_all,
            },
            "keys": selected_keys,
        }


class PengwinTrainerBoundaryFragmentV3(_GroupedSplitMixin, nnUNetTrainer):
    """Native nnU-Net BoundaryFragment V3 specialist.

    Contract:
        - raw labels are exactly 0..4, not instance IDs;
        - output is a 5-class softmax semantic boundary-band map;
        - checkpoint score is a custom boundary/topology proxy, not mean Dice;
        - final IoU-F remains an explicit full-volume eval step.
    """

    NUM_EPOCHS_DEFAULT = 80
    WARMUP_EPOCHS = 10
    ES_MIN_EPOCHS = 20
    ES_PATIENCE = 30
    ES_MIN_DELTA = 2e-3
    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3"
    DEFAULT_CE_CLASS_WEIGHTS = "auto"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3"
    DISABLE_X_MIRROR_DATASETS = {"Dataset537_PelvicBoundaryFragmentV3"}
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = False
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = False

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BOUNDARY_FRAGMENT_EPOCHS", self.NUM_EPOCHS_DEFAULT))
        if os.environ.get("PENGWIN_BOUNDARY_FRAGMENT_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BOUNDARY_FRAGMENT_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BOUNDARY_FRAGMENT_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BOUNDARY_FRAGMENT_VAL_ITERS"])
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self._pengwin_oversample_profile = os.environ.get(
            "PENGWIN_OVERSAMPLE_PROFILE", self.DEFAULT_OVERSAMPLE_PROFILE
        ).strip().lower()
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = (
            1.00 if self._pengwin_oversample_profile == "boundary_fragment_v3_ridge80" else 0.70
        )
        self._es_best_score = -1.0
        self._es_no_improve = 0
        self._es_triggered = False
        self._best_barrier_stable_score = -1.0

    def initialize(self) -> None:
        super().initialize()
        self.print_to_log_file(
            "[BoundaryFragmentV3] runtime contract "
            f"epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch} "
            f"case_subset={self._overfit_case_ids()} "
            f"loss_profile={self._pengwin_loss_profile} "
            f"oversample_profile={self._pengwin_oversample_profile} "
            f"oversample_fg={self.oversample_foreground_percent:.2f}"
        )

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        ds_name = self.plans_manager.dataset_name
        if ds_name in self.DISABLE_X_MIRROR_DATASETS:
            new_axes = tuple(a for a in mirror_axes if a != 2)
            self.print_to_log_file(
                f"[BoundaryFragmentV3] {ds_name}: disable axis-2 mirror "
                f"for left/right pelvic anatomy. {mirror_axes} -> {new_axes}"
            )
            self.inference_allowed_mirroring_axes = new_axes
            mirror_axes = new_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def configure_optimizers(self):
        optimizer = SGD(self.network.parameters(), self.initial_lr,
                        weight_decay=self.weight_decay, momentum=0.99, nesterov=True)

        def lr_lambda(epoch):
            if epoch < self.WARMUP_EPOCHS:
                return (epoch + 1) / max(1, self.WARMUP_EPOCHS)
            import math
            progress = (epoch - self.WARMUP_EPOCHS) / max(1, self.num_epochs - self.WARMUP_EPOCHS)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optimizer, LambdaLR(optimizer, lr_lambda)

    def _overfit_case_ids(self) -> list[str] | None:
        """Return an explicit fixed case subset for V3 root-cause overfits.

        BoundaryFragment V3 promotion is gated by full-volume behavior, so the
        1-case and 20-case experiments must not silently fall back to the
        regular fold split. The environment contract keeps this path out of the
        public CLI while making the exact case list auditable in train logs.
        """
        raw_path = os.environ.get("PENGWIN_BOUNDARY_FRAGMENT_CASES_JSON", "").strip()
        raw_cases = os.environ.get("PENGWIN_BOUNDARY_FRAGMENT_CASES", "").strip()
        payload = None
        if raw_path:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"PENGWIN_BOUNDARY_FRAGMENT_CASES_JSON missing: {path}")
            payload = json.loads(path.read_text())
        elif raw_cases:
            payload = [v.strip() for v in raw_cases.replace(",", " ").split() if v.strip()]
        if payload is None:
            return None
        if isinstance(payload, dict):
            cases = payload.get("cases") or payload.get("train_cases") or payload.get("case_ids")
        else:
            cases = payload
        if not cases:
            raise ValueError("PENGWIN_BOUNDARY_FRAGMENT overfit case list is empty")

        normalized = []
        for case in cases:
            text = str(case).strip()
            if not text:
                continue
            if text.startswith("PENGWIN_"):
                normalized.append(text)
            else:
                normalized.append(f"PENGWIN_{int(text):03d}")
        return sorted(set(normalized))

    def get_tr_and_val_datasets(self):
        overfit_cases = self._overfit_case_ids()
        if overfit_cases is None:
            return super().get_tr_and_val_datasets()

        available = set(nnUNetDataset(self.preprocessed_dataset_folder).dataset.keys())
        missing = [case for case in overfit_cases if case not in available]
        if missing:
            raise RuntimeError(f"BoundaryFragmentV3 overfit cases missing from preprocessed Dataset537: {missing}")
        self.print_to_log_file(
            "[BoundaryFragmentV3] overfit subset active: "
            f"n_cases={len(overfit_cases)} cases={overfit_cases}"
        )
        dataset_tr = nnUNetDataset(
            self.preprocessed_dataset_folder,
            overfit_cases,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        dataset_val = nnUNetDataset(
            self.preprocessed_dataset_folder,
            overfit_cases,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        return dataset_tr, dataset_val

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise RuntimeError("BoundaryFragmentV3 expects plain 5-class labels, not regions.")
        n_classes = self.label_manager.num_segmentation_heads
        if n_classes != 5:
            raise RuntimeError(f"BoundaryFragmentV3 expects 5 segmentation heads, got {n_classes}.")
        def _base_loss():
            return DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                {},
                weight_ce=1.0,
                weight_dice=1.0,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        profile = self._pengwin_loss_profile
        base = _base_loss()
        if profile == "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v291_head":
            loss_cls = {
                "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v291_head": BoundaryFragmentV3Core025StrongPeakNoContactABBCV291HeadLoss,
            }[profile]
            loss = loss_cls(base)
            if self._do_i_compile():
                base.dc = torch.compile(base.dc)
            if self.enable_deep_supervision:
                aux_base = _base_loss()
                if self._do_i_compile():
                    aux_base.dc = torch.compile(aux_base.dc)
                aux_loss = BoundaryFragmentV3Loss(aux_base)
                ds_scales = self._get_deep_supervision_scales()
                weights = np.array([1.0 / (2 ** i) for i in range(len(ds_scales))])
                weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0.0
                weights = weights / weights.sum()
                loss = BoundaryFragmentV3RidgeFitDeepSupervisionWrapper(loss, aux_loss, weights)
            self.print_to_log_file(
                f"[BoundaryFragmentV3] Loss = {profile} "
                "(explicit class4 core seed focal/Tversky/margin; "
                "core_ridge keeps class2 ridge terms with profile-specific precision/recall bias)"
            )
            return loss

        if profile != "boundary_fragment_v3":
            raise RuntimeError(f"BoundaryFragmentV3 unknown loss profile: {profile}")

        loss = BoundaryFragmentV3Loss(base)
        if self._do_i_compile():
            base.dc = torch.compile(base.dc)
        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1.0 / (2 ** i) for i in range(len(ds_scales))])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0.0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        self.print_to_log_file(
            "[BoundaryFragmentV3] Loss = DiceCE + soft adjacent CE + "
            "binary support Dice + class2 precision-Tversky + narrow-band surface proxy"
        )
        return loss

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        if len(patch_size) != 3:
            raise RuntimeError("BoundaryFragmentV3 supports only 3D fullres training.")
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        overfit_cases = self._overfit_case_ids()
        if overfit_cases is None:
            tr_keys, val_keys = self.do_split()
        else:
            available = set(nnUNetDataset(self.preprocessed_dataset_folder).dataset.keys())
            missing = [case for case in overfit_cases if case not in available]
            if missing:
                raise RuntimeError(f"BoundaryFragmentV3 overfit cases missing from preprocessed Dataset537: {missing}")
            tr_keys = overfit_cases
            val_keys = overfit_cases
            self.print_to_log_file(
                "[BoundaryFragmentV3] dataloader overfit subset active: "
                f"n_cases={len(overfit_cases)} cases={overfit_cases}"
            )
        dataset_tr = nnUNetDataset(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        dataset_val = nnUNetDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        forced_distribution = self._boundary_fragment_forced_distribution()
        self.print_to_log_file(
            "[BoundaryFragmentV3] train forced class distribution",
            [(int(k), float(v)) for k, v in forced_distribution],
        )
        dl_tr = PengwinBoundaryFragmentDataLoader3D(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            forced_class_distribution=forced_distribution,
        )
        dl_val = nnUNetDataLoader3D(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=0.0,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
        )
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def _boundary_fragment_forced_distribution(self):
        if self._pengwin_oversample_profile == "boundary_fragment_v3_ridge80":
            return PengwinBoundaryFragmentDataLoader3D.RIDGE80_CLASS_DISTRIBUTION
        return PengwinBoundaryFragmentDataLoader3D.FORCED_CLASS_DISTRIBUTION

    def _prepare_boundary_fragment_target(self, target):
        """Return target labels in the BoundaryFragmentV3 contract."""
        return target

    @staticmethod
    def _binary_counts(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        pred = pred.bool()
        target = target.bool()
        tp = float((pred & target).sum().detach().cpu())
        fp = float((pred & ~target).sum().detach().cpu())
        fn = float((~pred & target).sum().detach().cpu())
        tn = float((~pred & ~target).sum().detach().cpu())
        return np.asarray([tp, fp, fn, tn], dtype=np.float64)

    @staticmethod
    def _precision_recall_f(tp: float, fp: float, fn: float, beta: float = 1.0) -> tuple[float, float, float]:
        eps = 1e-8
        p = tp / max(tp + fp, eps)
        r = tp / max(tp + fn, eps)
        beta2 = float(beta) ** 2
        f = (1.0 + beta2) * p * r / max(beta2 * p + r, eps)
        return float(p), float(r), float(f)

    @staticmethod
    def _boundary_fragment_stable_score(metrics: dict[str, float]) -> float:
        """Checkpoint score that rejects barrier collapse and broad false masks."""
        barrier_p = float(metrics.get("barrier_precision", 0.0))
        barrier_r = float(metrics.get("barrier_recall", 0.0))
        barrier_f05 = float(metrics.get("barrier_f0_5", 0.0))
        barrier_ratio = float(metrics.get("barrier_pred_target_ratio", 0.0))
        core_f1 = float(metrics.get("core_f1", 0.0))
        shell_f1 = float(metrics.get("shell_f1", 0.0))
        support_f1 = float(metrics.get("support_f1", 0.0))

        ratio_safe = max(barrier_ratio, 1e-6)
        ratio_score = float(np.exp(-abs(np.log(ratio_safe))))
        if barrier_f05 <= 0.0 or barrier_p < 0.01 or barrier_r < 0.05:
            return 0.0
        if barrier_ratio < 0.5 or barrier_ratio > 12.0:
            return 0.0
        base = (
            0.18 * support_f1
            + 0.18 * shell_f1
            + 0.30 * barrier_f05
            + 0.14 * barrier_p
            + 0.10 * barrier_r
            + 0.10 * core_f1
            + 0.05 * ratio_score
        )

        # [QA][Gate:checkpoint_selection][Status:Measured on V173/V174 logs]
        # V173 epoch 3 was the useful checkpoint, while the next epoch overfit
        # into an 80x class-2 mask. The gates below keep high-core false-barrier
        # epochs from winning checkpoint_best or checkpoint_barrier_stable.
        gate = 1.0
        if barrier_ratio < 1.0 or barrier_ratio > 8.0:
            gate *= 0.50
        if barrier_p < 0.02:
            gate *= 0.50
        if barrier_r < 0.10:
            gate *= 0.50
        if core_f1 < 0.10:
            gate *= 0.50
        return float(base * gate)

    def _move_boundary_fragment_data(self, data) -> torch.Tensor:
        if torch.is_tensor(data):
            return data.to(self.device, non_blocking=True)
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def _move_boundary_fragment_target(self, target):
        if isinstance(target, dict):
            return {key: self._move_boundary_fragment_target(value) for key, value in target.items()}
        if isinstance(target, list):
            return [self._move_boundary_fragment_target(item) for item in target]
        if isinstance(target, tuple):
            return tuple(self._move_boundary_fragment_target(item) for item in target)
        if torch.is_tensor(target):
            return target.to(self.device, non_blocking=True)
        # [QC][Invariant:target_device_dtype]
        # Sidecar dataloaders bypass nnU-Net's torch transforms and return numpy
        # integer labels. Convert explicitly here so the shared BFV3 loss sees
        # the same device contract as native nnU-Net augmented batches.
        return torch.as_tensor(target, dtype=torch.long, device=self.device)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        fullres_target = target_for_loss[0] if isinstance(target_for_loss, list) else target_for_loss
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        pred = torch.argmax(logits.float(), dim=1)
        labels = fullres_target[:, 0].long() if fullres_target.ndim == 5 else fullres_target.long()
        support_p = (pred == 2) | (pred == 3) | (pred == 4)
        support_t = (labels == 2) | (labels == 3) | (labels == 4)
        barrier_p = pred == 2
        barrier_t = labels == 2
        core_p = pred == 4
        core_t = labels == 4
        shell_p = (pred == 2) | (pred == 3)
        shell_t = (labels == 2) | (labels == 3)
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(core_p, core_t),
            "shell_counts": self._binary_counts(shell_p, shell_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }

    def on_validation_epoch_end(self, val_outputs):
        outputs = collate_outputs(val_outputs)
        loss_here = float(np.mean(outputs["loss"]))
        support = np.sum(outputs["support_counts"], axis=0)
        barrier = np.sum(outputs["barrier_counts"], axis=0)
        core = np.sum(outputs["core_counts"], axis=0)
        shell = np.sum(outputs["shell_counts"], axis=0)
        barrier_ratio_raw = np.sum(outputs["pred_target_barrier"], axis=0)
        support_p, support_r, support_f1 = self._precision_recall_f(*support[:3], beta=1.0)
        barrier_p, barrier_r, barrier_f05 = self._precision_recall_f(*barrier[:3], beta=0.5)
        core_p, core_r, core_f1 = self._precision_recall_f(*core[:3], beta=1.0)
        shell_p, shell_r, shell_f1 = self._precision_recall_f(*shell[:3], beta=1.0)
        barrier_ratio = float(barrier_ratio_raw[0] / max(barrier_ratio_raw[1], 1.0))
        ratio_score = float(np.exp(-abs(np.log(max(barrier_ratio, 1e-6)))))
        if self._pengwin_loss_profile == "boundary_fragment_v3_ridgefit_precision":
            score = (
                0.15 * support_f1
                + 0.10 * shell_f1
                + 0.30 * barrier_f05
                + 0.20 * barrier_p
                + 0.10 * barrier_r
                + 0.10 * core_f1
                + 0.05 * ratio_score
            )
            if barrier_p < 0.20 or barrier_r < 0.30 or barrier_ratio > 12.0:
                score *= 0.10
            elif barrier_p < 0.30 or barrier_ratio > 5.0:
                score *= 0.50
        elif self._pengwin_loss_profile == "boundary_fragment_v3_ridgefit":
            score = (
                0.20 * support_f1
                + 0.15 * shell_f1
                + 0.35 * barrier_f05
                + 0.15 * barrier_r
                + 0.10 * core_f1
                + 0.05 * ratio_score
            )
            if barrier_p < 0.30 or barrier_r < 0.05 or barrier_ratio > 20.0:
                score *= 0.20
        else:
            score = (
                0.30 * support_f1
                + 0.25 * shell_f1
                + 0.25 * barrier_f05
                + 0.10 * core_f1
                + 0.10 * ratio_score
            )
            if barrier_p < 0.05 or barrier_r < 0.10 or barrier_ratio > 20.0:
                score *= 0.20
        metrics = {
            "score": score,
            "support_f1": support_f1,
            "support_precision": support_p,
            "support_recall": support_r,
            "barrier_precision": barrier_p,
            "barrier_recall": barrier_r,
            "barrier_f0_5": barrier_f05,
            "barrier_pred_target_ratio": barrier_ratio,
            "core_f1": core_f1,
            "core_precision": core_p,
            "core_recall": core_r,
            "shell_f1": shell_f1,
            "shell_precision": shell_p,
            "shell_recall": shell_r,
            "ratio_score": ratio_score,
        }
        stable_score = self._boundary_fragment_stable_score(metrics)
        metrics["score_before_stable_gate"] = score
        metrics["barrier_stable_score"] = stable_score
        if self.BOUNDARY_FRAGMENT_USE_STABLE_SCORE:
            score = stable_score
            metrics["score"] = score
        self._last_boundary_fragment_metrics = metrics
        self.logger.log("mean_fg_dice", score, self.current_epoch)
        self.logger.log(
            "dice_per_class_or_region",
            [support_f1, shell_f1, barrier_f05, core_f1, ratio_score],
            self.current_epoch,
        )
        self.logger.log("val_losses", loss_here, self.current_epoch)

        completed = self.current_epoch + 1
        if completed >= self.ES_MIN_EPOCHS:
            ema = float(self.logger.my_fantastic_logging["ema_fg_dice"][-1])
            if ema > self._es_best_score + self.ES_MIN_DELTA:
                self._es_best_score = ema
                self._es_no_improve = 0
            else:
                self._es_no_improve += 1
                if self._es_no_improve >= self.ES_PATIENCE and not self._es_triggered:
                    self._es_triggered = True
                    self.print_to_log_file(
                        f"Early stopping: no BoundaryFragmentV3 score improvement for "
                        f"{self.ES_PATIENCE} epochs. Best EMA score={self._es_best_score:.4f}."
                    )
                    self.num_epochs = completed

    def run_training(self):
        # [ES-ENFORCE][2026-06-05] nnUNetTrainer.run_training drives the epoch loop with
        # `for epoch in range(self.current_epoch, self.num_epochs)`, whose range() is frozen
        # at loop entry. The ES handler in on_validation_epoch_end signals stop by lowering
        # self.num_epochs, but a frozen range ignores that — so ES *fired* (logged the message)
        # yet training kept running to the original num_epochs (observed 2026-06-04: ES at ep95
        # ignored, ran to ep206+). Mirror the base PengwinTrainer fix: re-check self.num_epochs
        # every epoch via a while-loop. Body is otherwise byte-identical to nnUNetTrainer.run_training;
        # on_epoch_end increments self.current_epoch, so after ES sets num_epochs=current_epoch+1 the
        # next while-check (current_epoch == num_epochs) exits cleanly and on_train_end runs.
        self.on_train_start()
        while self.current_epoch < self.num_epochs:
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time.time(), self.current_epoch)
        self.print_to_log_file("train_loss", np.round(self.logger.my_fantastic_logging["train_losses"][-1], decimals=4))
        self.print_to_log_file("val_loss", np.round(self.logger.my_fantastic_logging["val_losses"][-1], decimals=4))
        metrics = getattr(self, "_last_boundary_fragment_metrics", {})
        self.print_to_log_file("BoundaryFragmentV3 val", {k: round(float(v), 4) for k, v in metrics.items()})
        self.print_to_log_file(
            "Diagnostic heads [support_f1, shell_f1, barrier_f0.5, core_f1, ratio_score]",
            [np.round(i, decimals=4) for i in self.logger.my_fantastic_logging["dice_per_class_or_region"][-1]],
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.pth"))
        ema_score = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
        if self._best_ema is None or ema_score > self._best_ema:
            self._best_ema = ema_score
            self.print_to_log_file(f"New best EMA BoundaryFragmentV3 score: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_best.pth"))
        if self.BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT:
            stable_score = float(metrics.get("barrier_stable_score", -1.0))
            if stable_score > 0.0 and stable_score > self._best_barrier_stable_score:
                self._best_barrier_stable_score = stable_score
                self.print_to_log_file(
                    "New best raw barrier-stable checkpoint score: "
                    f"{np.round(self._best_barrier_stable_score, decimals=4)}"
                )
                self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_barrier_stable.pth"))
                if self.local_rank == 0:
                    stable_payload = {
                        "epoch": int(current_epoch),
                        "barrier_stable_score": stable_score,
                        "metrics": {k: float(v) for k, v in metrics.items()},
                    }
                    (Path(self.output_folder) / "checkpoint_barrier_stable_metrics.json").write_text(
                        json.dumps(stable_payload, indent=2)
                    )
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.print_to_log_file(
            "[BoundaryFragmentV3] native final validation skipped; run "
            "`eval.py boundary-fragment-eval` for fixed-decoder IoU-F, HD95, and NSD."
        )
        return None




class PengwinTrainerBoundaryFragmentSidecarV164(PengwinTrainerBoundaryFragmentV3):
    """Dataset537 V164: train BFV3 from dense instance-derived target sidecars."""

    NUM_EPOCHS_DEFAULT = 2
    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar"

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.enable_deep_supervision = False
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V164_EPOCHS", self.NUM_EPOCHS_DEFAULT))
        if os.environ.get("PENGWIN_BFV3_V164_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V164_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V164_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V164_VAL_ITERS"])
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self._pengwin_oversample_profile = os.environ.get(
            "PENGWIN_OVERSAMPLE_PROFILE", self.DEFAULT_OVERSAMPLE_PROFILE
        ).strip().lower()
        self.oversample_foreground_percent = float(os.environ.get("PENGWIN_BFV3_SIDECAR_OVERSAMPLE_FG", "0.90"))
        if os.environ.get("PENGWIN_BFV3_SIDECAR_BATCH_SIZE"):
            self.batch_size = int(os.environ["PENGWIN_BFV3_SIDECAR_BATCH_SIZE"])
        self.print_to_log_file(
            "[BoundaryFragmentSidecarV164] changed_variable=bfv3_target_sidecar "
            f"epochs={self.num_epochs} train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch} oversample_fg={self.oversample_foreground_percent:.2f} "
            f"batch_size={self.batch_size}"
        )

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        if len(patch_size) != 3:
            raise RuntimeError("BoundaryFragment sidecar training supports only 3D fullres training.")
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        instance_sidecar_dir = Path(self.preprocessed_dataset_folder) / "bicm_v5_instance_targets"
        target_sidecar_dir = Path(self.preprocessed_dataset_folder) / BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR
        if not instance_sidecar_dir.is_dir():
            raise FileNotFoundError(f"BICM V5 instance sidecar directory missing: {instance_sidecar_dir}")
        if not target_sidecar_dir.is_dir():
            raise FileNotFoundError(
                f"BoundaryFragment V3 target sidecar directory missing: {target_sidecar_dir}"
            )
        edge_prob = float(os.environ.get(
            "PENGWIN_BFV3_SIDECAR_CONTACT_CENTER_PROB",
            str(getattr(self, "EDGE_CENTER_PROBABILITY", 0.65)),
        ))
        support_prob = float(os.environ.get("PENGWIN_BFV3_SIDECAR_SUPPORT_CENTER_PROB", "0.15"))
        tiny_prob = float(os.environ.get("PENGWIN_BFV3_SIDECAR_TINY_CENTER_PROB", "0.15"))
        hard_prob = float(os.environ.get("PENGWIN_BFV3_SIDECAR_HARD_CENTER_PROB", "0.05"))
        core_prob = float(os.environ.get(
            "PENGWIN_BFV3_SIDECAR_CORE_CENTER_PROB",
            str(getattr(self, "CORE_CENTER_PROBABILITY", 0.0)),
        ))
        self.print_to_log_file(
            "[BoundaryFragmentSidecarV164] sidecar dirs "
            f"instance={instance_sidecar_dir} target={target_sidecar_dir} "
            f"center_probs(core/contact/support/tiny/hard)=({core_prob},{edge_prob},{support_prob},{tiny_prob},{hard_prob})"
        )
        dl_tr = PengwinBoundaryFragmentTargetSidecarDataLoader3D(
            dataset_tr,
            self.batch_size,
            patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=None,
            sidecar_dir=instance_sidecar_dir,
            target_sidecar_dir=target_sidecar_dir,
            edge_center_probability=edge_prob,
            support_negative_center_probability=support_prob,
            tiny_center_probability=tiny_prob,
            hard_negative_center_probability=hard_prob,
            core_center_probability=core_prob,
        )
        dl_val = PengwinBoundaryFragmentTargetSidecarDataLoader3D(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=0.33,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=None,
            sidecar_dir=instance_sidecar_dir,
            target_sidecar_dir=target_sidecar_dir,
            edge_center_probability=0.40,
            support_negative_center_probability=0.30,
            tiny_center_probability=0.20,
            hard_negative_center_probability=0.10,
            core_center_probability=min(0.50, max(0.0, core_prob)),
        )
        # [V0.x][NOTE:DATALOADER][2026-06-04] SingleThreadedAugmenter kept on purpose:
        # NonDetMultiThreadedAugmenter was measured WORSE here (GPU util 5/12 vs 8/12,
        # workers spawn poorly under DDP for this custom sidecar loader). The real lever
        # is reducing per-sample CPU in the _fragment_* loops (see _present_fragment_slices
        # bbox optimization), not background workers.
        mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.print_to_log_file(
            "[BoundaryFragmentSidecarV164] native final validation skipped; "
            "run Dataset537 ROI prediction plus fixed hard-case gate."
        )
        return None






class PengwinTrainerBoundaryFragmentSidecarCoreRidgeV167(PengwinTrainerBoundaryFragmentSidecarV164):
    """Dataset537 V167: BFV3 sidecar target with class-2 ridge and class-4 core loss."""

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core_ridge"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_ridge"
    CORE_CENTER_PROBABILITY = 0.40

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V167_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V167_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V167_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V167_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V167_VAL_ITERS"])
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self._pengwin_oversample_profile = os.environ.get(
            "PENGWIN_OVERSAMPLE_PROFILE", self.DEFAULT_OVERSAMPLE_PROFILE
        ).strip().lower()
        self.oversample_foreground_percent = float(os.environ.get("PENGWIN_BFV3_SIDECAR_OVERSAMPLE_FG", "0.95"))
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRidgeV167] changed_variable=class2_ridge_plus_class4_core_loss "
            f"epochs={self.num_epochs} train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch}"
        )


class PengwinTrainerBoundaryFragmentSidecarCoreRidgeHighEdgeV168(
    PengwinTrainerBoundaryFragmentSidecarCoreRidgeV167
):
    """Dataset537 V168: V167 loss with restored class-2 crop exposure."""

    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_ridge_high_edge"
    EDGE_CENTER_PROBABILITY = 0.90

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V168_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V168_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V168_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V168_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V168_VAL_ITERS"])
        self._pengwin_oversample_profile = os.environ.get(
            "PENGWIN_OVERSAMPLE_PROFILE", self.DEFAULT_OVERSAMPLE_PROFILE
        ).strip().lower()
        # [AUDIT][Risk:High][Scope:barrier_positive_exposure]
        # V167's 40% core-center policy leaves only 39% effective edge-center
        # exposure. V168 keeps the loss fixed and raises only class-2 crop
        # exposure to test whether full-fold barrier collapse is sampler-driven.
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRidgeHighEdgeV168] changed_variable=high_class2_crop_exposure "
            f"edge_center_probability={self.EDGE_CENTER_PROBABILITY:.2f} "
            f"epochs={self.num_epochs} train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch}"
        )


class PengwinTrainerBoundaryFragmentSidecarCoreRecallRidgeV169(
    PengwinTrainerBoundaryFragmentSidecarCoreRidgeHighEdgeV168
):
    """Dataset537 V169: high-edge exposure plus recall-biased class-2 ridge loss."""

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core_ridge_recall"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V169_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V169_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V169_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V169_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V169_VAL_ITERS"])
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self._pengwin_oversample_profile = os.environ.get(
            "PENGWIN_OVERSAMPLE_PROFILE", self.DEFAULT_OVERSAMPLE_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallRidgeV169] changed_variable=recall_biased_class2_ridge_loss "
            f"edge_center_probability={self.EDGE_CENTER_PROBABILITY:.2f} "
            f"epochs={self.num_epochs} train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch}"
        )




















class PengwinTrainerBoundaryFragmentSidecarCoreRecallLogitCalibratedV179(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallRidgeV169
):
    """Dataset537 V179: class-2 logit calibration on V178-style BFV3 scaffold."""

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core_ridge_recall_logit_calibrated"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = True
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = True

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V179_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V179_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V179_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V179_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V179_VAL_ITERS"])
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self._pengwin_oversample_profile = os.environ.get(
            "PENGWIN_OVERSAMPLE_PROFILE", self.DEFAULT_OVERSAMPLE_PROFILE
        ).strip().lower()
        # [AUDIT][Risk:High][Scope:barrier_threshold_calibration]
        # V178 preserved support/core but produced no class-2 voxels at threshold
        # 0.5 on hard-case prediction. V179 changes only class-2 logit calibration
        # and keeps BFV3 target labels, high-edge sampler, and strict stable gate.
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallLogitCalibratedV179] "
            "changed_variable=balanced_class2_logit_calibration "
            f"edge_center_probability={self.EDGE_CENTER_PROBABILITY:.2f} "
            f"epochs={self.num_epochs} train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch}"
        )








class PengwinTrainerBoundaryFragmentSidecarCoreRecallBinaryBarrierHeadV183(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallLogitCalibratedV179
):
    """Dataset537 V183: BFV3 semantic scaffold plus independent barrier head."""

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core_ridge_recall_binary_barrier_head"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = True
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = True

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        # [QC][Invariant:output_contract]
        # The sixth barrier head is evaluated at full resolution only. Disable
        # deep supervision so auxiliary scales do not call five-class BFV3 losses
        # on six-channel tensors.
        self.enable_deep_supervision = False
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V183_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V183_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V183_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V183_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V183_VAL_ITERS"])
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self._pengwin_oversample_profile = os.environ.get(
            "PENGWIN_OVERSAMPLE_PROFILE", self.DEFAULT_OVERSAMPLE_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallBinaryBarrierHeadV183] "
            "output_channels=6 changed_variable=independent_binary_barrier_head "
            f"edge_center_probability={self.EDGE_CENTER_PROBABILITY:.2f} "
            f"epochs={self.num_epochs} train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch}"
        )

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [AUDIT][Risk:High][Scope:bfv3_output_contract]
        # The label manager still reports five BFV3 labels. V183 deliberately
        # adds one independent binary barrier logit while preserving the semantic
        # channels 0..4 for support/core decoding.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_BINARY_BARRIER_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        fullres_target = target_for_loss[0] if isinstance(target_for_loss, list) else target_for_loss
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_BINARY_BARRIER_OUTPUT_CHANNELS):
            raise ValueError(f"V183 expected validation logits [B, 6, D, H, W], got {tuple(logits.shape)}")
        semantic_logits = logits[:, :5].float()
        barrier_aux = torch.sigmoid(logits[:, 5].float()) > 0.5
        semantic_pred = torch.argmax(semantic_logits, dim=1)
        labels = fullres_target[:, 0].long() if fullres_target.ndim == 5 else fullres_target.long()
        support_p = (semantic_pred == 2) | (semantic_pred == 3) | (semantic_pred == 4)
        support_t = (labels == 2) | (labels == 3) | (labels == 4)
        barrier_p = barrier_aux
        barrier_t = labels == 2
        core_p = semantic_pred == 4
        core_t = labels == 4
        shell_p = barrier_aux | (semantic_pred == 3)
        shell_t = (labels == 2) | (labels == 3)
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(core_p, core_t),
            "shell_counts": self._binary_counts(shell_p, shell_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }








class PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateBinaryBarrierHeadV187(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallBinaryBarrierHeadV183
):
    """Dataset537 V187: candidate-restricted binary barrier head."""

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core_ridge_recall_candidate_binary_barrier_head"

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V187_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V187_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V187_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V187_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V187_VAL_ITERS"])
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallCandidateBinaryBarrierHeadV187] "
            "output_channels=6 changed_variable=candidate_restricted_binary_barrier_head "
            f"edge_center_probability={self.EDGE_CENTER_PROBABILITY:.2f} "
            f"epochs={self.num_epochs} train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch}"
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        fullres_target = target_for_loss[0] if isinstance(target_for_loss, list) else target_for_loss
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_BINARY_BARRIER_OUTPUT_CHANNELS):
            raise ValueError(f"V187 expected validation logits [B, 6, D, H, W], got {tuple(logits.shape)}")
        semantic_logits = logits[:, :5].float()
        semantic_prob = torch.softmax(semantic_logits, dim=1)
        semantic_pred = torch.argmax(semantic_logits, dim=1)
        candidate_p = semantic_prob[:, 2:4].sum(dim=1) > 0.5
        barrier_p = (torch.sigmoid(logits[:, 5].float()) > 0.5) & candidate_p
        labels = fullres_target[:, 0].long() if fullres_target.ndim == 5 else fullres_target.long()
        support_p = (semantic_pred == 2) | (semantic_pred == 3) | (semantic_pred == 4)
        support_t = (labels == 2) | (labels == 3) | (labels == 4)
        barrier_t = labels == 2
        core_p = semantic_pred == 4
        core_t = labels == 4
        shell_p = barrier_p | (semantic_pred == 3)
        shell_t = (labels == 2) | (labels == 3)
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(core_p, core_t),
            "shell_counts": self._binary_counts(shell_p, shell_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }












class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBinaryBarrierHeadV193(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateBinaryBarrierHeadV187
):
    """Dataset537 V193: dense local aux-positive target around class2 barrier."""

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core_ridge_recall_dense_candidate_binary_barrier_head"

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V193_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V193_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V193_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V193_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V193_VAL_ITERS"])
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateBinaryBarrierHeadV193] "
            "output_channels=6 changed_variable=dense_local_barrier_aux_positive_target "
            f"edge_center_probability={self.EDGE_CENTER_PROBABILITY:.2f} "
            f"epochs={self.num_epochs} train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch}"
        )
























class _PengwinTrainerBoundaryFragmentSeedHeadBase(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBinaryBarrierHeadV193
):
    """Shared Dataset537 BFV3 seed-head trainer contract."""

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [AUDIT][Risk:High][Scope:bfv3_seed_output_contract]
        # Channels 0..4 remain semantic BFV3, channel 5 remains the V197
        # barrier aux head, and channel 6 is the new fragment-scale seed head.
        # This keeps contact/core structure comparable while isolating seed
        # topology as the only output-contract change.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        if len(patch_size) != 3:
            raise RuntimeError("BoundaryFragment seed-head sidecar training supports only 3D fullres training.")
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        instance_sidecar_dir = Path(self.preprocessed_dataset_folder) / "bicm_v5_instance_targets"
        target_sidecar_dir = Path(self.preprocessed_dataset_folder) / BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR
        if not instance_sidecar_dir.is_dir():
            raise FileNotFoundError(f"BICM V5 instance sidecar directory missing: {instance_sidecar_dir}")
        if not target_sidecar_dir.is_dir():
            raise FileNotFoundError(
                f"BoundaryFragment V3 target sidecar directory missing: {target_sidecar_dir}"
            )
        edge_prob = float(os.environ.get(
            "PENGWIN_BFV3_SIDECAR_CONTACT_CENTER_PROB",
            str(getattr(self, "EDGE_CENTER_PROBABILITY", 0.65)),
        ))
        support_prob = float(os.environ.get("PENGWIN_BFV3_SIDECAR_SUPPORT_CENTER_PROB", "0.15"))
        tiny_prob = float(os.environ.get("PENGWIN_BFV3_SIDECAR_TINY_CENTER_PROB", "0.15"))
        hard_prob = float(os.environ.get("PENGWIN_BFV3_SIDECAR_HARD_CENTER_PROB", "0.05"))
        core_prob = float(os.environ.get(
            "PENGWIN_BFV3_SIDECAR_CORE_CENTER_PROB",
            str(getattr(self, "CORE_CENTER_PROBABILITY", 0.0)),
        ))
        seed_radius_vox = float(os.environ.get("PENGWIN_BFV3_SEED_RADIUS_VOX", "2.0"))
        # [AUDIT][Risk:High][Scope:v283_contact_removed_sampler]
        # Active V283 경로는 contact를 loss/input/decoder뿐 아니라 forced crop sampler에서도
        # 제거한다. contact-centered crop exposure가 남아 있으면 "contact를 버린 구조"라는
        # 실험 질문이 흐려진다. foreground crop은 instance sidecar priority center로만 간다.
        decoder_contact_center_prob = 0.0
        decoder_contact_negative_center_prob = 0.0
        decoder_contact_center_source = os.environ.get(
            "PENGWIN_BFV3_DECODER_CONTACT_CENTER_SOURCE",
            str(getattr(self, "DECODER_CONTACT_CENTER_SOURCE", "v5")),
        ).strip().lower()
        val_decoder_contact_center_prob = 0.0
        val_decoder_contact_negative_center_prob = 0.0
        val_oversample_foreground_percent = float(os.environ.get(
            "PENGWIN_BFV3_VAL_OVERSAMPLE_FOREGROUND_PERCENT",
            str(getattr(self, "VAL_OVERSAMPLE_FOREGROUND_PERCENT", 0.33)),
        ))
        self.print_to_log_file(
            "[BoundaryFragmentSeedHead] sidecar dirs "
            f"instance={instance_sidecar_dir} target={target_sidecar_dir} "
            f"seed_radius_vox={seed_radius_vox} "
            f"center_probs(core/contact/support/tiny/hard)=({core_prob},{edge_prob},{support_prob},{tiny_prob},{hard_prob}) "
            f"decoder_contact_center_probability={decoder_contact_center_prob} "
            f"decoder_contact_negative_center_probability={decoder_contact_negative_center_prob} "
            f"decoder_contact_center_source={decoder_contact_center_source} "
            f"val_decoder_contact_center_probability={val_decoder_contact_center_prob} "
            f"val_decoder_contact_negative_center_probability={val_decoder_contact_negative_center_prob} "
            f"val_oversample_foreground_percent={val_oversample_foreground_percent}"
        )
        dl_tr = PengwinBoundaryFragmentSeedTargetSidecarDataLoader3D(
            dataset_tr,
            self.batch_size,
            patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=None,
            sidecar_dir=instance_sidecar_dir,
            target_sidecar_dir=target_sidecar_dir,
            seed_radius_vox=seed_radius_vox,
            decoder_contact_center_probability=decoder_contact_center_prob,
            decoder_contact_negative_center_probability=decoder_contact_negative_center_prob,
            decoder_contact_center_source=decoder_contact_center_source,
            edge_center_probability=edge_prob,
            support_negative_center_probability=support_prob,
            tiny_center_probability=tiny_prob,
            hard_negative_center_probability=hard_prob,
            core_center_probability=core_prob,
        )
        dl_val = PengwinBoundaryFragmentSeedTargetSidecarDataLoader3D(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=val_oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=None,
            sidecar_dir=instance_sidecar_dir,
            target_sidecar_dir=target_sidecar_dir,
            seed_radius_vox=seed_radius_vox,
            decoder_contact_center_probability=val_decoder_contact_center_prob,
            decoder_contact_negative_center_probability=val_decoder_contact_negative_center_prob,
            decoder_contact_center_source=decoder_contact_center_source,
            edge_center_probability=0.40,
            support_negative_center_probability=0.30,
            tiny_center_probability=0.20,
            hard_negative_center_probability=0.10,
            core_center_probability=min(0.50, max(0.0, core_prob)),
        )
        # [V0.x][NOTE:DATALOADER][2026-06-04] SingleThreadedAugmenter kept on purpose:
        # NonDetMultiThreadedAugmenter was measured WORSE here (GPU util 5/12 vs 8/12,
        # workers spawn poorly under DDP for this custom sidecar loader). The real lever
        # is reducing per-sample CPU in the _fragment_* loops (see _present_fragment_slices
        # bbox optimization), not background workers.
        mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        semantic_target = target_for_loss["semantic"]
        seed_target = target_for_loss["seed"]
        if seed_target.ndim == 5 and seed_target.shape[1] == 1:
            seed_target = seed_target[:, 0]
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS):
            raise ValueError(f"BFV3 seed-head expected validation logits [B, 7, D, H, W], got {tuple(logits.shape)}")
        semantic_logits = logits[:, :5].float()
        semantic_prob = torch.softmax(semantic_logits, dim=1)
        semantic_pred = torch.argmax(semantic_logits, dim=1)
        candidate_p = semantic_prob[:, 2:4].sum(dim=1) > 0.5
        barrier_p = (torch.sigmoid(logits[:, 5].float()) > 0.5) & candidate_p
        seed_p = torch.sigmoid(logits[:, 6].float()) > 0.5
        labels = semantic_target[:, 0].long() if semantic_target.ndim == 5 else semantic_target.long()
        seed_t = seed_target.float() > 0.5
        support_p = (semantic_pred == 2) | (semantic_pred == 3) | (semantic_pred == 4)
        support_t = (labels == 2) | (labels == 3) | (labels == 4)
        barrier_t = labels == 2
        shell_p = barrier_p | (semantic_pred == 3)
        shell_t = (labels == 2) | (labels == 3)
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(seed_p, seed_t),
            "shell_counts": self._binary_counts(shell_p, shell_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }


class _PengwinTrainerBoundaryFragmentInstanceCoreGuardBase(
    _PengwinTrainerBoundaryFragmentSeedHeadBase
):
    """Six-channel BFV3 trainer that exposes instance sidecars only to the loss."""

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [AUDIT][Risk:High][Scope:bfv3_output_contract]
        # V213/V214 keep the V197 six-channel output contract. Instance sidecars
        # are consumed by the loss only; no new inference channel or decoder rule
        # is introduced for this training-level topology test.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_BINARY_BARRIER_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if not isinstance(target_for_loss, dict) or "semantic" not in target_for_loss:
            raise ValueError("instance-core-guard validation requires target dict with semantic labels")
        semantic_target = target_for_loss["semantic"]
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_BINARY_BARRIER_OUTPUT_CHANNELS):
            raise ValueError(f"BFV3 instance-core-guard expected validation logits [B, 6, D, H, W], got {tuple(logits.shape)}")
        semantic_logits = logits[:, :5].float()
        barrier_aux = torch.sigmoid(logits[:, 5].float()) > 0.5
        semantic_pred = torch.argmax(semantic_logits, dim=1)
        labels = semantic_target[:, 0].long() if semantic_target.ndim == 5 else semantic_target.long()
        support_p = (semantic_pred == 2) | (semantic_pred == 3) | (semantic_pred == 4)
        support_t = (labels == 2) | (labels == 3) | (labels == 4)
        barrier_p = barrier_aux
        barrier_t = labels == 2
        core_p = semantic_pred == 4
        core_t = labels == 4
        shell_p = barrier_aux | (semantic_pred == 3)
        shell_t = (labels == 2) | (labels == 3)
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(core_p, core_t),
            "shell_counts": self._binary_counts(shell_p, shell_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }
































class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakXYZAffinityV248(
    _PengwinTrainerBoundaryFragmentInstanceCoreGuardBase
):
    """Dataset537 V248: BFV3 semantic scaffold plus z/y/x same-fragment affinity.

    [AUDIT][Risk:High][Scope:representation_pivot]
    V247의 scalar decoder-contact head는 0.50 기준에서 contact가 열리면 shell
    false positive가 커지고, hard negative를 세게 걸면 contact recall이 닫혔다.
    V248은 channel5 하나를 더 튜닝하지 않고, decoder가 fragment를 나누는 데 필요한
    pairwise relation을 channel 5..7에 직접 둔다.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_xyz_affinity_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    DECODER_CONTACT_CENTER_SOURCE = "bfv3"
    DECODER_CONTACT_CENTER_PROBABILITY = 0.45
    DECODER_CONTACT_NEGATIVE_CENTER_PROBABILITY = 0.25
    VAL_DECODER_CONTACT_CENTER_PROBABILITY = 0.40
    VAL_DECODER_CONTACT_NEGATIVE_CENTER_PROBABILITY = 0.30
    VAL_OVERSAMPLE_FOREGROUND_PERCENT = 1.00
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = False
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [QC][Invariant:output_contract]
        # V248 output is exactly 8 channels: 0..4 BFV3 semantic logits and
        # 5..7 z/y/x same-fragment affinity logits. This is intentionally not
        # V38's support/contact/core/roi layout despite the same channel count.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_XYZ_AFFINITY_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V248_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V248_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V248_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V248_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V248_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V248_BATCH_SIZE", "").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V248_BATCH_SIZE must be >= 1, got {batch_size}")
            # [REPRO][Risk:Major][Scope:memory_gate]
            # V248 adds two output channels over V247. On RTX 3090, the first
            # 10e20i attempt with batch_size=2 sampled above the fixed 22118 MiB
            # gate. This override is explicit in the launch command and log so
            # downstream metrics are not compared as if batch size were unchanged.
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakXYZAffinityV248] "
            "output_channels=8 changed_variable=bfv3_semantic_plus_xyz_same_fragment_affinity "
            "contact_contract=boundary_fragment_xyz_affinity_v1 "
            f"decoder_contact_center_source={self.DECODER_CONTACT_CENTER_SOURCE} "
            f"decoder_contact_center_probability={self.DECODER_CONTACT_CENTER_PROBABILITY:.2f} "
            f"decoder_contact_negative_center_probability={self.DECODER_CONTACT_NEGATIVE_CENTER_PROBABILITY:.2f} "
            f"val_decoder_contact_center_probability={self.VAL_DECODER_CONTACT_CENTER_PROBABILITY:.2f} "
            f"val_decoder_contact_negative_center_probability={self.VAL_DECODER_CONTACT_NEGATIVE_CENTER_PROBABILITY:.2f} "
            f"batch_size={self.batch_size} "
            f"epochs={self.num_epochs} train_iters={self.num_iterations_per_epoch} "
            f"val_iters={self.num_val_iterations_per_epoch}"
        )

    @staticmethod
    def _edge_mask_to_voxel_mask(edge_mask: torch.Tensor) -> torch.Tensor:
        """Convert z/y/x edge mask [B,3,Z,Y,X] to endpoint voxel mask [B,Z,Y,X]."""
        if edge_mask.ndim != 5 or int(edge_mask.shape[1]) != 3:
            raise ValueError(f"expected edge mask [B,3,Z,Y,X], got {tuple(edge_mask.shape)}")
        out = torch.zeros(
            (int(edge_mask.shape[0]), *edge_mask.shape[2:]),
            dtype=torch.bool,
            device=edge_mask.device,
        )
        for axis in range(3):
            src_sl = [slice(None)] * 4
            dst_sl = [slice(None)] * 4
            src_sl[axis + 1] = slice(0, -1)
            dst_sl[axis + 1] = slice(1, None)
            axis_edges = edge_mask[:, axis]
            out[tuple(src_sl)] = out[tuple(src_sl)] | axis_edges[tuple(src_sl)]
            out[tuple(dst_sl)] = out[tuple(dst_sl)] | axis_edges[tuple(src_sl)]
        return out

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if (
            not isinstance(target_for_loss, dict)
            or "semantic" not in target_for_loss
            or "instance" not in target_for_loss
        ):
            raise ValueError("V248 validation requires target dict with semantic and instance labels")
        semantic_target = target_for_loss["semantic"]
        instance = target_for_loss["instance"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance_for_edges = instance[:, 0].long()
        else:
            instance_for_edges = instance.long()

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_XYZ_AFFINITY_OUTPUT_CHANNELS):
            raise ValueError(f"V248 expected validation logits [B, 8, D, H, W], got {tuple(logits.shape)}")
        semantic_logits = logits[:, :5].float()
        semantic_pred = torch.argmax(semantic_logits, dim=1)
        labels = semantic_target[:, 0].long() if semantic_target.ndim == 5 else semantic_target.long()

        affinity_target, affinity_valid = BoundaryFragmentV3Core025StrongPeakXYZAffinityHeadLoss._same_fragment_affinity_targets(
            instance_for_edges.to(device=logits.device)
        )
        affinity_prob = torch.sigmoid(logits[:, 5:8].float())
        # [METRIC][Scope:patch_affinity_diagnostic]
        # Patch validation에서 barrier_counts는 BFV3 label2가 아니라 pairwise break
        # edge를 endpoint voxel로 투영한 diagnostic이다. Full-volume promotion은
        # eval.py의 fixed decoder/IoU-F가 판단한다.
        edge_break_p = (affinity_prob < 0.5) & affinity_valid
        edge_break_t = affinity_valid & (affinity_target <= 0.5)
        barrier_p = self._edge_mask_to_voxel_mask(edge_break_p)
        barrier_t = self._edge_mask_to_voxel_mask(edge_break_t)

        support_p = (semantic_pred == 2) | (semantic_pred == 3) | (semantic_pred == 4)
        support_t = (labels == 2) | (labels == 3) | (labels == 4)
        core_p = semantic_pred == 4
        core_t = labels == 4
        shell_p = barrier_p | (semantic_pred == 3)
        shell_t = (labels == 2) | (labels == 3)
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(core_p, core_t),
            "shell_counts": self._binary_counts(shell_p, shell_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakAffinity13SeedHealedV253(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakXYZAffinityV248
):
    """Dataset537 V253: oracle-passing affinity13 + seed-healing train contract.

    [AUDIT][Risk:High][Scope:v253_full_train_entry]
    V251 8채널 oracle는 diagonal fragment를 split했고, V252 13-affinity oracle는
    disconnected same-instance island를 split했다. V253은 18채널 target과
    nearest-seed healing decoder로 hard trio oracle gate를 통과한 첫 contract다.
    이 trainer는 full train 진입 전 학습 경로가 그 contract를 그대로 보게 한다.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_affinity13_seed_healed_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = False
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [QC][Invariant:output_contract]
        # V253 output is exactly 18 independent logits:
        # 0..4 support/exterior/fracture/seed_center/seed_body,
        # 5..17 same-fragment affinity over 13 directed 26-neighborhood offsets.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_AFFINITY13_SEED_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V253_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V253_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V253_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V253_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V253_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V253_BATCH_SIZE", "").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V253_BATCH_SIZE must be >= 1, got {batch_size}")
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakAffinity13SeedHealedV253] "
            "output_channels=18 changed_variable=oracle_pass_affinity13_seed_healed_contract "
            "decoder=task1_v253_affinity13_seed_healed "
            f"batch_size={self.batch_size} epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch}"
        )

    @staticmethod
    def _affinity13_edge_mask_to_voxel_mask(edge_mask: torch.Tensor) -> torch.Tensor:
        if edge_mask.ndim != 5 or int(edge_mask.shape[1]) != 13:
            raise ValueError(f"expected edge mask [B,13,Z,Y,X], got {tuple(edge_mask.shape)}")
        out = torch.zeros(
            (int(edge_mask.shape[0]), *edge_mask.shape[2:]),
            dtype=torch.bool,
            device=edge_mask.device,
        )
        offsets = BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss.AFFINITY13_OFFSETS_ZYX
        shape = tuple(int(v) for v in edge_mask.shape[2:])
        for idx, offset in enumerate(offsets):
            src_sl_zyx, dst_sl_zyx = BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss._offset_slices(shape, offset)
            src_sl = (slice(None), *src_sl_zyx)
            dst_sl = (slice(None), *dst_sl_zyx)
            axis_edges = edge_mask[:, idx]
            out[src_sl] = out[src_sl] | axis_edges[(slice(None), *src_sl_zyx)]
            out[dst_sl] = out[dst_sl] | axis_edges[(slice(None), *src_sl_zyx)]
        return out

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if (
            not isinstance(target_for_loss, dict)
            or "semantic" not in target_for_loss
            or "instance" not in target_for_loss
            or "seed" not in target_for_loss
        ):
            raise ValueError("V253 validation requires semantic, instance, and seed targets")
        semantic_target = target_for_loss["semantic"]
        instance = target_for_loss["instance"]
        seed_target = target_for_loss["seed"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance_for_edges = instance[:, 0].long()
        else:
            instance_for_edges = instance.long()
        if seed_target.ndim == 5 and int(seed_target.shape[1]) == 1:
            seed_target = seed_target[:, 0]

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_AFFINITY13_SEED_OUTPUT_CHANNELS):
            raise ValueError(f"V253 expected validation logits [B, 18, D, H, W], got {tuple(logits.shape)}")
        labels = semantic_target[:, 0].long() if semantic_target.ndim == 5 else semantic_target.long()
        support_t = (instance_for_edges > 0) & (instance_for_edges <= MAX_INSTANCE_ID)
        fracture_t = labels == 2
        seed_t = seed_target.float() > 0.5
        support_p = torch.sigmoid(logits[:, 0].float()) >= 0.5
        fracture_p = torch.sigmoid(logits[:, 2].float()) >= 0.5
        seed_p = torch.sigmoid(logits[:, 4].float()) >= 0.5
        affinity_target, affinity_valid = BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss._same_fragment_affinity13_targets(
            instance_for_edges.to(device=logits.device)
        )
        affinity_prob = torch.sigmoid(logits[:, 5:].float())
        edge_break_p = (affinity_prob < 0.5) & affinity_valid
        edge_break_t = affinity_valid & (affinity_target <= 0.5)
        barrier_p = self._affinity13_edge_mask_to_voxel_mask(edge_break_p) | fracture_p
        barrier_t = self._affinity13_edge_mask_to_voxel_mask(edge_break_t) | fracture_t
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(seed_p, seed_t),
            "shell_counts": self._binary_counts(fracture_p | seed_p, fracture_t | seed_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedHealedV255(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakAffinity13SeedHealedV253
):
    """Dataset537 V255: attractive affinity plus repulsive mutex contact veto.

    [AUDIT][Risk:High][Scope:v255_fulltrain_gate]
    V254 kept one affinity logit and only changed hard-negative loss. Hard3
    forensic still joined all different-fragment/contact edges at threshold
    0.50. V255 changes the decoder-facing representation: the graph joins only
    when attractive affinity is high and repulsive mutex is low.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_mutex13_seed_healed_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = False
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [QC][Invariant:output_contract]
        # V255 output is 31 logits:
        # 0..4 base support/seed heads, 5..17 attractive same-fragment affinity,
        # 18..30 repulsive different-fragment/contact mutex veto.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_MUTEX13_SEED_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V255_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V255_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V255_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V255_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V255_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V255_BATCH_SIZE", "").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V255_BATCH_SIZE must be >= 1, got {batch_size}")
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedHealedV255] "
            "output_channels=31 changed_variable=repulsive_mutex13_decoder_veto "
            "decoder=task1_v255_mutex13_seed_healed "
            f"batch_size={self.batch_size} epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch}"
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if (
            not isinstance(target_for_loss, dict)
            or "semantic" not in target_for_loss
            or "instance" not in target_for_loss
            or "seed" not in target_for_loss
        ):
            raise ValueError("V255 validation requires semantic, instance, and seed targets")
        semantic_target = target_for_loss["semantic"]
        instance = target_for_loss["instance"]
        seed_target = target_for_loss["seed"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance_for_edges = instance[:, 0].long()
        else:
            instance_for_edges = instance.long()
        if seed_target.ndim == 5 and int(seed_target.shape[1]) == 1:
            seed_target = seed_target[:, 0]

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_MUTEX13_SEED_OUTPUT_CHANNELS):
            raise ValueError(f"V255 expected validation logits [B, 31, D, H, W], got {tuple(logits.shape)}")
        labels = semantic_target[:, 0].long() if semantic_target.ndim == 5 else semantic_target.long()
        support_t = (instance_for_edges > 0) & (instance_for_edges <= MAX_INSTANCE_ID)
        fracture_t = labels == 2
        seed_t = seed_target.float() > 0.5
        support_p = torch.sigmoid(logits[:, 0].float()) >= 0.5
        fracture_p = torch.sigmoid(logits[:, 2].float()) >= 0.5
        seed_p = torch.sigmoid(logits[:, 4].float()) >= 0.5
        same_edge, contact_edge, valid_edge = BoundaryFragmentV3Core025StrongPeakMutex13SeedHealedHeadLoss._same_and_contact_edges(
            instance_for_edges.to(device=logits.device)
        )
        attractive_prob = torch.sigmoid(logits[:, 5:18].float())
        repulsive_prob = torch.sigmoid(logits[:, 18:31].float())
        edge_join_p = (attractive_prob >= 0.5) & (repulsive_prob < 0.5) & valid_edge
        edge_break_p = (~edge_join_p) & valid_edge
        barrier_p = self._affinity13_edge_mask_to_voxel_mask(edge_break_p) | fracture_p
        barrier_t = self._affinity13_edge_mask_to_voxel_mask(contact_edge) | fracture_t
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(seed_p, seed_t),
            "shell_counts": self._binary_counts(fracture_p | seed_p, fracture_t | seed_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakPairwiseSoftmaxMutex13SeedHealedV266(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedHealedV255
):
    """Dataset537 V266: pairwise-softmax graph partition contract.

    [AUDIT][Risk:Blocker][Scope:v266_fulltrain_gate]
    V265 restored support Dice/Local Dice but failed Task1 fracture and
    instance topology through false center markers. V266 returns to the
    oracle-pass 13-affinity partition family, but replaces independent
    attraction/repulsion sigmoid heads with join-vs-cut relative logits so the
    fixed decoder sees one mutually exclusive edge decision.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_pairwise_softmax_mutex13_seed_healed_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = True
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = True

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V266_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V266_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V266_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V266_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V266_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V266_BATCH_SIZE", "1").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V266_BATCH_SIZE must be >= 1, got {batch_size}")
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakPairwiseSoftmaxMutex13SeedHealedV266] "
            "output_channels=31 changed_variable=edge_pairwise_softmax_join_vs_cut_and_instance_contact_fracture_target "
            "decoder=task1_v266_pairwise_softmax_mutex13_seed_healed "
            f"batch_size={self.batch_size} epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch}"
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if (
            not isinstance(target_for_loss, dict)
            or "semantic" not in target_for_loss
            or "instance" not in target_for_loss
            or "seed" not in target_for_loss
        ):
            raise ValueError("V266 validation requires semantic, instance, and seed targets")
        semantic_target = target_for_loss["semantic"]
        instance = target_for_loss["instance"]
        seed_target = target_for_loss["seed"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance_for_edges = instance[:, 0].long()
        else:
            instance_for_edges = instance.long()
        if seed_target.ndim == 5 and int(seed_target.shape[1]) == 1:
            seed_target = seed_target[:, 0]

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_MUTEX13_SEED_OUTPUT_CHANNELS):
            raise ValueError(f"V266 expected validation logits [B, 31, D, H, W], got {tuple(logits.shape)}")
        labels = semantic_target[:, 0].long() if semantic_target.ndim == 5 else semantic_target.long()
        support_t = (instance_for_edges > 0) & (instance_for_edges <= MAX_INSTANCE_ID)
        same_edge, contact_edge, _leak_edge, valid_edge = BoundaryFragmentV3Core025StrongPeakPairwiseSoftmaxMutex13SeedHealedHeadLoss._same_contact_leak_edges(
            instance_for_edges.to(device=logits.device)
        )
        fracture_t = BoundaryFragmentV3Core025StrongPeakPairwiseSoftmaxMutex13SeedHealedHeadLoss._edge_mask_to_voxel_mask(
            contact_edge
        )
        seed_t = seed_target.float() > 0.5
        support_p = torch.sigmoid(logits[:, 0].float()) >= 0.5
        fracture_p = torch.sigmoid(logits[:, 2].float()) >= 0.5
        seed_p = torch.sigmoid(logits[:, 3].float()) >= 0.5
        edge_diff = logits[:, 5:18].float() - logits[:, 18:31].float()
        edge_break_p = (edge_diff < 0.0) & valid_edge
        edge_break_t = contact_edge
        barrier_p = self._affinity13_edge_mask_to_voxel_mask(edge_break_p) | fracture_p
        barrier_t = self._affinity13_edge_mask_to_voxel_mask(edge_break_t) | fracture_t
        # [METRIC][Scope:validation_proxy]
        # patch validation은 fulltrain gate가 아니지만, V266 edge decision은
        # sigmoid(row5)나 sigmoid(row18) 단독이 아니라 join-cut diff로만 해석해야 한다.
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(seed_p, seed_t),
            "shell_counts": self._binary_counts(fracture_p | seed_p, fracture_t | seed_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedV268(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakPairwiseSoftmaxMutex13SeedHealedV266
):
    """Dataset537 V268: contact head 없이 graph partition으로 instance identity를 학습한다.

    [AUDIT][Risk:Blocker][Scope:v268_fulltrain_gate]
    V267 forensic은 support는 학습되지만 center/offset identity가 붕괴됨을 보였다.
    V268은 support scaffold를 유지하고, Task1 Instance/Merge/Split/Topology가 직접
    보는 join-vs-cut edge decision으로 돌아간다. 별도 contact/fracture probability는
    학습하지 않는다.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_mutex13_seed_healed_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = True
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = True

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V268_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V268_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V268_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V268_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V268_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V268_BATCH_SIZE", "1").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V268_BATCH_SIZE must be >= 1, got {batch_size}")
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedV268] "
            "output_channels=31 changed_variable=remove_contact_head_use_pairwise_instance_identity "
            "decoder=task1_v268_no_contact_pairwise_softmax_mutex13_seed_healed "
            f"batch_size={self.batch_size} epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch}"
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if (
            not isinstance(target_for_loss, dict)
            or "instance" not in target_for_loss
            or "seed" not in target_for_loss
        ):
            raise ValueError("V268 validation requires instance and seed targets")
        instance = target_for_loss["instance"]
        seed_target = target_for_loss["seed"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance_for_edges = instance[:, 0].long()
        else:
            instance_for_edges = instance.long()
        if seed_target.ndim == 5 and int(seed_target.shape[1]) == 1:
            seed_target = seed_target[:, 0]

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_MUTEX13_SEED_OUTPUT_CHANNELS):
            raise ValueError(f"V268 expected validation logits [B, 31, D, H, W], got {tuple(logits.shape)}")

        support_t = (instance_for_edges > 0) & (instance_for_edges <= MAX_INSTANCE_ID)
        same_edge, different_fragment_edge, _leak_edge, valid_edge = (
            BoundaryFragmentV3Core025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedHeadLoss
            ._same_contact_leak_edges(instance_for_edges.to(device=logits.device))
        )
        seed_t = seed_target.float() > 0.5
        support_p = torch.sigmoid(logits[:, 0].float()) >= 0.5
        seed_p = torch.sigmoid(logits[:, 4].float()) >= 0.5
        edge_diff = logits[:, 5:18].float() - logits[:, 18:31].float()
        edge_cut_p = (edge_diff < 0.0) & valid_edge
        edge_cut_t = different_fragment_edge
        barrier_p = self._affinity13_edge_mask_to_voxel_mask(edge_cut_p)
        barrier_t = self._affinity13_edge_mask_to_voxel_mask(edge_cut_t)
        # [METRIC][Scope:validation_proxy]
        # V268 patch validation의 barrier는 contact head가 아니라 graph cut edge다.
        # fulltrain 판단은 이 patch proxy가 아니라 hard3 Task1 proxy와 visual audit로만 한다.
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(seed_p, seed_t),
            "shell_counts": self._binary_counts(barrier_p | seed_p, barrier_t | seed_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxSeedHealedV273(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedV268
):
    """Dataset537 V273: no-contact boundary-exact pairwise graph contract.

    [AUDIT][Risk:Blocker][Scope:v273_fulltrain_gate]
    V273 is allowed to proceed only because the no-contact pairwise oracle passed
    hard3 Task1 proxy and visual audit. It removes the inert/contact-like rows
    left in V268 and trains exactly the decoder-facing heads: support, seed_body,
    join13, cut13.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_seed_healed_v273_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = True
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = True

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [QC][Invariant:output_contract]
        # V273 output is exactly 28 logits:
        # 0 support, 1 seed_body, 2..14 join13, 15..27 cut13. There are no
        # contact/fracture/exterior compatibility rows.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_NO_CONTACT_PAIRWISE_V273_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V273_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V273_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V273_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V273_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V273_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V273_BATCH_SIZE", "1").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V273_BATCH_SIZE must be >= 1, got {batch_size}")
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxSeedHealedV273] "
            "output_channels=28 changed_variable=remove_contact_rows_boundary_exact_pairwise_join_cut "
            "decoder=task1_v273_no_contact_pairwise_seed_healed "
            f"channel_names={','.join(BFV3_NO_CONTACT_PAIRWISE_V273_CHANNEL_NAMES)} "
            f"batch_size={self.batch_size} epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch}"
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if (
            not isinstance(target_for_loss, dict)
            or "instance" not in target_for_loss
            or "seed" not in target_for_loss
        ):
            raise ValueError("V273 validation requires instance and seed targets")
        instance = target_for_loss["instance"]
        seed_target = target_for_loss["seed"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance_for_edges = instance[:, 0].long()
        else:
            instance_for_edges = instance.long()
        if seed_target.ndim == 5 and int(seed_target.shape[1]) == 1:
            seed_target = seed_target[:, 0]

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_NO_CONTACT_PAIRWISE_V273_OUTPUT_CHANNELS):
            raise ValueError(f"V273 expected validation logits [B, 28, D, H, W], got {tuple(logits.shape)}")

        support_t = (instance_for_edges > 0) & (instance_for_edges <= MAX_INSTANCE_ID)
        same_edge, different_fragment_edge, _leak_edge, valid_edge = (
            BoundaryFragmentV3Core025StrongPeakNoContactPairwiseSoftmaxSeedHealedV273HeadLoss
            ._same_contact_leak_edges(instance_for_edges.to(device=logits.device))
        )
        seed_t = seed_target.float() > 0.5
        support_p = torch.sigmoid(logits[:, 0].float()) >= 0.5
        seed_p = torch.sigmoid(logits[:, 1].float()) >= 0.5
        edge_diff = logits[:, 2:15].float() - logits[:, 15:28].float()
        edge_cut_p = (edge_diff < 0.0) & valid_edge
        edge_cut_t = different_fragment_edge
        barrier_p = self._affinity13_edge_mask_to_voxel_mask(edge_cut_p)
        barrier_t = self._affinity13_edge_mask_to_voxel_mask(edge_cut_t)
        # [METRIC][Scope:validation_proxy]
        # patch validation은 fulltrain gate가 아니다. 여기서는 support/seed/cut의
        # 최소 sanity만 보고, 승격은 hard3 Task1 proxy와 visual audit에서 결정한다.
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(barrier_p, barrier_t),
            "core_counts": self._binary_counts(seed_p, seed_t),
            "shell_counts": self._binary_counts(barrier_p | seed_p, barrier_t | seed_t),
            "pred_target_barrier": np.asarray([
                float(barrier_p.sum().detach().cpu()),
                float(barrier_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorGapV277(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxSeedHealedV273
):
    """Dataset537 V277: old contact head를 제거한 support + separator-gap contract.

    [AUDIT][Risk:Blocker][Scope:v277_contact_removed]
    V247 이후 contact scalar는 0.50 threshold에서 재현성이 없었고, V275/V276은
    arbitrary local-ID class collapse로 Merge/Topology를 회복하지 못했다. V277은
    output을 2채널로 줄이고, learned field를 connected component decoder가 직접 쓰는
    support와 separator gap으로 제한한다. fulltrain은 Task1 proxy + visual gate 전 금지다.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_no_contact_separator_gap_v277_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = False
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [QC][Invariant:output_contract]
        # V277 output은 정확히 2개 logit이다. channel 0은 foreground support,
        # channel 1은 instance component를 끊는 separator gap이다. old contact,
        # seed, pairwise join/cut row가 섞이면 learned checkpoint를 평가할 수 없다.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_SEPARATOR_GAP_V277_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedV268.__init__(
            self,
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V277_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V277_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V277_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V277_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V277_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V277_BATCH_SIZE", "1").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V277_BATCH_SIZE must be >= 1, got {batch_size}")
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorGapV277] "
            "output_channels=2 changed_variable=remove_contact_predict_support_separator_gap "
            "decoder=task1_v277_separator_gap_cc_min100 "
            f"channel_names={','.join(BFV3_SEPARATOR_GAP_V277_CHANNEL_NAMES)} "
            f"batch_size={self.batch_size} epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch}"
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if not isinstance(target_for_loss, dict) or "instance" not in target_for_loss:
            raise ValueError("V277 validation requires instance targets")
        instance = target_for_loss["instance"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance_for_target = instance[:, 0].long()
        else:
            instance_for_target = instance.long()

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_SEPARATOR_GAP_V277_OUTPUT_CHANNELS):
            raise ValueError(f"V277 expected validation logits [B, 2, D, H, W], got {tuple(logits.shape)}")

        support_t, separator_t = (
            BoundaryFragmentV3Core025StrongPeakNoContactSeparatorGapV277HeadLoss
            ._separator_gap_targets(instance_for_target.to(device=logits.device))
        )
        support_p = torch.sigmoid(logits[:, 0].float()) >= 0.5
        separator_p = (torch.sigmoid(logits[:, 1].float()) >= 0.5) & support_p
        core_p = support_p & ~separator_p
        core_t = support_t & ~separator_t
        # [METRIC][Scope:validation_proxy]
        # patch validation은 support/separator sanity만 본다. 승격 기준은 hard3/true-overfit
        # Task1 proxy와 visual audit이며, old contact recall/ratio는 active metric이 아니다.
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(separator_p, separator_t),
            "core_counts": self._binary_counts(core_p, core_t),
            "shell_counts": self._binary_counts(support_p | separator_p, support_t | separator_t),
            "pred_target_barrier": np.asarray([
                float(separator_p.sum().detach().cpu()),
                float(separator_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorSoftmaxV287(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorGapV277
):
    """Dataset537 V287: contact-free background/body/separator softmax contract.

    [AUDIT][Risk:Blocker][Scope:v287_contact_removed_separator_learning]
    V277/V278 showed that deleting contact is necessary but insufficient:
    support learned, while the independent separator sigmoid collapsed to zero.
    V287 keeps the V277 oracle-pass decoder target and changes only the learned
    head/loss contract to mutually exclusive background/body/separator classes.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_no_contact_separator_softmax_v287_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = False
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [QC][Invariant:output_contract]
        # V287 output is exactly 3-class softmax logits: background/body/separator.
        # Contact, fracture surface, and old independent separator sigmoid are not
        # part of this active path.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_SEPARATOR_SOFTMAX_V287_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V287_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V287_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V287_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V287_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V287_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V287_BATCH_SIZE", "1").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V287_BATCH_SIZE must be >= 1, got {batch_size}")
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorSoftmaxV287] "
            "output_channels=3 changed_variable=contact_removed_background_body_separator_softmax "
            "decoder=task1_v287_separator_softmax_to_v277_cc_min100 "
            f"channel_names={','.join(BFV3_SEPARATOR_SOFTMAX_V287_CHANNEL_NAMES)} "
            f"batch_size={self.batch_size} epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch}"
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if not isinstance(target_for_loss, dict) or "instance" not in target_for_loss:
            raise ValueError("V287 validation requires instance targets")
        instance = target_for_loss["instance"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance_for_target = instance[:, 0].long()
        else:
            instance_for_target = instance.long()

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_SEPARATOR_SOFTMAX_V287_OUTPUT_CHANNELS):
            raise ValueError(f"V287 expected validation logits [B, 3, D, H, W], got {tuple(logits.shape)}")

        support_t, separator_t = (
            BoundaryFragmentV3Core025StrongPeakNoContactSeparatorGapV277HeadLoss
            ._separator_gap_targets(instance_for_target.to(device=logits.device))
        )
        probs = torch.softmax(logits.float(), dim=1)
        support_p = (probs[:, 1] + probs[:, 2]) >= 0.5
        separator_p = (probs[:, 2] >= 0.5) & support_p
        core_p = support_p & ~separator_p
        core_t = support_t & ~separator_t
        # [METRIC][Scope:v287_validation_proxy]
        # barrier slots represent separator-gap class, not contact. Fulltrain
        # promotion still requires full-volume Task1 proxy and visual audit.
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": self._binary_counts(support_p, support_t),
            "barrier_counts": self._binary_counts(separator_p, separator_t),
            "core_counts": self._binary_counts(core_p, core_t),
            "shell_counts": self._binary_counts(support_p | separator_p, support_t | separator_t),
            "pred_target_barrier": np.asarray([
                float(separator_p.sum().detach().cpu()),
                float(separator_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV288(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorSoftmaxV287
):
    """Dataset537 V288: contact-free ABBC 4-class [background, border, boundary, core].

    [AUDIT][Risk:Blocker][Scope:v288_abbc_pengwin_2024_1st_place_aligned]
    PENGWIN 2024 CT 1st place (MIC-DKFZ, IoU-F 0.9296) ABBC representation. V287's
    3-class [background, body, separator] gave no decoder seed class, so even with
    abundant separator voxels the connected-component decoder could not split touching
    fragments. V288 introduces explicit core seed class and adaptive boundary thickness,
    with watershed-style boundary-to-nearest-seed merge at decode time.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v288_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = False
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # [QC][Invariant:output_contract]
        # V288 output is exactly 4-class softmax logits:
        # 0 background, 1 border, 2 boundary, 3 core.
        # Decoder uses core CC -> seeds, boundary watershed-merge to nearest seed.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_ABBC_V288_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        # V287 super().__init__ already populated env-driven epochs/iters/batch
        # from PENGWIN_BFV3_V287_* env vars (or defaults if not set). V288 keeps
        # those defaults but lets PENGWIN_BFV3_V288_* override when present.
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V288_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V288_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V288_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V288_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V288_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V288_BATCH_SIZE", "").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V288_BATCH_SIZE must be >= 1, got {batch_size}")
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV288] "
            "output_channels=4 changed_variable=contact_removed_background_border_boundary_core_softmax "
            "decoder=task1_v288_abbc_core_seed_watershed "
            f"channel_names={','.join(BFV3_ABBC_V288_CHANNEL_NAMES)} "
            f"batch_size={self.batch_size} epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch}"
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = self._move_boundary_fragment_data(data)
        target_for_loss = self._move_boundary_fragment_target(target)
        target_for_loss = self._prepare_boundary_fragment_target(target_for_loss)
        if not isinstance(target_for_loss, dict) or "instance" not in target_for_loss:
            raise ValueError("V288 validation requires instance targets")
        instance = target_for_loss["instance"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance_for_target = instance[:, 0].long()
        else:
            instance_for_target = instance.long()

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target_for_loss)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.ndim != 5 or int(logits.shape[1]) != int(BFV3_ABBC_V288_OUTPUT_CHANNELS):
            raise ValueError(f"V288 expected validation logits [B, 4, D, H, W], got {tuple(logits.shape)}")

        class_target, support_t = (
            BoundaryFragmentV3Core025StrongPeakNoContactABBCV288HeadLoss
            ._abbc_class_target(
                instance_for_target.to(device=logits.device),
                boundary_dilate_vox=2,
                core_erode_vox=2,
            )
        )
        probs = torch.softmax(logits.float(), dim=1)
        background_p = probs[:, 0] >= 0.5
        support_p = ~background_p
        core_p = (probs[:, 3] >= 0.5) & support_p
        boundary_p = (probs[:, 2] >= 0.5) & support_p
        border_p = (probs[:, 1] >= 0.5) & support_p
        core_t = class_target == 3
        boundary_t = class_target == 2
        border_t = class_target == 1
        # [METRIC][Scope:v288_validation_proxy]
        # validation patch metric은 fulltrain promotion 기준이 아니다. Phase 4/5는
        # eval.py Task1 official-aligned proxy로 판단한다. on_validation_epoch_end는
        # V277/V287 시절의 키를 그대로 기대하므로 boundary/support를 V287의
        # barrier/shell 슬롯에 alias한다 (의미적으로 동일 — separator == boundary).
        #
        # [METRIC][FIX:M6][2026-05-31] shell_counts 가 support_counts 와 동일했던
        # copy-paste 버그를 수정. V287 의 shell 의도 (support ∪ separator, 즉 표면
        # 강조) 를 V288 ABBC 4-class 컨트랙트에서는 boundary alone 으로 매핑한다.
        # boundary 가 V287 separator 의 후속 클래스이므로 가장 의미적으로 정합.
        boundary_counts = self._binary_counts(boundary_p, boundary_t)
        support_counts = self._binary_counts(support_p, support_t)
        return {
            "loss": loss.detach().cpu().numpy(),
            "support_counts": support_counts,
            "barrier_counts": boundary_counts,
            "boundary_counts": boundary_counts,
            "core_counts": self._binary_counts(core_p, core_t),
            "border_counts": self._binary_counts(border_p, border_t),
            "shell_counts": boundary_counts,  # FIX:M6 — support_counts 와 동일했던 버그
            "pred_target_barrier": np.asarray([
                float(boundary_p.sum().detach().cpu()),
                float(boundary_t.sum().detach().cpu()),
            ], dtype=np.float64),
            "pred_target_core": np.asarray([
                float(core_p.sum().detach().cpu()),
                float(core_t.sum().detach().cpu()),
            ], dtype=np.float64),
        }




class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV291(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV288
):
    """Dataset537 V291: V288 ABBC 기반 트레이너에 경계(boundary) 클래스 가중치를 CE+Dice 손실에 적용한 버전 (SDF 미사용).

    [AUDIT][Risk:Blocker][Scope:v291_boundary_class_weight]
    V288의 boundary head precision/recall이 약 0.07/0.21에 그치는 이유는
    경계 클래스가 희소(sparse)한데 4-class softmax 손실에서 다른 클래스와
    동일한 가중치로 학습되기 때문이다. V291은 경계 클래스의 CE와 Dice 항을
    5배로 곱해 학습 신호를 강화한다. SDF나 EDT는 사용하지 않으므로
    V288의 처리량(에폭당 약 38초)을 그대로 물려받는다.
    디코더/예측/평가 경로는 모두 V288의 것을 재사용한다.
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v291_head"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "boundary_fragment_v3_sidecar_core_recall_high_edge"
    BOUNDARY_FRAGMENT_USE_STABLE_SCORE = False
    BOUNDARY_FRAGMENT_SAVE_STABLE_CHECKPOINT = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_ABBC_BWEIGHT_V291_OUTPUT_CHANNELS,
            enable_deep_supervision=False,
        )

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_BFV3_V291_EPOCHS", self.num_epochs))
        if os.environ.get("PENGWIN_BFV3_V291_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V291_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_BFV3_V291_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_BFV3_V291_VAL_ITERS"])
        batch_override = os.environ.get("PENGWIN_BFV3_V291_BATCH_SIZE", "").strip()
        if batch_override:
            batch_size = int(batch_override)
            if batch_size < 1:
                raise ValueError(f"PENGWIN_BFV3_V291_BATCH_SIZE must be >= 1, got {batch_size}")
            self.configuration_manager.configuration["batch_size"] = int(batch_size)
            self.batch_size = int(batch_size)
        self._pengwin_loss_profile = os.environ.get(
            "PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE
        ).strip().lower()
        self.print_to_log_file(
            "[BoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV291] "
            "output_channels=4 changed_variable=boundary_class_weight_on_ce_and_dice "
            "decoder=task1_v288_abbc_core_seed_watershed "
            f"channel_names={','.join(BFV3_ABBC_BWEIGHT_V291_CHANNEL_NAMES)} "
            f"batch_size={self.batch_size} epochs={self.num_epochs} "
            f"train_iters={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch}"
        )




# =============================================================================
# [V0.x][PHASE 2A][2026-05-31] BADB: Boundary Attention Decoder Branch
# =============================================================================
#
# V0~V72 ablation 의 종합 교훈: contact precision/recall 트레이드오프는 BICM 5-class
# softmax + factorized head 의 fundamental 한계. V0.x 의 진단 #4 에서 ABBC 의 contact
# 비율이 0.03% 수준으로 sparse class collapse 가 확인됨.
#
# PENGWIN 2024 1위 (MIC-DKFZ) 의 차별점: ABBC + medial axis 기반 dynamic boundary
# thickness. 그러나 architecture 측 inductive bias 없이는 (= V291 처럼 단순 loss
# weight 변경만으로는) 0.03% sparse contact 학습이 어려움.
#
# Phase 1 (Dynamic boundary, utils.compute_abbc_official_target_dynamic): target 측
#   contact ratio 를 0.03% → 1-3% 로 자연 확장.
#
# Phase 2A (본 V300, BADB): network 측 boundary-aware inductive bias 도입. 기존
#   V291 의 4-class ABBC softmax 출력을 base 로 하되, 입력 CT 와 함께 refinement
#   conv block 을 추가하여 boundary 채널을 명시적으로 학습시킨다. base + delta
#   residual 구조이므로 학습 초기에는 V291 와 동일하게 시작, 학습 진행에 따라
#   boundary refinement 가 활성화됨.
# =============================================================================


class _V300BoundaryAttentionRefinementNetwork(nn.Module):
    """V0.x Phase 2A: V291 base + boundary refinement residual block.

    구조:
        x (3-ch input: ct_lut, anatomy prob, sdf)
          ↓
        base_net (nnUNet ResEnc-L, V291 backbone)
          ↓ [B, 4, D, H, W] logits (4-class ABBC)
          ↓
        boundary_refine_conv([logits || ct]):
          Conv3D(5→16) → InstanceNorm → LeakyReLU →
          Conv3D(16→8) → InstanceNorm → LeakyReLU →
          Conv3D(8→1)
          ↓ delta_boundary [B, 1, D, H, W]
          ↓
        refined_logits = logits.clone()
        refined_logits[:, 2] += delta_boundary  (boundary channel = 2)
          ↓
        return refined_logits  (deep supervision tuple 보존)

    Note:
        - 4-class ABBC 채널 레이아웃: [0=background, 1=border, 2=boundary, 3=core]
          (BFV3_ABBC_V288_CHANNEL_NAMES, core.py:330). boundary = channel 2.
        - delta 가 0 으로 초기화되도록 마지막 Conv3D 의 weight/bias 를 0 으로 초기화 →
          학습 초기 V291 와 byte-identical, 학습 진행에 따라 refinement 활성.
        - Channel 2 (boundary) 만 refinement 대상. background/border/core 는 base 그대로.
        - Refinement 입력은 (logits, ct) 5-channel — ct 가 boundary 의 raw signal 제공.
    """

    def __init__(self, base_network: nn.Module, num_output_channels: int = 4,
                 boundary_channel_index: int = 2):
        super().__init__()
        self.base = base_network
        self.boundary_channel_index = int(boundary_channel_index)
        self.num_output_channels = int(num_output_channels)

        # Boundary refinement: 3-layer 3D conv residual block.
        # 입력: (num_output_channels logits) + (1 CT channel) = N+1 channels.
        self.boundary_refine = nn.Sequential(
            nn.Conv3d(self.num_output_channels + 1, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(8, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(8, 1, kernel_size=1),
        )
        # 마지막 Conv3D 의 weight/bias 0 으로 초기화 → delta=0 학습 시작 → V291 byte-identical 초기 상태.
        with torch.no_grad():
            self.boundary_refine[-1].weight.zero_()
            self.boundary_refine[-1].bias.zero_()

    @property
    def decoder(self):
        # [QC][Invariant:nnunet_trainer_api][FIX:DS][2026-06-01]
        # nnUNetTrainer.on_train_start 가 set_deep_supervision_enabled →
        # network.decoder.deep_supervision 를 직접 토글한다. wrapper 가 decoder 를
        # 숨기면 첫 step 전에 AttributeError 로 깨진다. base 의 decoder 를 그대로 노출.
        # (_CoordConvInputWrapper 와 동일한 패턴, core.py:926)
        return self.base.decoder

    @property
    def encoder(self):
        # [QC][Invariant:nnunet_trainer_api][FIX:STUNET][2026-06-01] 일부 nnUNet utility 가
        # encoder attribute 를 전제하지만, STUNet 등 일부 백본은 .encoder 가 없다(.decoder 만 노출).
        # 방어적으로 반환 — 없으면 None (set_deep_supervision_enabled 는 .decoder 만 필요).
        return getattr(self.base, "encoder", None)

    def forward(self, x):
        """x: [B, in_C, D, H, W] (in_C=3 for Ds538: ct_lut+anatprob+sdf)."""
        base_out = self.base(x)
        # nnUNetv2 deep supervision → tuple/list of logits at multiple resolutions.
        if isinstance(base_out, (list, tuple)):
            # Refinement only on the highest resolution output (index 0).
            primary_logits = base_out[0]
            other_outputs = list(base_out[1:])
        else:
            primary_logits = base_out
            other_outputs = []

        # Refinement input: primary_logits + first input channel (CT).
        # x[:, :1] = ct_lut channel (input channel 0).
        refine_in = torch.cat([primary_logits, x[:, :1]], dim=1)
        delta = self.boundary_refine(refine_in)  # [B, 1, D, H, W]

        # Residual add to boundary channel only.
        refined = primary_logits.clone()
        refined[:, self.boundary_channel_index:self.boundary_channel_index + 1] = (
            refined[:, self.boundary_channel_index:self.boundary_channel_index + 1] + delta
        )

        if other_outputs:
            return tuple([refined] + other_outputs)
        return refined


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV300(
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV291
):
    """[V0.x][PHASE 2A][2026-05-31] V300 = V291 + BADB (Boundary Attention Decoder Branch).

    V291 의 4-class ABBC softmax 학습 위에 boundary-aware residual refinement block 을
    추가한 V0.x novelty. V291 의 5x boundary weight 만으로 0.03% sparse contact class
    가 학습되지 않는 문제를 architecture 측 inductive bias 로 해결.

    학습 안정성:
        - Refinement block 의 마지막 conv weight/bias 가 0 으로 초기화 → 학습 초기에는
          V291 와 byte-identical 동작 → V291 의 학습 curve 와 동등하게 출발.
        - 학습 진행하며 boundary refinement 가 활성화 → 후반에 contact 학습 강화.

    예상 효과 (literature 기반):
        - barrier_f0_5: V291 의 0.10-0.18 정체 → 0.25+ 도달 가능
        - barrier_pred_target_ratio: 자가 보정 (refinement 가 sparse class 의 over-prediction 방지)
        - 학습 시간: V291 대비 +5-10% (refinement block 의 small overhead)

    의존성:
        - Dataset538 (4-anatomy BICM V5 with femur) 필요.
        - Phase 1 의 dynamic boundary target 와 조합 시 시너지 기대 (sparse → 1-3% contact + arch bias).

    TODO:
        - [P1] PyTorch DDP 호환 검증 (다중 GPU 학습 시 module wrapping).
        - [P2] Refinement block 의 출력 magnitude clamp (delta 가 너무 크면 학습 unstable).
        - [P3] Boundary attention (multiplicative gating) 추가 옵션 (현재는 additive residual).
    """

    DEFAULT_LOSS_PROFILE = "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v291_head"

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list[str],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # Step 1: nnUNet ResEnc-L base network 구성 (V291 와 동일).
        base = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            BFV3_ABBC_BWEIGHT_V291_OUTPUT_CHANNELS,  # 4-class output
            enable_deep_supervision=False,
        )
        # Step 2: BADB wrapper 로 감싸기 — boundary refinement residual block 추가.
        # [V0.x][FIX:CH][2026-06-01] boundary_channel_index 1→2 수정.
        #   4-class ABBC 채널 레이아웃(BFV3_ABBC_V288_CHANNEL_NAMES, core.py:330)은
        #   [0=background, 1=border, 2=boundary, 3=core] 이다. loss/validation 의
        #   barrier_target = (labels==2), boundary_p = probs[:,2] (loss.py:125, core.py:7537)
        #   도 모두 channel 2 를 boundary 로 정의한다. 이전 값 1 은 "border" 를 가리켜
        #   refinement 가 엉뚱한 클래스를 최적화하고 barrier_f0_5 개선이 불가능했다.
        wrapped = _V300BoundaryAttentionRefinementNetwork(
            base_network=base,
            num_output_channels=BFV3_ABBC_BWEIGHT_V291_OUTPUT_CHANNELS,
            boundary_channel_index=2,  # ABBC: 0=background, 1=border, 2=boundary, 3=core
        )
        return wrapped

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.print_to_log_file(
            "[V300 BADB Phase 2A] V291 base + boundary refinement residual conv "
            "(delta 초기값 0 → V291 byte-identical 학습 시작) "
            f"output_channels={BFV3_ABBC_BWEIGHT_V291_OUTPUT_CHANNELS} "
            f"refinement_block_params="
            f"Conv3d(5→16)+InstanceNorm+LReLU+Conv3d(16→8)+InstanceNorm+LReLU+Conv3d(8→1)"
        )


# =============================================================================
# (위는 V300 BADB Phase 2A skeleton 종료)
# =============================================================================


# =============================================================================
# [V0.x][BACKBONE][2026-06-01] STU-Net-B 백본 trainer (anatomy + ABBC fracture)
# =============================================================================
def _build_stunet_from_plan(variant: str, arch_init_kwargs: dict,
                            num_input_channels: int, num_output_channels: int,
                            enable_deep_supervision: bool = True):
    """nnU-Net 2.5.2 plan 의 arch_init_kwargs 에서 strides 를 뽑아 STUNet 을 구성한다.

    STUNet 원본은 configuration_manager.pool_op_kernel_sizes[1:] 를 strides 로 썼다.
    2.5.2 에서는 동일 값이 arch_init_kwargs['strides'] 에 들어있다(첫 entry=[1,1,1]=stage0
    no-downsample). [1:] 로 다운샘플 strides 5개를 취하고, 6-stage(dims 6개)에 맞춰
    cap(5)/pad([1,1,1]) 한다. conv 가중치는 stride 무관 shape 이므로 사전학습 가중치가
    그대로 로드된다.
    """
    if not _STUNET_AVAILABLE:
        raise RuntimeError("STUNet 백본 import 실패 — stunet.py 확인 필요")
    strides = [list(s) for s in arch_init_kwargs["strides"][1:]]
    if len(strides) > 5:
        strides = strides[:5]
    while len(strides) < 5:
        strides.append([1, 1, 1])
    kernel_sizes = [[3, 3, 3]] * 6
    v = STUNET_VARIANTS[variant]
    return STUNet(num_input_channels, num_output_channels,
                  depth=v["depth"], dims=v["dims"],
                  pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                  enable_deep_supervision=enable_deep_supervision)


def _maybe_apply_stunet_warmstart(trainer):
    """[V0.x][FIX:DDP][2026-06-01] env `PENGWIN_STUNET_PRETRAINED` 의 STU-Net 사전학습
    가중치를 trainer.network 에 warm-start 한다.

    DDP-safe: nnUNet 의 `-pretrained_weights` 경로는 단일 GPU(부모 프로세스)에서만
    monkey-patch 가능하다 — `num_gpus>1` 이면 `mp.spawn(run_ddp)` 로 뜨는 자식 프로세스가
    run_training 을 fresh import 하므로 부모의 patch 가 적용되지 않아 STUNet 기본 로더가
    깨진다. 본 훅은 각 프로세스의 initialize() 끝에서 직접 적용하므로 DDP child 에서도
    동작한다. 따라서 STU-Net warm-start 는 `-pretrained_weights` 대신
    `PENGWIN_STUNET_PRETRAINED` 환경변수를 사용한다(둘 다 주면 이중 적용되니 금지).

    모든 rank 가 동일 파일을 로드 → 가중치 일관(DDP 시작 불변식 유지). 로더는 DDP/compile
    래퍼를 자동 언랩한다. continue(-c) 체크포인트는 initialize() 이후 별도로 로드되어
    warm-start 를 덮어쓰므로(재개 시 올바름) 상호 안전하다.
    """
    import os as _os
    path = _os.environ.get("PENGWIN_STUNET_PRETRAINED", "").strip()
    if not path or getattr(trainer, "_stunet_warmstart_done", False):
        return
    if not _STUNET_AVAILABLE:
        trainer.print_to_log_file("[STU-Net warm-start] stunet 모듈 없음 — 스킵")
        return
    from stunet import load_stunet_pretrained_weights
    inflate = _os.environ.get("PENGWIN_STUNET_INFLATE", "ct0")
    stats = load_stunet_pretrained_weights(trainer.network, path, inflate=inflate)
    trainer._stunet_warmstart_done = True
    trainer.print_to_log_file(f"[STU-Net warm-start] {path}: {stats}")


# =============================================================================
# [V0.x][PARTIAL-LABEL][2026-06-02] Marginal Dice+CE loss (Shi et al., MedIA 2021)
# =============================================================================
class MarginalDiceCELoss(nn.Module):
    """Partial-label marginal Dice+CE — 부분 라벨 충돌 supervision 을 loss 레벨에서 근본 해결.

    조사 B 결론: Ds539 는 pelvic(=sacrum/LHip/RHip 만 라벨) 과 femur(=femur 만 라벨) 의 disjoint
    부분 라벨이라, 보이는데 미라벨된 뼈가 'background' 로 학습돼 충돌(같은 뼈가 한 케이스에선
    sacrum, 다른 케이스에선 background). Marginal loss 는 케이스별 '미라벨 foreground 클래스' 를
    **background marginal 로 접어** 페널티를 제거한다.

    - labeled_mask[B, C] (bool): 케이스별 라벨된 클래스. c=0(bg) 은 항상 labeled 취급.
      트레이너가 매 배치 batch['keys'] → case 라벨셋으로 설정한다(None 이면 전부 labeled = 표준).
    - CE: 미라벨 fg 클래스 logit 을 {bg}∪{미라벨} logsumexp 로 묶어 marginal bg log-prob 계산
      → bg 영역(미라벨 뼈 포함)에서 미라벨 클래스 예측에 페널티 0. labeled 클래스는 표준 log-softmax.
    - Dice(do_bg=False, batch_dice=False): labeled fg 클래스만 per-sample soft dice, 미라벨 제외.
    - softmax 유지 → STU-Net TotalSeg warm-start 보존. nnUNet DC_and_CE 관례(weight 1/1, smooth 1e-5).

    DeepSupervisionWrapper 와 호환: labeled_mask 는 클래스 단위(해상도 무관)이므로 모든 DS scale 에
    동일 적용된다.
    """

    def __init__(self, num_classes: int, batch_dice: bool = False, smooth: float = 1e-5,
                 weight_ce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.batch_dice = bool(batch_dice)
        self.smooth = float(smooth)
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.labeled_mask = None  # [B, C] bool — 트레이너가 배치마다 설정

    def forward(self, logits, target):
        B, C = logits.shape[:2]
        if target.shape[1] != 1:
            target = target[:, :1]
        tgt = target.long()
        dev = logits.device
        mask = self.labeled_mask
        if mask is None:
            mask = torch.ones(B, C, dtype=torch.bool, device=dev)
        else:
            mask = mask.to(dev).bool()
            if mask.shape[0] != B:  # DDP/grad accum 등으로 B 불일치 시 안전 폴백
                mask = torch.ones(B, C, dtype=torch.bool, device=dev)
        mask = mask.clone()
        mask[:, 0] = True  # bg 항상 labeled

        # bg-supergroup = {0} ∪ {미라벨 fg}
        ch = torch.arange(C, device=dev)[None, :]              # [1, C]
        fg_unlabeled = (~mask) & (ch != 0)                     # [B, C]
        bg_group = fg_unlabeled.clone()
        bg_group[:, 0] = True                                  # [B, C]

        # ---- marginal CE ----
        logp = torch.log_softmax(logits, dim=1)               # [B, C, ...]
        lse_all = torch.logsumexp(logits, dim=1, keepdim=True) # [B, 1, ...]
        neg_inf = torch.finfo(logits.dtype).min
        bgmask = bg_group.view(B, C, *([1] * (logits.ndim - 2)))
        masked = logits.masked_fill(~bgmask, neg_inf)
        bg_lse = torch.logsumexp(masked, dim=1, keepdim=True)  # [B, 1, ...]
        logq_bg = bg_lse - lse_all                             # [B, 1, ...] log marginal bg
        gathered = torch.gather(logp, 1, tgt.clamp(0, C - 1))  # [B, 1, ...]
        target_logq = torch.where(tgt == 0, logq_bg, gathered)
        ce = -(target_logq).mean()

        # ---- marginal Dice (do_bg=False, labeled fg only) ----
        p = torch.exp(logp)
        spatial = tuple(range(2, logits.ndim))
        dice_vals = []
        for c in range(1, C):
            sel = mask[:, c]                                   # [B]
            if not bool(sel.any()):
                continue
            pc = p[:, c]                                       # [B, ...]
            gc = (tgt[:, 0] == c).to(pc.dtype)                 # [B, ...]
            sdims = tuple(range(1, pc.ndim))
            inter = (pc * gc).sum(sdims)
            denom = pc.sum(sdims) + gc.sum(sdims)
            dpc = (2 * inter + self.smooth) / (denom + self.smooth)  # [B]
            dice_vals.append(dpc[sel])
        if dice_vals:
            mean_dice = torch.cat(dice_vals).mean()
        else:
            mean_dice = logits.sum() * 0.0
        return self.weight_ce * ce - self.weight_dice * mean_dice


# =============================================================================
# [V0.x][WARN-FIX][2026-06-02] nnUNet 2.5.2 deprecated-API 경고 근본 수정 mixin
# =============================================================================
def _register_numpy_safe_globals():
    """nnUNet 체크포인트(.pth) 의 numpy 메타데이터를 weights_only=True 로 안전 로드하기 위한
    allowlist (임의코드 실행과 무관한 numpy 재구성 global 만)."""
    try:
        import numpy as _np
        import torch.serialization as _ts
        g = [_np.ndarray, _np.dtype, _np.core.multiarray.scalar, _np.core.multiarray._reconstruct]
        try:
            import numpy.dtypes as _nd
            g += [getattr(_nd, _n) for _n in dir(_nd) if _n.endswith("DType")]
        except Exception:
            pass
        _ts.add_safe_globals(g)
    except Exception:
        pass


class _PengwinPolyLR:
    """torch `_LRScheduler` 를 상속하지 않는 PolyLR — nnUNet PolyLRScheduler 와 수식 동일하나
    torch 의 step-순서/epoch-인자 deprecation 경고를 발생시키지 않는다.

    nnUNet 의 lr_scheduler contract 는 (a) step(current_epoch) 호출(on_train_epoch_start),
    (b) param_groups['lr'] 갱신, (c) checkpoint 에 저장 안 함(current_epoch 로 재계산) 뿐이라
    plain 객체로 충분하다."""

    def __init__(self, optimizer, initial_lr, max_steps, exponent=0.9):
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.max_steps = int(max_steps)
        self.exponent = float(exponent)
        self.ctr = 0
        self.step(0)

    def step(self, current_step=None):
        if current_step is None:
            current_step = self.ctr
            self.ctr += 1
        frac = max(0.0, 1.0 - current_step / max(1, self.max_steps))
        new_lr = self.initial_lr * (frac ** self.exponent)
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr


class _StunetCleanTrainerMixin:
    """STU-Net trainer 공용 mixin. nnUNet 2.5.2 의 deprecated API 사용으로 뜨는 경고를
    modern API 로 **근본 수정**한다(억제 아님). 모든 STU-Net trainer 가 이 mixin 을 가장 먼저
    상속하여 아래 오버라이드가 MRO 상 우선 적용된다.

    수정 대상:
      #4 torch.compile+batch1+deep-supervision → `_do_i_compile()=False` 로 compile 비활성.
         validation 시 DS 토글에 compile 이 재컴파일 안 되어 예측이 손상되던 문제까지 근본 해결.
      #3 torch.cuda.amp.GradScaler deprecated → `torch.amp.GradScaler('cuda')` 로 교체.
      #1/#2/#5 lr_scheduler.step(epoch)/step-order deprecation → `_LRScheduler` 비상속
         `_PengwinPolyLR` 로 교체(on_train_epoch_start 의 step 호출이 우리 객체로 감).
      #6 torch.load(weights_only=False) FutureWarning → weights_only=True(+numpy allowlist) 안전 로드.
    """

    def _do_i_compile(self):  # #4
        return False

    def initialize(self):
        super().initialize()  # #3 GradScaler 는 모듈 레벨 patch 가 생성 시점에 신 API 로 처리
        _maybe_apply_stunet_warmstart(self)  # DDP-safe warm-start (env PENGWIN_STUNET_PRETRAINED)
        # [V0.x][FIX:DDP_BADB][2026-06-04] V300 BADB 는 output=base+delta·refinement 를
        # delta 초기값 0 으로 시작한다(V291 byte-identical). delta=0 이면 refinement conv
        # 파라미터가 첫 iteration 에 grad 0 → DDP 의 unused-parameter 검사가 실패한다
        # ("parameters that were not used in producing loss"). delta 가 움직이기 시작하면
        # 이후 grad 가 흐르므로, find_unused_parameters=True 로 re-wrap 해 학습을 허용한다.
        # 같은 .module 을 재포장하므로 파라미터 텐서 동일성이 유지되어 optimizer 는 무영향.
        if getattr(self, "is_ddp", False):
            from torch.nn.parallel import DistributedDataParallel as _DDP
            if isinstance(self.network, _DDP):
                self.network = _DDP(
                    self.network.module,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                )

    def configure_optimizers(self):  # #1/#2/#5
        optimizer, _orig_sched = super().configure_optimizers()
        self.lr_scheduler = _PengwinPolyLR(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, self.lr_scheduler

    def load_checkpoint(self, filename_or_checkpoint):  # #6
        if isinstance(filename_or_checkpoint, str):
            _register_numpy_safe_globals()
            _orig_load = torch.load

            def _safe_load(f, *a, **k):
                k.setdefault("weights_only", True)
                return _orig_load(f, *a, **k)

            torch.load = _safe_load
            try:
                return super().load_checkpoint(filename_or_checkpoint)
            finally:
                torch.load = _orig_load
        return super().load_checkpoint(filename_or_checkpoint)


class PengwinTrainerSTUNetBaseAnatomyV301(_StunetCleanTrainerMixin, PengwinTrainer):
    """[V0.x][BACKBONE] Ds539 anatomy 5-class 용 STU-Net-B 백본 trainer.

    ResEnc-L → STU-Net-B(58.26M) 교체. TotalSegmentator(59개 뼈: sacrum/hip/femur 포함)
    사전학습 warm-start 전제. do_split(grouped split)·loss(DC+CE)·env profile 등
    PengwinTrainer 동작은 그대로 유지하고 네트워크만 STU-Net 으로 바꾼다.
    fine-tune 레시피: lr 1e-3 (STU-Net _ft 권장), epochs 1000.
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000
        # [V0.x][ES][2026-06-01] warm-start 용 early-stop 재튜닝.
        # 상속 기본값(PengwinTrainer: MIN_EPOCHS=100, PATIENCE=50)은 from-scratch 기준이다.
        # STU-Net 뼈 사전학습 warm-start 는 수십 epoch 안에 수렴하므로, 수렴 후 낭비 학습을
        # 막도록 ES 를 낮춘다(MIN_EPOCHS=30 부터 EMA-dice 정체 PATIENCE=25 epoch 시 종료).
        # 상속된 while-loop run_training + on_validation_epoch_end 의 ES 로직이 그대로 적용되며,
        # checkpoint_best 는 상시 저장되므로 종료 시점 최적 가중치가 보존된다. env 로 튜닝 가능.
        import os as _os_es
        self.ES_MIN_EPOCHS = int(_os_es.environ.get("PENGWIN_ES_MIN_EPOCHS", "30"))
        self.ES_PATIENCE = int(_os_es.environ.get("PENGWIN_ES_PATIENCE", "25"))
        self.ES_MIN_DELTA = float(_os_es.environ.get("PENGWIN_ES_MIN_DELTA", "5e-3"))
        # [V0.x][PARTIAL-LABEL][2026-06-02] marginal loss용 case→labeled-class 맵 로드.
        self._marginal_loss = None
        self._case_labeled_map = {}
        self._load_case_labeled_map()

    def _load_case_labeled_map(self):
        """case_id → 라벨된 fg 클래스 리스트. pelvic 케이스={1,2,3}, femur 케이스={4} (disjoint).
        GT 존재 클래스로 사전 산출된 json(result/reports/ds539_case_labeled_classes.json) 로드."""
        import json as _json
        path = os.environ.get("PENGWIN_CASE_LABELED_MAP",
                              str(RESULT_REPORT / "ds539_case_labeled_classes.json"))
        try:
            raw = _json.load(open(path))
            self._case_labeled_map = {str(k): [int(x) for x in v] for k, v in raw.items()}
            self.print_to_log_file(
                f"[marginal] case→labeled 맵 로드: {len(self._case_labeled_map)} cases ({path})")
        except Exception as e:  # pragma: no cover
            self._case_labeled_map = {}
            self.print_to_log_file(f"[marginal] 맵 로드 실패({e}) → 전부 labeled 폴백(표준 동작)")

    def _set_marginal_mask(self, keys):
        """batch['keys'](case 식별자) → per-sample labeled_mask[B,C] 를 marginal loss 에 설정."""
        if self._marginal_loss is None or keys is None:
            return
        ncls = self.label_manager.num_segmentation_heads
        B = len(keys)
        mask = torch.zeros(B, ncls, dtype=torch.bool, device=self.device)
        mask[:, 0] = True  # bg 항상 labeled
        for b, key in enumerate(keys):
            cid = str(key).replace("PENGWIN_", "").split("_")[0].zfill(3)
            labeled = self._case_labeled_map.get(cid)
            if not labeled:
                mask[b, :] = True  # 미상 → 전부 labeled(안전 폴백)
            else:
                for c in labeled:
                    if 0 <= int(c) < ncls:
                        mask[b, int(c)] = True
        self._marginal_loss.labeled_mask = mask

    def _build_loss(self):
        """표준 DC+CE 대신 Marginal Dice+CE(부분 라벨 충돌 근본 해결). DS wrapper 는
        nnUNet 관례 그대로(가중치 1/2^i, 최저 scale 0, DDP+no-compile 시 1e-6)."""
        import numpy as _np
        ncls = self.label_manager.num_segmentation_heads
        marg = MarginalDiceCELoss(
            num_classes=ncls,
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5, weight_ce=1.0, weight_dice=1.0)
        self._marginal_loss = marg
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = _np.array([1 / (2 ** i) for i in range(len(scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            return DeepSupervisionWrapper(marg, weights)
        return marg

    def train_step(self, batch: dict) -> dict:
        self._set_marginal_mask(batch.get("keys"))
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        # marginal mask 설정(val loss 일관) + 미라벨 예측을 bg 로 접어 메트릭을 marginal-aware 로.
        self._set_marginal_mask(batch.get("keys"))
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg
        # [MARGINAL] 미라벨 클래스 예측을 bg(0) 로 이동 → 그 클래스는 라벨된 케이스에서만 평가됨
        lm = getattr(self._marginal_loss, "labeled_mask", None)
        if lm is not None:
            lm = lm.to(predicted_segmentation_onehot.device)
            B, C = predicted_segmentation_onehot.shape[:2]
            for c in range(1, C):
                unl = ~lm[:, c]
                if bool(unl.any()):
                    sl = predicted_segmentation_onehot[:, c]
                    moved = sl * unl.view(B, *([1] * (sl.ndim - 1))).to(sl.dtype)
                    predicted_segmentation_onehot[:, 0] += moved
                    predicted_segmentation_onehot[:, c] -= moved
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=None)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard,
                "fp_hard": fp_hard, "fn_hard": fn_hard}

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = True):
        return _build_stunet_from_plan("base", arch_init_kwargs, num_input_channels,
                                       num_output_channels,
                                       enable_deep_supervision=enable_deep_supervision)


class PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCSTUNetBV301(
    _StunetCleanTrainerMixin,
    PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV300
):
    """[V0.x][BACKBONE] Ds538 ABBC fracture 용 STU-Net-B 백본 + BADB(V300) trainer.

    V300(ResEnc base + BADB boundary refinement)의 백본만 STU-Net-B 로 교체한다.
    ABBC 4-class loss/decode/BFv3 sidecar dataloader/validation/BADB(boundary ch2 refine)
    는 V300 에서 그대로 상속. 입력 3ch(ct_lut+anatprob+sdf) → warm-start 시 stem inflate.
    fine-tune 레시피: lr 1e-3, epochs 1000.
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000
        # [V0.x][ES][2026-06-01] warm-start 용 early-stop 재튜닝.
        # 상속 기본값(PengwinTrainer: MIN_EPOCHS=100, PATIENCE=50)은 from-scratch 기준이다.
        # STU-Net 뼈 사전학습 warm-start 는 수십 epoch 안에 수렴하므로, 수렴 후 낭비 학습을
        # 막도록 ES 를 낮춘다(MIN_EPOCHS=30 부터 EMA-dice 정체 PATIENCE=25 epoch 시 종료).
        # 상속된 while-loop run_training + on_validation_epoch_end 의 ES 로직이 그대로 적용되며,
        # checkpoint_best 는 상시 저장되므로 종료 시점 최적 가중치가 보존된다. env 로 튜닝 가능.
        import os as _os_es
        self.ES_MIN_EPOCHS = int(_os_es.environ.get("PENGWIN_ES_MIN_EPOCHS", "30"))
        self.ES_PATIENCE = int(_os_es.environ.get("PENGWIN_ES_PATIENCE", "25"))
        self.ES_MIN_DELTA = float(_os_es.environ.get("PENGWIN_ES_MIN_DELTA", "5e-3"))

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = True):
        # 1) STU-Net-B base (4-class ABBC logits, V291 와 동일하게 DS off)
        base = _build_stunet_from_plan(
            "base", arch_init_kwargs, num_input_channels,
            BFV3_ABBC_BWEIGHT_V291_OUTPUT_CHANNELS, enable_deep_supervision=False)
        # 2) BADB 래퍼 (boundary channel=2 refine, zero-init → 초기 base byte-identical)
        wrapped = _V300BoundaryAttentionRefinementNetwork(
            base_network=base,
            num_output_channels=BFV3_ABBC_BWEIGHT_V291_OUTPUT_CHANNELS,
            boundary_channel_index=2,  # ABBC: 0=background, 1=border, 2=boundary, 3=core
        )
        return wrapped



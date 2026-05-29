#!/usr/bin/env python3
"""PENGWIN 2026 Task 1 — Training orchestrator.

Subcommands:
    train        Train active nnU-Net datasets (wraps nnUNetv2_train CLI)

Usage:
    python train.py train 532 --gpu 0                     # PelvicAnatomyV2
    python train.py train 533 --gpu 1                     # FemurAnatomyV2
    python train.py train 537 --gpu 0 --trainer PengwinTrainer

Hard-mining/self-training orchestration was removed from the active workflow.
"""
from __future__ import annotations
import argparse, ast, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core import (
    DATASETS, NN_RES, RESULT_DATE, RESULT_REPORT, RESULT_WEIGHT,
    configure_nnunet_env, get_logger,
)
configure_nnunet_env()
log = get_logger(__name__)


def _nnunet_train_bin() -> str:
    """Resolve nnU-Net training CLI in the current runtime environment."""
    explicit = os.environ.get("PENGWIN_NNUNET_TRAIN_BIN", "").strip()
    if explicit:
        return explicit
    found = shutil.which("nnUNetv2_train")
    if found:
        return found
    legacy = "/root/miniconda3/envs/pengwin_v2/bin/nnUNetv2_train"
    if Path(legacy).exists():
        return legacy
    raise FileNotFoundError(
        "nnUNetv2_train not found. Install nnunetv2 or set PENGWIN_NNUNET_TRAIN_BIN."
    )


def _write_json(path: Path, payload) -> None:
    """Write indented JSON and create the parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


PRETRAINED_DEFAULT = str(RESULT_WEIGHT / "pelvic_s1_swa.pth")
PELVIC_FOUNDATION_DEFAULT = (
    NN_RES / "Dataset532_PelvicAnatomyV2"
    / "PengwinTrainer__nnUNetResEncUNetLPlans__3d_fullres"
    / "fold_0" / "checkpoint_best.pth"
)
PELVIC_FOUNDATION_WEIGHTS_ONLY = RESULT_WEIGHT / "pretrained_ds532_fold0_network_weights_only.pth"
NEED_INIT_FROM_PELVIC: set[int] = set()
DEFAULT_FOLD0_ORDER = [532, 533]

# Speed best practices for nnUNet training (reduces bottleneck on dual-GPU).
# Applied automatically in train(); can be overridden via extra_env.
SPEED_ENV = {
    "nnUNet_compile": "t",                  # torch.compile (~10-20% speedup on TITAN RTX)
    "nnUNet_n_proc_DA": "16",               # data augmentation workers (default 12)
    "OMP_NUM_THREADS": "1",                 # prevent CPU oversubscription (per worker)
    "MKL_NUM_THREADS": "1",                 # same for MKL
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # better memory fragmentation
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",   # cuBLAS workspace (deterministic + faster)
}

# Stable profile trades throughput for reliability by disabling torch.compile
# and lowering augmentation workers. It is still useful for v2 fold resumes
# on memory-constrained GPUs.
STABLE_ENV = {
    "nnUNet_compile": "f",
    "nnUNet_n_proc_DA": "8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

# Single-worker profile for tiny overfit/debug runs. nnU-Net unpacks .npz
# files into .npy sidecars before the training dataloader starts. On very
# small rebuilt datasets, multi-process augmentation can race with a stale or
# partial unpacked file and surface as numpy memmap length errors. Keep this
# profile explicit so V5/V5.1 one-case gates can be reproduced without changing
# the normal `speed`/`stable` production defaults.
SINGLE_ENV = {
    "nnUNet_compile": "f",
    "nnUNet_n_proc_DA": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

ENV_PROFILES = {
    "speed": SPEED_ENV,
    "stable": STABLE_ENV,
    "single": SINGLE_ENV,
}

LOSS_PROFILES = [
    "dc_ce",
    "bd_dou_005",
    "bd_dou_01",
    "bd_dou_03",
    "tversky_07",
    "tversky_08",
    "combo_tversky_bd005",
    "bicm_v5_sparse",
    "bicm_v5_support_geometry",
    "bicm_v62_semantic_boundary",
    "bicm_v64_topology_precision",
    "bicm_v65_balanced_contact_core",
    "bicm_v67_semantic_edge_pair",
    "bicm_v6_factorized",
    "bicm_v250_oracle_aligned_direct",
    "bicm_v6_precision",
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
    "bicm_v160_core_only_center_seed_offset_sampler",
    "bicm_v161_core_heatmap_center_seed_offset_sampler",
    "boundary_fragment_v3",
    "boundary_fragment_v3_core_ridge",
    "boundary_fragment_v3_core_ridge_recall",
    "boundary_fragment_v3_core_ridge_recall_masscap",
    "boundary_fragment_v3_core_ridge_recall_pos_masscap",
    "boundary_fragment_v3_core_ridge_recall_soft_pos_masscap",
    "boundary_fragment_v3_core_ridge_recall_very_soft_pos_masscap",
    "boundary_fragment_v3_core_ridge_recall_logit_calibrated",
    "boundary_fragment_v3_core_ridge_recall_precision_logit_calibrated",
    "boundary_fragment_v3_core_ridge_recall_shell_contrast_logit_calibrated",
    "boundary_fragment_v3_core_ridge_recall_strong_shell_contrast_logit_calibrated",
    "boundary_fragment_v3_core_ridge_recall_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_precision_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_positive_floor_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_positive_floor_masscap_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_masscap_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_logit_ridge_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_logit_ridge_precision_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_soft_ridge_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_soft_ridge_contrast_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_barrier_contrast_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_seed_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_sparse_seed_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_compact_seed_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_compact_sparse_seed_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_recall_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_coverage_open_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_coverage_guarded_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_balanced_contact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_fragment_peak_core_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_weak_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_balanced_contact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_tight_balanced_contact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_soft_contact_ridge_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_precision_soft_contact_ridge_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_contact_shell_contrast_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_strong_contact_shell_contrast_barrier_head",
    "boundary_fragment_v3_core025_strong_peak_decoder_contact_head",
    "boundary_fragment_v3_core025_strong_peak_xyz_affinity_head",
    "boundary_fragment_v3_core025_strong_peak_affinity13_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_affinity13_contact_hard_negative_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_seed_stable_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_seed_peak_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_seed_peak_body_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_fracture_seed_calibrated_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_pairwise_softmax_mutex13_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_mutex13_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_seed_healed_v273_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_seed_calibrated_v274_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_pairwise_v284_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_fragment_position_v275_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_fragment_position_dice_v276_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_separator_gap_v277_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_separator_energy_v278_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_separator_softmax_v287_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v288_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_sdf_v289_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_sdf_fdm_v290_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v291_head",
    "boundary_fragment_v3_core025_strong_peak_center_flow_head",
    "boundary_fragment_v3_core025_strong_peak_center_peak_flow_head",
    "boundary_fragment_v3_core025_strong_peak_center_peak_flow_calibrated_head",
    "boundary_fragment_v3_core025_strong_peak_dense_center_heatmap_flow_calibrated_head",
    "boundary_fragment_v3_core025_strong_peak_topology_constrained_center_heatmap_flow_calibrated_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_center_heatmap_flow_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_spatial_embedding_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_spatial_embedding_contrastive_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_coord_spatial_embedding_v279_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_query_mask_v280_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_query_mask_pn_v281_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_query_mask_v285_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_query_mask_pn_v286_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_free_embedding_v282_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_free_embedding_v283_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_contact_preserve_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_coverage_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_original_contact_mid_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_contact_cap_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_tight_contact_cap_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_support_seed_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_conservative_coretail_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_weak_fragment_mass_core_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_loose_fragment_mass_core_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_instance_core_guard_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_instance_core_strong_guard_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_masscap_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_weak_masscap_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_soft_dense_candidate_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_soft_dense_candidate_barrier_contrast_head",
    "boundary_fragment_v3_core_ridge_recall_masscap_tight",
    "boundary_fragment_v3_core_seed",
    "boundary_fragment_v3_ridgefit_precision",
    "factorized_instance_v4_embedding_v163",
    "bicm_v6_support_core",
    "bicm_v6_support_precision",
    "bicm_v6_support_surface",
]

CE_WEIGHT_PROFILES = ["off", "auto"]
OVERSAMPLE_PROFILES = [
    "default",
    "weak0123",
    "bicm_v5_sparse",
    "bicm_v6_support_mixed",
    "bicm_v7_contact_mixed",
    "bicm_v7_contact15_mixed",
    "bicm_v11_contact_roi",
    "bicm_v18_roi_balanced",
    "boundary_fragment_v3_v5_remap",
    "boundary_fragment_v3_sidecar",
    "boundary_fragment_v3_sidecar_core_recall_high_edge",
    "boundary_fragment_v3_sidecar_core_ridge_high_edge",
    "boundary_fragment_v3_sidecar_core_ridge",
    "boundary_fragment_v3_sidecar_core_seed",
    "boundary_fragment_v3_sidecar_precision",
    "factorized_instance_v4_v163",
]
TRAINERS = [
    "PengwinTrainer",
    "PengwinTrainerBICMFactorizedV6",
    "PengwinTrainerBICMOracleAlignedDirectV250",
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
    "PengwinTrainerBICMSemanticCandidateV66",
    "PengwinTrainerBICMSemanticEdgePairV67",
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
    "PengwinTrainerBICMJointSupportProductV111",
    "PengwinTrainerBICMEdgeGraphAssignmentV112",
    "PengwinTrainerBICMAdaptiveBoundaryProductV113",
    "PengwinTrainerBICMAdaptiveBoundaryProductV113HighLR",
    "PengwinTrainerBICMEncoderAdapterV114",
    "PengwinTrainerBICMSemanticOracleAdapterV115",
    "PengwinTrainerBICMGraphCostSeparatorV116",
    "PengwinTrainerBICMSupportGateSemanticContactV117",
    "PengwinTrainerBICMWarmSupportGateSemanticContactV118",
    "PengwinTrainerBICMAllNetworkAdaptiveBoundaryV119",
    "PengwinTrainerBICMSameFragmentAffinityV120",
    "PengwinTrainerBICMWarmSameFragmentAffinityV121",
    "PengwinTrainerBICMWarmEdgeProductPrecisionV122",
    "PengwinTrainerBICMHighRecallGateCleanupV123",
    "PengwinTrainerBICMSupportConditionedEdgePrecisionV124",
    "PengwinTrainerBICMSaturatedEdgeDenseGateV125",
    "PengwinTrainerBICMLocalAdjacencyProductV126",
    "PengwinTrainerBICMLocalAdjacencyProductV126FromV96",
    "PengwinTrainerBICMEdgeResetLocalAdjacencyProductV127",
    "PengwinTrainerBICMSupportBridgeSuppressionV128",
    "PengwinTrainerBICMAffinitySharpeningV129",
    "PengwinTrainerBICMSupportTopologyRepairV130",
    "PengwinTrainerBICMAllNetworkSupportTopologyRepairV131",
    "PengwinTrainerBICMSupportVetoGateV132",
    "PengwinTrainerBICMSupportVetoSemanticV133",
    "PengwinTrainerBICMSupportVetoGateHighLRV134",
    "PengwinTrainerBICMSupportVetoGateUltraLRV135",
    "PengwinTrainerBICMSupportTopologyHighLRV136",
    "PengwinTrainerBICMSupportTopologyUltraLRV137",
    "PengwinTrainerBICMSupportTopologyMidLRV138",
    "PengwinTrainerBICMSupportTopologyUpperMidLRV139",
    "PengwinTrainerBICMSupportTopologyLowBracketLRV140",
    "PengwinTrainerBICMSupportTopologyHighBracketLRV141",
    "PengwinTrainerBICMSupportTopologyCoreSeedLowLRV142",
    "PengwinTrainerBICMSupportTopologyCoreSeedMidLRV143",
    "PengwinTrainerBICMStrongCoreSeedLowLRV144",
    "PengwinTrainerBICMStrongCoreSeedMidLRV145",
    "PengwinTrainerBICMFragmentSeedPresenceLowLRV146",
    "PengwinTrainerBICMFragmentSeedPresenceMidLRV147",
    "PengwinTrainerBICMCoreHeatmapSeedLowLRV148",
    "PengwinTrainerBICMCoreHeatmapSeedMidLRV149",
    "PengwinTrainerBICMCoreOnlyHeatmapSeedMidLRV150",
    "PengwinTrainerBICMCoreOnlyHeatmapSeedHighLRV151",
    "PengwinTrainerBICMCoreOnlyCenterSeedMidLRV152",
    "PengwinTrainerBICMCoreHeatmapCenterSeedMidLRV153",
    "PengwinTrainerBICMCoreOnlyCenterSeedSamplerV154",
    "PengwinTrainerBICMCoreHeatmapCenterSeedSamplerV155",
    "PengwinTrainerBICMCoreOnlyCenterSeedAffinitySamplerV158",
    "PengwinTrainerBICMCoreHeatmapCenterSeedAffinitySamplerV159",
    "PengwinTrainerBICMCoreOnlyCenterSeedOffsetSamplerV160",
    "PengwinTrainerBICMCoreHeatmapCenterSeedOffsetSamplerV161",
    "PengwinTrainerBICMBoundaryFragmentRemapV162",
    "PengwinTrainerFactorizedInstanceEmbeddingV163",
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
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakDecoderContactV247",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakXYZAffinityV248",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakAffinity13SeedHealedV253",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakAffinity13ContactHardNegativeSeedHealedV254",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedHealedV255",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SupportLeakGuardSeedHealedV256",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedStableSupportLeakGuardSeedHealedV257",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedPeakSupportLeakGuardSeedHealedV258",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedPeakBodySupportLeakGuardSeedHealedV259",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13FractureSeedCalibratedSupportLeakGuardSeedHealedV260",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakPairwiseSoftmaxMutex13SeedHealedV266",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedV268",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxSeedHealedV273",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxSeedCalibratedV274",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactGlobalCoordPairwiseV284",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactFragmentPositionV275",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactFragmentPositionDiceV276",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorGapV277",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorEnergyV278",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorSoftmaxV287",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV288",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCSDFV289",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCSDFFDMV290",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV291",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakCenterFlowV261",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakCenterPeakFlowV262",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakCenterPeakFlowCalibratedV263",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakDenseCenterHeatmapFlowCalibratedV264",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakTopologyConstrainedCenterHeatmapFlowCalibratedV265",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactCenterHeatmapFlowV267",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSpatialEmbeddingV270",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSpatialEmbeddingContrastiveV271",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactCoordSpatialEmbeddingV279",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactQueryMaskV280",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactQueryMaskPositiveNegativeV281",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactGlobalCoordQueryMaskV285",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactGlobalCoordQueryMaskPositiveNegativeV286",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactFreeEmbeddingV282",
    "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactGlobalCoordFreeEmbeddingV283",
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
]


def _resolve_training_profiles(trainer: str,
                               loss_profile: str | None,
                               ce_class_weights: str | None,
                               oversample_profile: str | None) -> tuple[str, str, str]:
    """Apply trainer-specific defaults without changing explicit user choices."""
    if trainer == "PengwinTrainerBICMFactorizedV6":
        return (
            loss_profile or "bicm_v6_factorized",
            ce_class_weights or "off",
            oversample_profile or "bicm_v5_sparse",
        )
    if trainer == "PengwinTrainerBICMOracleAlignedDirectV250":
        return (
            loss_profile or "bicm_v250_oracle_aligned_direct",
            ce_class_weights or "off",
            oversample_profile or "bicm_v18_roi_balanced",
        )
    if trainer == "PengwinTrainerBICMContactV8":
        return (
            loss_profile or "bicm_v8_contact_contour",
            ce_class_weights or "off",
            oversample_profile or "bicm_v6_support_mixed",
        )
    if trainer == "PengwinTrainerBICMContactV9":
        return (
            loss_profile or "bicm_v9_contact_energy_pair",
            ce_class_weights or "off",
            oversample_profile or "bicm_v6_support_mixed",
        )
    if trainer == "PengwinTrainerBICMContactV10":
        return (
            loss_profile or "bicm_v10_adaptive_topology",
            ce_class_weights or "off",
            oversample_profile or "bicm_v6_support_mixed",
        )
    if trainer == "PengwinTrainerBICMContactV11":
        return (
            loss_profile or "bicm_v10_adaptive_topology",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV12":
        return (
            loss_profile or "bicm_v12_contact_persistent",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV13":
        return (
            loss_profile or "bicm_v13_contact_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV14":
        return (
            loss_profile or "bicm_v14_contact_curriculum",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV15":
        return (
            loss_profile or "bicm_v15_contact_presence_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV16":
        return (
            loss_profile or "bicm_v16_roi_presence_classifier",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV17":
        return (
            loss_profile or "bicm_v16_roi_presence_classifier",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV18":
        return (
            loss_profile or "bicm_v16_roi_presence_classifier",
            ce_class_weights or "off",
            oversample_profile or "bicm_v18_roi_balanced",
        )
    if trainer == "PengwinTrainerBICMContactV19":
        return (
            loss_profile or "bicm_v19_roi_calibrated_presence",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV20":
        return (
            loss_profile or "bicm_v19_roi_calibrated_presence",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSemanticCandidateV66":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V66 is a checkpoint-retention probe, not a new learning objective.
        # Keep the V62 semantic loss/sampler defaults here so CLI launches
        # cannot accidentally compare a different method while only changing
        # full-volume candidate selection.
        return (
            loss_profile or "bicm_v62_semantic_boundary",
            ce_class_weights or "off",
            oversample_profile or "bicm_v5_sparse",
        )
    if trainer == "PengwinTrainerBICMSemanticEdgePairV67":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V67 changes the semantic loss to local edge-pair ranking while
        # keeping Dataset537, sampler, decoder, and candidate-retention policy
        # fixed relative to V66.
        return (
            loss_profile or "bicm_v67_semantic_edge_pair",
            ce_class_weights or "off",
            oversample_profile or "bicm_v5_sparse",
        )
    if trainer == "PengwinTrainerBICMSemanticTopologyV68":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V68 is the first post-V67 representation diagnostic: semantic
        # support/exterior labels stay in a 5-class branch while contact/core
        # topology is learned from independent sidecar heads.
        return (
            loss_profile or "bicm_v68_semantic_topology",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMTopologyCalibratedV69":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V69 must differ from V68 only by sparse topology-head loss
        # calibration. Keep Dataset537, sampler, decoder, and 10-channel layout
        # fixed so the 003 diagnostic answers one question.
        return (
            loss_profile or "bicm_v69_topology_calibrated",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMTopologyConsistencyV70":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V70 is the direct follow-up to V69: only the loss terms that bind core
        # markers to semantic support and away from contact/edge ridges change.
        return (
            loss_profile or "bicm_v70_topology_consistency",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgeCutPrimaryV71":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V71 keeps V70's representation and decoder, but makes edge-break
        # affinity the primary split signal. Dense voxel contact is auxiliary.
        return (
            loss_profile or "bicm_v71_edge_cut_primary",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMLogitCalibratedV72":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V72 isolates fixed topology-head logit calibration. It reuses the V70
        # loss and decoder; only the initial edge/contact bias changes.
        return (
            loss_profile or "bicm_v72_logit_calibrated",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMInstanceTopologyV73":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V73 keeps the V72 representation, bias, sampler, and decoder. The
        # only active change is instance-aware seed-presence/contact-precision
        # loss from the existing sidecars.
        return (
            loss_profile or "bicm_v73_instance_topology",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMAdaptiveInstanceTopologyV74":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V74 keeps V72 representation/bias/sampler/decoder and replaces V73's
        # broad visible-fragment seed pressure with target-core-aware seed/mass
        # supervision plus adaptive contact precision-recall balancing.
        return (
            loss_profile or "bicm_v74_adaptive_instance_topology",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgePrecisionSeedTopologyV75":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V75 keeps V72 representation/bias/sampler/decoder, retains V73 seed
        # coverage pressure, and directly optimizes the edge-primary contact
        # head used by fixed full-volume promotion.
        return (
            loss_profile or "bicm_v75_edge_precision_seed_topology",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMGentleEdgePrecisionV76":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V76 backs off V75's strong precision pressure. It keeps V73 seed
        # behavior and adds only gentle edge-primary ranking/false-positive
        # pressure under the same fixed decoder.
        return (
            loss_profile or "bicm_v76_gentle_edge_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgeCurriculumV77":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V77 keeps the V73 representation/seed objective and separates edge
        # recall and precision pressures by epoch, avoiding V75's immediate
        # precision-driven recall collapse.
        return (
            loss_profile or "bicm_v77_edge_recall_precision_curriculum",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreAnchoredEdgeCurriculumV78":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V78 keeps Dataset537/V72 representation and the fixed full-volume
        # decoder. Only the loss changes: hard edge precision is gated by
        # fragment core viability and late phases preserve core/support.
        return (
            loss_profile or "bicm_v78_core_anchored_edge_curriculum",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDuplicateSeedEdgeV79":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V79 keeps the V68/V72 structure and edge-primary eval contract fixed.
        # The only active change is loss-only suppression of secondary core
        # peaks plus a gentler core-gated edge curriculum.
        return (
            loss_profile or "bicm_v79_duplicate_seed_edge",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMTopologyStateAdaptiveV80":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V80 keeps the V68/V72 structure and fixed edge-primary eval contract.
        # It changes only loss scheduling: seed coverage gates duplicate and
        # edge precision pressure while background core noise is suppressed.
        return (
            loss_profile or "bicm_v80_topology_state_adaptive",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreStableEdgePrecisionV81":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V81 keeps V80's core topology guard and changes only late edge
        # precision/rank/mass pressure after core seed readiness is viable.
        return (
            loss_profile or "bicm_v81_core_stable_edge_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSeedRecallEdgeSeparationV82":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V82 keeps the V68/V72/V80/V81 10-channel structure and fixed
        # full-volume eval contract. It only tests whether stronger seed recall
        # plus edge logit separation can fix V81's missing LeftHip seed and
        # insufficient fixed-threshold contact precision.
        return (
            loss_profile or "bicm_v82_seed_recall_edge_separation",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSeedSafeEdgeCalibrationV83":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V83 keeps V81's stable contact/core contract and tests a smaller
        # calibration step after V82 showed strong edge precision pressure can
        # collapse recall and seed coverage.
        return (
            loss_profile or "bicm_v83_seed_safe_edge_calibration",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDualHeadContactCalibrationV84":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V84 keeps the V68/V72 10-channel structure but trains channel 5 as a
        # dense contact gate for the existing semantic_topology_v68 contract.
        # Fulltrain remains blocked until fixed full-volume gates pass.
        return (
            loss_profile or "bicm_v84_dual_head_contact_calibration",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSemanticGateContactV85":
        # [AUDIT][Risk:Medium][Scope:experiment_contract]
        # V85 keeps V81's recovered core topology and trains only the fixed
        # semantic_topology_v68 product gate, without changing thresholds or
        # promoting to fulltrain before the case003 full-volume gate passes.
        return (
            loss_profile or "bicm_v85_semantic_gate_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEvalBandEdgePrimaryV86":
        # [AUDIT][Risk:High][Scope:target_contract_alignment]
        # V86 corrects the measured train/eval mismatch: sidecar direct-contact
        # masks supervise topology, while promotion measures the semantic
        # contact band. Keep Dataset537, sampler, channels, and thresholds fixed.
        return (
            loss_profile or "bicm_v86_eval_band_edge_primary",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEvalBandSemanticGateV87":
        # [AUDIT][Risk:High][Scope:target_contract_alignment]
        # V87 tests the same semantic contact-band target under the fixed
        # semantic_topology_v68 product contract. Fulltrain remains blocked
        # until fixed full-volume gates pass.
        return (
            loss_profile or "bicm_v87_eval_band_semantic_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCorePreservingBandPrecisionV88":
        # [AUDIT][Risk:High][Scope:core_preserving_contact_precision]
        # V88 keeps V81's contact-core structure and uses the eval contact band
        # only as a late hard-false calibration signal. No threshold, decoder,
        # dataset, or sampler change is allowed before the case003 gate passes.
        return (
            loss_profile or "bicm_v88_core_preserving_band_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMBandFalseOnlyPrecisionV89":
        # [AUDIT][Risk:High][Scope:negative_only_contact_precision]
        # V89 removes band-positive pressure and tests whether non-band hard
        # false suppression alone can raise edge-primary precision while V81's
        # direct edge-break objective preserves contact recall.
        return (
            loss_profile or "bicm_v89_band_false_only_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSemanticBandProductV90":
        # [AUDIT][Risk:High][Scope:semantic_contact_contract]
        # V90 changes the fixed contact representation under test: semantic
        # class-4 contact-band probability gates edge-primary contact. Dataset,
        # sampler, threshold, and core decoder remain fixed.
        return (
            loss_profile or "bicm_v90_semantic_band_product",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMTopologyAwareEdgeBalanceV91":
        # [AUDIT][Risk:High][Scope:left_hip_bottleneck]
        # V91 preserves V89's support/core base and targets the two remaining
        # case003 blockers: duplicate LeftHip seed components and underbalanced
        # eval-band edge positives. Contact contract remains v68_edge_primary.
        return (
            loss_profile or "bicm_v91_topology_aware_edge_balance",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEvalAlignedSupportContactV93":
        # [AUDIT][Risk:High][Scope:eval_aligned_geometry]
        # V93 preserves V89's core base after the peak17 decoder diagnostic and
        # targets the measured LeftHip blockers: semantic support hard errors
        # and eval-band edge barrier separation. Contact contract remains
        # v68_edge_primary and fulltrain remains blocked until case003 passes.
        return (
            loss_profile or "bicm_v93_eval_aligned_support_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMFinalRowCalibratedV94":
        # [AUDIT][Risk:High][Scope:core_preserving_refinement]
        # V94 must be initialized from a viable V89 checkpoint. It freezes
        # shared features and core row 6, then updates only final semantic,
        # dense-contact, and edge rows with low-gain eval-aligned losses.
        return (
            loss_profile or "bicm_v94_final_row_support_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDecoderFeatureContactV95":
        # [AUDIT][Risk:High][Scope:decoder_feature_contact]
        # V95 keeps V89/V94's fixed 10-channel contract and peak17 decoder, but
        # opens decoder features after V94 showed final-row-only calibration was
        # not expressive enough to separate contact-band false positives.
        return (
            loss_profile or "bicm_v95_decoder_feature_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDenseBandGateV96":
        # [AUDIT][Risk:High][Scope:dense_band_gate]
        # V96 changes the evaluated contact structure from edge-primary to
        # dense-edge product and trains only dense row 5 as a semantic-band gate
        # over the frozen V89 support/core/edge rows.
        return (
            loss_profile or "bicm_v96_dense_band_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMPositiveDenseBandGateV97":
        # [AUDIT][Risk:High][Scope:dense_band_gate_recall]
        # V97 keeps V96's structural contact contract but increases positive
        # band pressure after V96 proved the gate was precise but under-opened.
        return (
            loss_profile or "bicm_v97_positive_dense_band_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMTeacherDistilledDenseGateV98":
        # [AUDIT][Risk:High][Scope:teacher_distilled_gate]
        # V98 opens decoder features for contact separability, but distills
        # semantic/core/edge outputs from the V89 teacher to avoid repeating
        # V95's support/core regressions.
        return (
            loss_profile or "bicm_v98_teacher_distilled_dense_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMAdaptiveDualMarginDenseGateV99":
        # [AUDIT][Risk:High][Scope:adaptive_dense_gate]
        # V98 diagnostics showed dense row 5 is precise but closed while
        # semantic/edge heads are recalled but false-positive heavy. V99 keeps
        # V96's row-only structure and adapts false-contact pressure to the
        # positive product response.
        return (
            loss_profile or "bicm_v99_adaptive_dual_margin_dense_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMNegativeBalancedDenseGateV100":
        # [AUDIT][Risk:High][Scope:negative_balanced_dense_gate]
        # V99 showed the loss can move between high-precision/low-recall and
        # high-recall/low-precision states. V100 keeps that loss and changes only
        # V38-family crop exposure toward support/hard negatives.
        return (
            loss_profile or "bicm_v100_negative_balanced_dense_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDualFieldProductGateV101":
        # [AUDIT][Risk:High][Scope:dual_contact_field]
        # V96-V100 exhausted row-5-only dense gating under the fixed
        # semantic_topology_v68 contract. V101 changes exactly one structural
        # assumption: rows 7..9 are trained with row 5 as a joint product field
        # while semantic/core rows remain warm-start restored.
        return (
            loss_profile or "bicm_v101_dual_field_product_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDistanceRankDenseGateV102":
        # [AUDIT][Risk:High][Scope:distance_rank_contact_gate]
        # V101 proved that jointly opening row 5 and edge rows 7..9 still
        # oscillates between closed/precise and open/noisy contact. V102 keeps
        # V89 support/core/edge rows fixed and changes only the row-5 loss shape:
        # contact shoulders are below-threshold soft negatives, far support is
        # hard-suppressed, and the fixed semantic_topology_v68 decoder is kept.
        return (
            loss_profile or "bicm_v102_distance_rank_dense_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSemanticDistanceContactV103":
        # [AUDIT][Risk:High][Scope:semantic_contact_contract]
        # V96-V102 show the dense/edge product cannot satisfy fixed contact
        # precision and recall together. V103 trains only semantic row 4 as the
        # direct contact field and keeps the measured-safe V89 support/core/edge
        # rows fixed for the case003 gate.
        return (
            loss_profile or "bicm_v103_semantic_distance_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMTeacherSemanticContactV104":
        # [AUDIT][Risk:High][Scope:capacity_vs_methodology]
        # V103 final-row-only training failed. V104 opens decoder features with
        # V89 teacher preservation to test whether direct semantic contact is
        # capacity-limited before abandoning the contact-scalar family.
        return (
            loss_profile or "bicm_v104_teacher_semantic_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMOffsetAssignmentV105":
        # [AUDIT][Risk:High][Scope:methodology_shift]
        # V96-V104 rejected contact/product/semantic-contact calibration as the
        # primary split mechanism. V105 keeps support/core teacher preservation
        # but trains row5 as a metric head and rows7..9 as center offsets.
        return (
            loss_profile or "bicm_v105_offset_assignment",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMOffsetAttractorV106":
        # [AUDIT][Risk:High][Scope:offset_oracle_response]
        # V105 oracle decomposition showed GT offsets recover LeftHip assignment
        # with the same fixed decoder. V106 resets edge-logit offset rows and
        # makes endpoint attraction the primary supervised variable.
        return (
            loss_profile or "bicm_v106_offset_attractor",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMRadialSupportOffsetV107":
        # [AUDIT][Risk:High][Scope:v106_oracle_response]
        # V106 still under-predicts offset magnitude and caps LeftHip assignment
        # through support false positives. V107 keeps the V106 offset decoder
        # contract but opens semantic support rows for direct hard-false
        # calibration before any fulltrain is allowed.
        return (
            loss_profile or "bicm_v107_radial_support_offset",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMWatershedBarrierV108":
        # [AUDIT][Risk:High][Scope:offset_branch_rejection]
        # V105-V107 rejected learned center-offset assignment. V108 returns to
        # the stronger V89/V92 watershed evidence and trains only edge rows as
        # a fixed-threshold watershed barrier while preserving semantic/core
        # behavior with the V89 teacher.
        return (
            loss_profile or "bicm_v108_watershed_barrier",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMPrecisionLockedRecallV109":
        # [AUDIT][Risk:High][Scope:v109_stage_from_v96]
        # V96 e10 is the best measured precision/topology state. V109 does not
        # change the decoder or support/core/edge rows; it continues row 5 with
        # a teacher negative-expansion lock to test whether recall can open
        # without losing the V96 precision budget.
        return (
            loss_profile or "bicm_v109_precision_locked_recall",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSemanticGeometryBridgeV110":
        # [AUDIT][Risk:High][Scope:v110_oracle_bridge]
        # Mixed-oracle diagnostics showed GT semantic support/contact with the
        # predicted core passes the case003 gate. V110 therefore trains only
        # semantic rows and evaluates semantic contact directly while preserving
        # core and edge rows.
        return (
            loss_profile or "bicm_v110_semantic_geometry_bridge",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMJointSupportProductV111":
        # [AUDIT][Risk:High][Scope:v111_joint_contract]
        # V110 proved direct semantic-contact evaluation is unstable. V111 keeps
        # the V96/V109 dense-edge product decoder contract, opens decoder-feature
        # capacity, and jointly repairs semantic support plus contact rows while
        # preserving the already sufficient row6 core with a teacher.
        return (
            loss_profile or "bicm_v111_joint_support_product",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgeGraphAssignmentV112":
        # [AUDIT][Risk:High][Scope:v112_graph_assignment]
        # V111 showed the dense-edge product remains stuck. V112 changes the
        # assignment contract: rows 7..9 are trained as local edge-break graph
        # cuts and evaluated with an edge-graph decoder while preserving row6 core.
        return (
            loss_profile or "bicm_v112_edge_graph_assignment",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer in {
        "PengwinTrainerBICMAdaptiveBoundaryProductV113",
        "PengwinTrainerBICMAdaptiveBoundaryProductV113HighLR",
    }:
        # [AUDIT][Risk:High][Scope:v113_methodology_check]
        # V112 disproved the edge-graph assignment contract. V113 keeps the
        # measured V92 watershed decoder and changes only loss allocation:
        # support-boundary emphasis plus fixed-threshold adaptive contact P/R.
        return (
            loss_profile or "bicm_v113_adaptive_boundary_product",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEncoderAdapterV114":
        # [AUDIT][Risk:High][Scope:v114_methodology_check]
        # V113 showed the adaptive loss still cannot move full-volume support
        # geometry under the decoder-only calibration regime. V114 keeps the
        # loss/decoder fixed and opens late encoder capacity as the single
        # changed variable.
        return (
            loss_profile or "bicm_v114_encoder_adapter_boundary_product",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSemanticOracleAdapterV115":
        # [AUDIT][Risk:High][Scope:v115_oracle_contract]
        # The mixed oracle passed only when semantic support/contact were GT and
        # row6 core stayed predicted. V115 directly trains that semantic
        # contract with late encoder capacity instead of the dense-edge product.
        return (
            loss_profile or "bicm_v115_semantic_oracle_adapter",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMGraphCostSeparatorV116":
        # [AUDIT][Risk:High][Scope:v116_structured_separator]
        # V115 confirmed direct semantic contact is recall-biased and imprecise.
        # V116 trains local separator costs from edge_break sidecars and uses a
        # graph-cost assignment decoder instead of scalar contact watershed.
        return (
            loss_profile or "bicm_v116_graph_cost_separator",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportGateSemanticContactV117":
        # [AUDIT][Risk:High][Scope:v117_support_geometry_contract]
        # Mixed oracles showed GT contact alone fails with predicted support,
        # while GT semantic support/contact passes with row6 predicted core.
        # V117 makes row5 an independent fixed-threshold support gate before
        # re-testing semantic contact, instead of adding another edge/contact
        # calibration loss.
        return (
            loss_profile or "bicm_v117_support_gate_semantic_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMWarmSupportGateSemanticContactV118":
        # [AUDIT][Risk:High][Scope:v118_gate_collapse_fix]
        # V117 proved the support-gate contract is too brittle when row5 starts
        # from a contact head and false-mass pressure closes the gate. V118
        # keeps the same evaluator contract but warm-initializes row5 from
        # semantic support-vs-background rows and relaxes gate false pressure.
        return (
            loss_profile or "bicm_v118_warm_support_gate_semantic_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMAllNetworkAdaptiveBoundaryV119":
        # [AUDIT][Risk:High][Scope:v119_methodology_probe]
        # V114 showed late-encoder adapters remain baseline-stuck. V119 changes
        # the trainable scope to the full network at very low LR while keeping
        # the V92 semantic-topology contract and row6 teacher preservation.
        return (
            loss_profile or "bicm_v119_all_network_adaptive_boundary",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSameFragmentAffinityV120":
        # [AUDIT][Risk:High][Scope:v120_topology_contract]
        # V112/V116 trained sparse break/separator rows and failed under graph
        # assignment. V120 changes only the row7..9 target meaning to dense
        # same-fragment affinity, then decodes with low affinity as traversal
        # barrier. Row5 contact and row6 core remain teacher-preserved.
        return (
            loss_profile or "bicm_v120_same_fragment_affinity",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMWarmSameFragmentAffinityV121":
        # [AUDIT][Risk:High][Scope:v121_affinity_initialization]
        # V120 preserved the old checkpoint under V92 decoding but collapsed the
        # affinity graph because rows7..9 stayed low everywhere. V121 changes only
        # initialization of those rows to a support-interior affinity prior.
        return (
            loss_profile or "bicm_v121_warm_same_fragment_affinity",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMWarmEdgeProductPrecisionV122":
        # [AUDIT][Risk:High][Scope:v122_contract_return]
        # V120/V121 rejected the affinity graph structure. V122 keeps the proven
        # semantic_topology_v68 product decoder, applies a smaller support-derived
        # row7..9 warm start, and locks negative expansion against the V96 teacher.
        return (
            loss_profile or "bicm_v122_warm_edge_product_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMHighRecallGateCleanupV123":
        # [AUDIT][Risk:High][Scope:v123_single_variable]
        # V122 showed joint row5+edge training still leaves broad false contact.
        # V123 freezes the V121 high-recall edge rows and trains only row5 as a
        # precision cleanup gate under the original semantic_topology_v68 decoder.
        return (
            loss_profile or "bicm_v123_high_recall_gate_cleanup",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportConditionedEdgePrecisionV124":
        # [AUDIT][Risk:High][Scope:v124_false_field_cleanup]
        # V123 proved row5-only cleanup is too weak. V124 freezes row5/semantic/core
        # and trains only rows7..9 against non-contact support hard negatives so
        # the false field is reduced before the product gate.
        return (
            loss_profile or "bicm_v124_support_conditioned_edge_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSaturatedEdgeDenseGateV125":
        # [AUDIT][Risk:High][Scope:v125_saturated_edge_diagnosis]
        # V124 full-volume ablation showed rows7..9 are saturated and row5 is
        # the effective contact gate. V125 opens decoder features but preserves
        # semantic/core/edge outputs with teacher distillation, changing only the
        # learned row5 contact geometry under the fixed semantic_topology_v68 decoder.
        return (
            loss_profile or "bicm_v125_saturated_edge_dense_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer in {
        "PengwinTrainerBICMLocalAdjacencyProductV126",
        "PengwinTrainerBICMLocalAdjacencyProductV126FromV96",
        "PengwinTrainerBICMEdgeResetLocalAdjacencyProductV127",
    }:
        # [AUDIT][Risk:High][Scope:v126_instance_adjacency]
        # V125 proved row5/edge threshold retuning stays on the same tradeoff
        # curve. V126 adds fragment-ID sidecar supervision: local different-
        # fragment adjacencies become break positives and same-fragment/support
        # adjacencies become hard negatives under the unchanged V92 decoder.
        # V127 reuses the same loss and changes only rows7..9 initialization.
        return (
            loss_profile or "bicm_v126_local_adjacency_product",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportBridgeSuppressionV128":
        # [AUDIT][Risk:High][Scope:v128_support_topology_probe]
        # Partition probes isolated predicted support false bridges as the main
        # blocker. V128 changes support pressure only while preserving V121
        # affinity/contact/core signals through the trainer/loss contract.
        return (
            loss_profile or "bicm_v128_support_bridge_suppression",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMAffinitySharpeningV129":
        # [AUDIT][Risk:High][Scope:v129_affinity_probe]
        # V121 affinity reached the graph-partition upper band under oracle
        # support/core. V129 changes only rows7..9 affinity sharpness so it can
        # be compared against V128 support-topology repair.
        return (
            loss_profile or "bicm_v129_affinity_sharpening",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportTopologyRepairV130":
        # [AUDIT][Risk:High][Scope:v130_support_topology_probe]
        # V128's generic non-support mining did not move the mixed-oracle support
        # control. V130 adds instance-aware false-bridge zones and opens one
        # earlier encoder stage while preserving contact/core/affinity rows.
        return (
            loss_profile or "bicm_v130_support_topology_repair",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMAllNetworkSupportTopologyRepairV131":
        # [AUDIT][Risk:High][Scope:v131_capacity_probe]
        # Same support-topology loss as V130, but all-network low-LR capacity.
        # This tests whether support false bridges are representation-limited
        # instead of only a final/late-adapter calibration issue.
        return (
            loss_profile or "bicm_v131_all_network_support_topology_repair",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportVetoGateV132":
        # [AUDIT][Risk:High][Scope:v132_support_veto_gate]
        # V128-V131 left high-confidence semantic support bridges. V132 changes
        # row5 from contact to an independent support-veto gate while preserving
        # semantic/core/affinity rows for a gated V120 decoder probe.
        return (
            loss_profile or "bicm_v132_support_veto_gate",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportVetoSemanticV133":
        # [AUDIT][Risk:High][Scope:v133_support_veto_semantic]
        # V133 tests whether the row5 veto gate needs semantic rows to co-adapt
        # instead of acting as a final-row-only mask.
        return (
            loss_profile or "bicm_v133_support_veto_semantic",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportVetoGateHighLRV134":
        # [AUDIT][Risk:High][Scope:v134_gate_lr_probe]
        # V132/V133 left row5 distribution effectively unchanged from V121.
        # V134 changes only LR magnitude to test whether the gate failed because
        # the update was too weak.
        return (
            loss_profile or "bicm_v134_support_veto_gate_highlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportVetoGateUltraLRV135":
        # [AUDIT][Risk:High][Scope:v135_gate_lr_probe]
        # Same row5 support-veto contract as V132/V134, with a more aggressive
        # LR to bound whether row5 can be moved at all in a short probe.
        return (
            loss_profile or "bicm_v135_support_veto_gate_ultralr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportTopologyHighLRV136":
        # [AUDIT][Risk:High][Scope:v136_semantic_lr_probe]
        # V130 had the right semantic-support topology objective but barely
        # moved. V136 changes only LR magnitude to test whether semantic support
        # rows can repair bridges when updates are stronger.
        return (
            loss_profile or "bicm_v136_support_topology_highlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportTopologyUltraLRV137":
        # [AUDIT][Risk:High][Scope:v137_semantic_lr_probe]
        # Same V130 support-topology loss with a more aggressive LR upper bound.
        return (
            loss_profile or "bicm_v137_support_topology_ultralr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportTopologyMidLRV138":
        # [AUDIT][Risk:High][Scope:v138_semantic_lr_bracket]
        # V136 improved false-support volume at 3e-4 while V137 collapsed at
        # 1e-3. V138 samples the middle of that LR bracket.
        return (
            loss_profile or "bicm_v138_support_topology_midlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportTopologyUpperMidLRV139":
        # [AUDIT][Risk:High][Scope:v139_semantic_lr_bracket]
        # Upper-mid LR bracket for the V130 semantic support topology objective.
        return (
            loss_profile or "bicm_v139_support_topology_uppermidlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportTopologyLowBracketLRV140":
        # [AUDIT][Risk:High][Scope:v140_semantic_lr_bracket]
        # V136 was the best observed point at 3e-4; V140 tests whether a lower
        # nearby LR preserves support while still reducing false bridges.
        return (
            loss_profile or "bicm_v140_support_topology_lowbracket",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportTopologyHighBracketLRV141":
        # [AUDIT][Risk:High][Scope:v141_semantic_lr_bracket]
        # V138 at 5e-4 already regressed. V141 samples the narrow upper side of
        # V136 before crossing into the observed collapse band.
        return (
            loss_profile or "bicm_v141_support_topology_highbracket",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportTopologyCoreSeedLowLRV142":
        # [AUDIT][Risk:High][Scope:v142_core_seed_oracle_gap]
        # Mixed-oracle probes show target core raises V140 from ~0.765 to ~0.860.
        # V142 changes only row6/core supervision at the V140 LR anchor.
        return (
            loss_profile or "bicm_v142_support_topology_coreseed_lowlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportTopologyCoreSeedMidLRV143":
        # [AUDIT][Risk:High][Scope:v143_core_seed_oracle_gap]
        # Same core-seed intervention at the V136 LR anchor to separate LR from
        # row6 supervision effects.
        return (
            loss_profile or "bicm_v143_support_topology_coreseed_midlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMStrongCoreSeedLowLRV144":
        # [AUDIT][Risk:High][Scope:v144_strong_core_seed_probe]
        # V142 under-moved row6. V144 keeps the V140 LR anchor and increases
        # only core-seed pressure to test whether the oracle gap is actionable.
        return (
            loss_profile or "bicm_v144_strong_coreseed_lowlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMStrongCoreSeedMidLRV145":
        # [AUDIT][Risk:High][Scope:v145_strong_core_seed_probe]
        # Same strong core-seed objective at the V136/V143 LR anchor.
        return (
            loss_profile or "bicm_v145_strong_coreseed_midlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMFragmentSeedPresenceLowLRV146":
        # [AUDIT][Risk:High][Scope:v146_zero_seed_roi]
        # Cases 011/017 failed through missing row6 seeds. V146 changes only the
        # fragment-level seed-presence objective at the V144 LR anchor.
        return (
            loss_profile or "bicm_v146_fragment_seed_presence_lowlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMFragmentSeedPresenceMidLRV147":
        # [AUDIT][Risk:High][Scope:v147_zero_seed_roi]
        # Same seed-presence objective at the V145 LR anchor to isolate whether
        # the effect is loss-driven or LR-sensitive.
        return (
            loss_profile or "bicm_v147_fragment_seed_presence_midlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreHeatmapSeedLowLRV148":
        # [AUDIT][Risk:High][Scope:v148_core_seed_contract]
        # V146/V147 kept failing core topology. V148 changes only row6 seed
        # supervision from semantic core labels to the dataloader's per-fragment
        # core heatmap target at the V144 LR anchor.
        return (
            loss_profile or "bicm_v148_core_heatmap_seed_lowlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreHeatmapSeedMidLRV149":
        # [AUDIT][Risk:High][Scope:v149_core_seed_contract]
        # Same heatmap-target seed contract at the V145 LR anchor. Decoder,
        # contact contract, data, and sampler stay fixed.
        return (
            loss_profile or "bicm_v149_core_heatmap_seed_midlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreOnlyHeatmapSeedMidLRV150":
        # [AUDIT][Risk:High][Scope:v150_core_only_seed_contract]
        # Oracle controls show target core rescues 011/017 but predicted core
        # fails even with target support. V150 opens only row6 at the V149 LR.
        return (
            loss_profile or "bicm_v150_core_only_heatmap_seed_midlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreOnlyHeatmapSeedHighLRV151":
        # [AUDIT][Risk:High][Scope:v151_core_only_seed_lr_bracket]
        # Same row6-only heatmap target at a higher LR to test whether the
        # under-moved core field needs stronger calibration without moving
        # semantic/contact/affinity rows.
        return (
            loss_profile or "bicm_v151_core_only_heatmap_seed_highlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreOnlyCenterSeedMidLRV152":
        # [AUDIT][Risk:High][Scope:v152_core_target_geometry]
        # V150/V151 rejected LR-only row6 calibration. V152 changes only the
        # row6 target geometry to one compact instance-center seed per visible
        # fragment while preserving non-core rows.
        return (
            loss_profile or "bicm_v152_core_only_center_seed_midlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreHeatmapCenterSeedMidLRV153":
        # [AUDIT][Risk:High][Scope:v153_center_seed_with_support_topology]
        # Matched to V149: same support-topology branch and LR, but dataloader
        # target["core"] becomes a compact per-fragment center marker.
        return (
            loss_profile or "bicm_v153_core_heatmap_center_seed_midlr",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreOnlyCenterSeedSamplerV154":
        # [AUDIT][Risk:High][Scope:v154_sampler_ablation]
        # V152 changed target geometry but hard cases still missed sacrum seeds.
        # V154 changes only crop exposure toward GT fragment centers.
        return (
            loss_profile or "bicm_v154_core_only_center_seed_sampler",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreHeatmapCenterSeedSamplerV155":
        # [AUDIT][Risk:High][Scope:v155_sampler_ablation]
        # Matched V153 branch with the same sampler-only intervention as V154.
        return (
            loss_profile or "bicm_v155_core_heatmap_center_seed_sampler",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreOnlyCenterSeedAffinitySamplerV158":
        # [AUDIT][Risk:High][Scope:v158_affinity_grouping_ablation]
        # V154 fixed center-seed exposure but kept rows7..9 teacher-preserved.
        # V158 opens only row6 plus affinity rows so grouping can be tested.
        return (
            loss_profile or "bicm_v158_core_only_center_seed_affinity_sampler",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreHeatmapCenterSeedAffinitySamplerV159":
        # [AUDIT][Risk:High][Scope:v159_affinity_grouping_ablation]
        # Matched V155 support/core branch with rows7..9 affinity supervision.
        return (
            loss_profile or "bicm_v159_core_heatmap_center_seed_affinity_sampler",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreOnlyCenterSeedOffsetSamplerV160":
        # [AUDIT][Risk:High][Scope:v160_offset_grouping_ablation]
        # V158/V159 rejected same-fragment affinity. V160 keeps V154's scaffold
        # and changes rows7..9 to raw center offsets for soft-seed assignment.
        return (
            loss_profile or "bicm_v160_core_only_center_seed_offset_sampler",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCoreHeatmapCenterSeedOffsetSamplerV161":
        # [AUDIT][Risk:High][Scope:v161_offset_grouping_ablation]
        # Matched V155 support/core branch with rows7..9 raw center offsets.
        return (
            loss_profile or "bicm_v161_core_heatmap_center_seed_offset_sampler",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMBoundaryFragmentRemapV162":
        # [AUDIT][Risk:High][Scope:v162_abbc_contract_probe]
        # V162 keeps Dataset537/fold0 and only remaps existing V5 semantic
        # labels into the BoundaryFragmentV3 boundary/shell/core contract.
        return (
            loss_profile or "boundary_fragment_v3",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_v5_remap",
        )
    if trainer == "PengwinTrainerFactorizedInstanceEmbeddingV163":
        # [AUDIT][Risk:High][Scope:v163_embedding_ablation]
        # V163 keeps the factorized support/contact/core scaffold and enables
        # compact embedding supervision from dense instance sidecars.
        return (
            loss_profile or "factorized_instance_v4_embedding_v163",
            ce_class_weights or "off",
            oversample_profile or "factorized_instance_v4_v163",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarV164":
        # [AUDIT][Risk:High][Scope:v164_bfv3_target_sidecar]
        # V164 tests actual dense-instance-derived BFV3 targets, not V5
        # semantic label remapping. Dataset537/fold0/checkpoint gate stay fixed.
        return (
            loss_profile or "boundary_fragment_v3",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarPrecisionV165":
        # [AUDIT][Risk:High][Scope:v165_bfv3_loss_ablation]
        # V165 keeps V164 target/data contract and changes only the BFV3 ridge
        # precision loss to test whether class-2 collapse is loss-driven.
        return (
            loss_profile or "boundary_fragment_v3_ridgefit_precision",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_precision",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreSeedV166":
        # [AUDIT][Risk:High][Scope:v166_bfv3_core_seed_recovery]
        # V166 tests whether explicit class-4 core supervision can recover
        # fragment seeds after V164/V165 collapsed core to zero.
        return (
            loss_profile or "boundary_fragment_v3_core_seed",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_seed",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRidgeV167":
        # [AUDIT][Risk:High][Scope:v167_bfv3_core_ridge_combined]
        # V167 keeps V166's core exposure and adds V165-style class-2 ridge
        # precision to test whether both decoder prerequisites can coexist.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_ridge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRidgeHighEdgeV168":
        # [AUDIT][Risk:High][Scope:v168_bfv3_barrier_exposure]
        # V168 keeps V167's loss and changes only class-2 crop exposure to test
        # whether full-fold barrier collapse is sampler-driven.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_ridge_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallRidgeV169":
        # [AUDIT][Risk:High][Scope:v169_bfv3_barrier_recall_bias]
        # V169 keeps V168's high class-2 exposure and changes only the ridge loss
        # bias from precision-heavy to recall-oriented.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallMassCapV170":
        # [AUDIT][Risk:High][Scope:v170_bfv3_barrier_mass_cap]
        # V170 keeps V169's recall-biased ridge/core setup and changes only the
        # class-2 mass budget to reduce the measured barrier overprediction.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_masscap",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallTightMassCapV171":
        # [AUDIT][Risk:High][Scope:v171_bfv3_tight_barrier_mass_cap]
        # V171 is paired with V170 and tightens only the class-2 mass cap to
        # bracket the precision/recall tradeoff under the same data exposure.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_masscap_tight",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallPositiveMassCapV172":
        # [AUDIT][Risk:High][Scope:v172_bfv3_positive_only_mass_cap]
        # V172 keeps the V170 positive-patch cap but removes the empty-patch
        # penalty to isolate which term closed barrier recall.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_pos_masscap",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftPositiveMassCapV173":
        # [AUDIT][Risk:High][Scope:v173_bfv3_soft_positive_mass_cap]
        # V173 softens only the positive mass cap relative to V172 while keeping
        # V169's high-edge exposure and recall-biased ridge/core loss.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_soft_pos_masscap",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallVerySoftPositiveMassCapV174":
        # [AUDIT][Risk:High][Scope:v174_bfv3_very_soft_positive_mass_cap]
        # V174 changes only positive cap strength relative to V173 to check if
        # barrier recall recovers without reverting to V169-level overprediction.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_very_soft_pos_masscap",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftMassCapStableScoreV175":
        # [AUDIT][Risk:High][Scope:v175_bfv3_checkpoint_selection]
        # V175 keeps V173's soft positive mass-cap loss and changes only
        # checkpoint scoring/retention toward barrier-stable epochs.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_soft_pos_masscap",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallVerySoftMassCapStableScoreV176":
        # [AUDIT][Risk:High][Scope:v176_bfv3_checkpoint_selection]
        # V176 keeps V174's very-soft positive mass-cap loss and changes only
        # checkpoint scoring/retention toward barrier-stable epochs.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_very_soft_pos_masscap",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftMassCapStrictStableV177":
        # [AUDIT][Risk:High][Scope:v177_bfv3_strict_checkpoint_gate]
        # V177 keeps V175's loss and requires nonzero barrier precision/recall
        # plus bounded class-2 ratio before saving barrier-stable checkpoints.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_soft_pos_masscap",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallVerySoftMassCapStrictStableV178":
        # [AUDIT][Risk:High][Scope:v178_bfv3_strict_checkpoint_gate]
        # V178 keeps V176's loss and applies the same strict nonzero
        # barrier-stable checkpoint gate.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_very_soft_pos_masscap",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallLogitCalibratedV179":
        # [AUDIT][Risk:High][Scope:v179_bfv3_class2_logit_calibration]
        # V179 keeps the BFV3 target/scaffold and adds direct class-2 logit
        # calibration to test whether threshold-0.5 barrier can recover.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_logit_calibrated",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallPrecisionLogitCalibratedV180":
        # [AUDIT][Risk:High][Scope:v180_bfv3_precision_calibration]
        # V180 is paired with V179 and changes only the class-2 calibration
        # strength toward precision and false-positive suppression.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_precision_logit_calibrated",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallShellContrastV181":
        # [AUDIT][Risk:High][Scope:v181_bfv3_shell_barrier_contrast]
        # V181 targets the measured V180 hard-case failure where most class-2
        # false positives came from shell label 3.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_shell_contrast_logit_calibrated",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallStrongShellContrastV182":
        # [AUDIT][Risk:High][Scope:v182_bfv3_strong_shell_barrier_contrast]
        # V182 changes only shell-vs-barrier contrast strength relative to V181.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_strong_shell_contrast_logit_calibrated",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallBinaryBarrierHeadV183":
        # [AUDIT][Risk:High][Scope:v183_bfv3_binary_barrier_head]
        # V183 keeps BFV3 semantic channels 0..4 and adds only channel 5 as an
        # independent binary barrier/contact head.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_binary_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallPrecisionBinaryBarrierHeadV184":
        # [AUDIT][Risk:High][Scope:v184_bfv3_precision_binary_barrier_head]
        # V184 pairs with V183 and changes only binary-head precision pressure.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_precision_binary_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallPositiveFloorBinaryBarrierHeadV185":
        # [AUDIT][Risk:High][Scope:v185_bfv3_binary_positive_floor]
        # V185 tests whether the independent binary head needs explicit positive
        # threshold-floor pressure to place true barrier above 0.5.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_positive_floor_binary_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallPositiveFloorMassCapBinaryBarrierHeadV186":
        # [AUDIT][Risk:High][Scope:v186_bfv3_binary_positive_floor_masscap]
        # V186 pairs V185 with a predicted-mass cap to bracket recall and false
        # barrier spread under the same six-channel contract.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_positive_floor_masscap_binary_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateBinaryBarrierHeadV187":
        # [AUDIT][Risk:High][Scope:v187_bfv3_candidate_barrier_head]
        # V187 trains the binary barrier head only inside GT class2/3 candidate
        # bands and validates it inside predicted semantic class2/3 candidates.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_candidate_binary_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateMassCapBinaryBarrierHeadV188":
        # [AUDIT][Risk:High][Scope:v188_bfv3_candidate_masscap_barrier_head]
        # V188 keeps V187's candidate restriction and adds only a candidate-band
        # mass cap to bracket recall/precision.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_candidate_masscap_binary_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateSoftRidgeBarrierHeadV189":
        # [AUDIT][Risk:High][Scope:v189_bfv3_soft_ridge_barrier]
        # V189 keeps the six-channel candidate scaffold but changes the aux
        # target from hard class2-vs-shell binary to a fixed local soft ridge.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_candidate_soft_ridge_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateSoftRidgeContrastBarrierHeadV190":
        # [AUDIT][Risk:High][Scope:v190_bfv3_soft_ridge_contrast]
        # V190 pairs V189 with stronger shell contrast and candidate mass cap to
        # test whether the soft ridge can keep barrier open without shell spread.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_candidate_soft_ridge_contrast_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateLogitRidgeBarrierHeadV191":
        # [AUDIT][Risk:High][Scope:v191_bfv3_logit_ridge_barrier]
        # V191 removes hard binary BCE/Tversky/focal aux terms and trains the
        # sixth channel only with fixed soft-target logit regression.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_candidate_logit_ridge_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallCandidateLogitRidgePrecisionBarrierHeadV192":
        # [AUDIT][Risk:High][Scope:v192_bfv3_logit_ridge_precision_barrier]
        # V192 keeps V191's pure regression contract but lowers/weights shell
        # targets to test a precision-biased bracket on the second GPU.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_candidate_logit_ridge_precision_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBinaryBarrierHeadV193":
        # [AUDIT][Risk:High][Scope:v193_dense_aux_barrier_target]
        # V193 tests whether class2 barrier target sparsity is the bottleneck by
        # marking adjacent GT shell voxels as aux positives only.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_binary_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateMassCapBarrierHeadV194":
        # [AUDIT][Risk:High][Scope:v194_dense_aux_barrier_masscap]
        # V194 pairs V193 with a mass cap against the original class2 target to
        # bracket recall and shell overpaint under the same dense aux target.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_masscap_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateWeakMassCapBarrierHeadV195":
        # [AUDIT][Risk:High][Scope:v195_dense_aux_weak_masscap]
        # V194 closed the barrier. V195 keeps V193's dense target and adds only
        # a weak original-class2 mass control to reduce early overpaint.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_weak_masscap_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftDenseCandidateBarrierHeadV196":
        # [AUDIT][Risk:High][Scope:v196_soft_dense_aux_target]
        # V196 keeps target densification but labels adjacent shell as a soft
        # aux positive instead of the hard positive used in V193.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_soft_dense_candidate_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateLongProbeV197":
        # [AUDIT][Risk:High][Scope:v197_v193_long_probe]
        # V197 keeps V193's dense aux-positive target exactly and extends only
        # the run length to test whether the short-gate signal survives.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_binary_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateLowLRLongProbeV198":
        # [AUDIT][Risk:High][Scope:v198_v193_low_lr_long_probe]
        # V198 keeps V193's target/sampler fixed and changes only LR via the
        # trainer to bracket optimization stability on the second GPU.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_binary_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCompactV199":
        # [AUDIT][Risk:High][Scope:v199_core_seed_topology_compactness]
        # V197 final improved support/contact recall but hard-case core seeds
        # over-fragmented single GT fragments. V199 keeps the dense barrier head
        # and sampler fixed, changing only semantic core compactness pressure.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreRecallV200":
        # [AUDIT][Risk:High][Scope:v200_core_seed_coverage]
        # V197 final still missed core seeds for some hard-case fragments. V200
        # keeps the dense barrier head and sampler fixed, changing only semantic
        # core recall pressure to bracket V199's compactness bias.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_recall_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBarrierContrastV205":
        # [AUDIT][Risk:High][Scope:v205_contact_calibration]
        # V197 threshold sweep showed decoder-only threshold changes trade
        # contact recall against severe contact mass. V205 keeps the six-channel
        # scaffold and adds only class2-vs-shell contrast to sharpen barriers.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_barrier_contrast_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallSoftDenseCandidateBarrierContrastV206":
        # [AUDIT][Risk:High][Scope:v206_soft_contact_calibration_bracket]
        # V206 uses the same contrast terms as V205 but brackets V197's hard
        # adjacent-shell aux target with the earlier soft-dense target contract.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_soft_dense_candidate_barrier_contrast_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCoverageOpenV209":
        # [AUDIT][Risk:High][Scope:v209_core_coverage]
        # V207/V208 decoder-only soft peaks showed V197 core probabilities are
        # too sparse for multi-fragment coverage. V209 keeps contact/barrier
        # fixed and opens only the semantic core coverage pressure.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_coverage_open_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCoverageGuardedV210":
        # [AUDIT][Risk:High][Scope:v210_core_coverage_guard]
        # V210 pairs V209 with one weak false-core guard to separate useful core
        # coverage from seed overgrowth while keeping the contact loss unchanged.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_coverage_guarded_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateInstanceCoreGuardV213":
        # [AUDIT][Risk:High][Scope:v213_instance_core_topology]
        # V211/V212 showed seedless support is not the dominant failure. V213
        # keeps V197's six-channel output and adds only instance-sidecar core
        # mass/bridge guards to reduce multi-fragment semantic core markers.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_instance_core_guard_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateInstanceCoreStrongGuardV214":
        # [AUDIT][Risk:High][Scope:v214_strong_instance_core_topology]
        # V214 is paired with V213 on the second GPU. It tightens only the
        # per-fragment core mass/bridge penalty to bracket over-suppression.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_instance_core_strong_guard_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateBalancedContactV215":
        # [AUDIT][Risk:High][Scope:v215_contact_calibration]
        # V197 contact thresholding is either sparse or over-massive, while
        # V205/V206 contrast can erase contact. V215 changes only the aux
        # barrier calibration with fixed original-class2 mass bounds.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_balanced_contact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateFragmentPeakCoreV216":
        # [AUDIT][Risk:High][Scope:v216_seed_topology]
        # V213/V214 stronger core mass/bridge guards lowered coverage. V216
        # isolates topology by using only a weak per-fragment core peak/gap
        # signal while keeping the V197 six-channel decoder contract.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_fragment_peak_core_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateWeakFragmentMassCoreV217":
        # [AUDIT][Risk:High][Scope:v217_seed_component_control]
        # V197's remaining hard-case blocker is core seed topology: contact
        # ratio is near 1.0, but seed components are too fragmented. V217 keeps
        # V197 contact/support loss fixed and adds only weak per-fragment core
        # mass pressure, with V213/V214 bridge penalties disabled.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_weak_fragment_mass_core_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateLooseFragmentMassCoreV218":
        # [AUDIT][Risk:High][Scope:v218_seed_component_control_bracket]
        # V218 is the looser V217 bracket. If V217 suppresses core coverage or
        # contact/support, this run separates useful fragment-mass control from
        # over-regularization without changing split, decoder, or output schema.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_loose_fragment_mass_core_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025WeakPeakCompactV219":
        # [AUDIT][Risk:High][Scope:v219_marker_shaping]
        # Decoder diagnostics found V197 core_threshold=0.25 improved IoU-F but
        # not topology. V219 keeps the six-channel V197 scaffold and tests only
        # weak low-threshold core peak compactness; mass/bridge guards remain off.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_weak_peak_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakCompactV220":
        # [AUDIT][Risk:High][Scope:v220_marker_shaping]
        # V220 is the paired stronger bracket for V219. The only intended
        # variable is compactness/purity strength under the same V197 output
        # schema, sampler, split, and decoder evaluation path.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakBalancedContactV241":
        # [AUDIT][Risk:High][Scope:v241_contact_calibration]
        # V220 has the strongest validation IoU-F signal but contact is a cliff:
        # threshold 0.5 erases recall and threshold 0.25 overpaints >19x mass.
        # V241 keeps V220 compactness and changes only bounded contact calibration.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_balanced_contact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakTightBalancedContactV242":
        # [AUDIT][Risk:High][Scope:v242_contact_calibration_bracket]
        # V242 is the paired tighter bracket for V241. Split, sampler, decoder,
        # output channels, and V220 compactness remain fixed.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_tight_balanced_contact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakSoftContactRidgeV243":
        # [AUDIT][Risk:High][Scope:v243_contact_calibration]
        # V241/V242 kept hard dense aux contact targets and added scalar mass
        # pressure, but the measured output still had a threshold cliff. V243
        # keeps V220 compactness and changes only the aux target to soft logit
        # regression inside the class2/class3 candidate band.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_soft_contact_ridge_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakPrecisionSoftContactRidgeV244":
        # [AUDIT][Risk:High][Scope:v244_contact_precision_bracket]
        # V244 is the paired precision-biased soft-target bracket for V243.
        # Split, sampler, decoder, output channels, and V220 core compactness
        # remain fixed.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_precision_soft_contact_ridge_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakContactShellContrastV245":
        # [AUDIT][Risk:High][Scope:v245_contact_shell_separation]
        # Calibration probe showed label2 and label3 aux probabilities overlap
        # in V220/V241/V242, while V243/V244 suppress both. V245 keeps V220
        # compactness and changes only local class2-vs-class3 separation.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_contact_shell_contrast_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakStrongContactShellContrastV246":
        # [AUDIT][Risk:High][Scope:v246_contact_shell_separation_bracket]
        # V246 is the stronger shell-suppression bracket for V245. It keeps
        # split, sampler, decoder, output channels, and V220 compactness fixed.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_strong_contact_shell_contrast_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakDecoderContactV247":
        # [AUDIT][Risk:High][Scope:v247_decoder_contract]
        # V247은 scalar bracket 반복을 멈추고 train/eval contract를 고정한다.
        # channel5 positive는 BFV3 label2(thin_contact_ridge)로 고정하고, 최초
        # label3-only 계획에서 누락됐던 BFV3 label1/3/4 visible negative mass를
        # 같이 닫는다. 10e20i hard-case audit에서 V5 label4까지 positive에 합친
        # hybrid target은 recall은 올렸지만 ratio 4.5730, IoU-F 0.3266으로 실패했다.
        # 따라서 기본 경로는 측정상 더 나은 BFV3-only mass-budget contract다.
        # CE weight는 V220 경로에 남긴다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_decoder_contact_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakXYZAffinityV248":
        # [AUDIT][Risk:High][Scope:v248_affinity_contract]
        # V248은 V247 contact-head 실패 뒤의 representation pivot이다. 출력은
        # 8채널로 바뀌지만 0..4 BFV3 semantic scaffold와 sampler는 유지하고,
        # channel5..7만 z/y/x same-fragment affinity로 바꾼다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_xyz_affinity_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakAffinity13SeedHealedV253":
        # [AUDIT][Risk:High][Scope:v253_oracle_passing_contract]
        # V253은 검색 근거와 oracle gate를 모두 통과한 첫 Task1 proxy contract다.
        # channel5..17은 26-neighborhood의 undirected half인 13개 affinity이고,
        # decoder는 seedless same-fragment island를 가장 가까운 seed에 붙인다.
        # 여기서 loss/oversample/default를 고정해 두지 않으면 다시 contact scalar
        # bracket 또는 V248 3-axis affinity로 잘못 돌아갈 수 있다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_affinity13_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakAffinity13ContactHardNegativeSeedHealedV254":
        # [AUDIT][Risk:High][Scope:v254_hard_negative_affinity]
        # V254는 V253 output/decoder/sampler를 유지하고, different-fragment/contact
        # hard negative affinity를 0.50 아래로 밀어내는 loss contract만 바꾼다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_affinity13_contact_hard_negative_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedHealedV255":
        # [AUDIT][Risk:High][Scope:v255_mutex_affinity]
        # V255는 fulltrain gate가 보는 Merge/Topology 실패를 직접 겨냥한다.
        # same-fragment attractive affinity와 different-fragment/contact repulsive
        # mutex veto를 분리하고, sampler/CE profile은 V253/V254와 동일하게 고정한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_mutex13_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SupportLeakGuardSeedHealedV256":
        # [AUDIT][Risk:High][Scope:v256_support_leak_guard]
        # V256은 V255 output/decoder를 유지하고, learned hard3에서 확인된
        # false-support edge join 문제만 loss contract로 추가한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_mutex13_support_leak_guard_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedStableSupportLeakGuardSeedHealedV257":
        # [AUDIT][Risk:High][Scope:v257_seed_stability]
        # V257은 V256 output/decoder/support-leak을 유지하고, channel4 seed_body가
        # 0.50 threshold에서 0개 또는 대량 false seed로 흔들리는 문제만 고정한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_mutex13_seed_stable_support_leak_guard_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedPeakSupportLeakGuardSeedHealedV258":
        # [AUDIT][Risk:High][Scope:v258_seed_peak_contract]
        # V258은 V256 mutex/support-leak contract를 유지하고, Task1 Instance/Topology
        # gate가 직접 보는 marker seed만 channel3 local peak contract로 바꾼다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_mutex13_seed_peak_support_leak_guard_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13SeedPeakBodySupportLeakGuardSeedHealedV259":
        # [AUDIT][Risk:High][Scope:v259_seed_peak_body_target]
        # V259은 V258 decoder를 유지하되 channel3 학습 target을 one-voxel center에서
        # compact seed_body로 바꿔 top1 fallback 없는 marker를 만들 수 있는지 검증한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_mutex13_seed_peak_body_support_leak_guard_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakMutex13FractureSeedCalibratedSupportLeakGuardSeedHealedV260":
        # [AUDIT][Risk:High][Scope:v260_fracture_seed_calibration]
        # V260은 V259 support geometry를 유지하고, Task1 실패축인 Fracture Dice,
        # Instance Recall/Merge/Topology가 직접 보는 channel2 barrier와 channel3
        # per-fragment seed peak calibration만 바꾼다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_mutex13_fracture_seed_calibrated_support_leak_guard_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakPairwiseSoftmaxMutex13SeedHealedV266":
        # [AUDIT][Risk:Blocker][Scope:v266_pairwise_partition]
        # V265는 support geometry는 살렸지만 center heatmap false peak로 split/topology가
        # 실패했다. V266은 oracle-pass affinity graph family로 돌아가되, V255-V260의
        # independent sigmoid mutex 붕괴를 막기 위해 edge를 join-vs-cut relative logit
        # decision으로 학습한다. fulltrain 판단은 여전히 Task1 proxy와 시각화 gate다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_pairwise_softmax_mutex13_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedV268":
        # [AUDIT][Risk:Blocker][Scope:v268_no_contact_graph_partition]
        # V267 forensic에서 support는 살아났지만 center/offset identity가 학습되지 않았다.
        # V268은 contact/fracture head 없이 same-instance join/cut graph를 직접 학습한다.
        # fulltrain 판단은 hard3 Task1 proxy와 시각화 gate에 계속 고정한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_mutex13_seed_healed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxSeedHealedV273":
        # [AUDIT][Risk:Blocker][Scope:v273_fulltrain_gate]
        # V273은 hard3 oracle과 visual audit를 통과한 no-contact boundary-exact
        # pairwise contract다. fulltrain 전에는 true overfit003과 hard3 short를
        # 같은 Task1 proxy로 통과해야 한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_seed_healed_v273_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactPairwiseSoftmaxSeedCalibratedV274":
        # [AUDIT][Risk:Blocker][Scope:v274_seed_anchor_calibration]
        # V273 learned true-overfit는 support는 맞췄지만 seed가 support 전체로 번져
        # raw component 50k+ split failure를 만들었다. V274는 contact 없이 같은
        # 28채널 decoder를 유지하고 seed sparse anchor만 검증한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_seed_calibrated_v274_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactGlobalCoordPairwiseV284":
        # [AUDIT][Risk:Blocker][Scope:v284_pairwise_topology]
        # V283은 support/full-volume coordinate 문제를 줄였지만 LeftHip partition이
        # GT-support에서도 실패했다. V284는 oracle-pass V273 join/cut graph를 유지하고
        # V283의 full-ROI coordinate input만 추가한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_pairwise_v284_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactFragmentPositionV275":
        # [AUDIT][Risk:Blocker][Scope:v275_contact_removed]
        # V274도 seed/pairwise graph learned path에서 split 폭발을 막지 못했다.
        # V275는 contact/seed/pairwise를 폐기하고 51-way anatomy-local fragment
        # position class를 직접 학습한다. fulltrain은 Task1 proxy gate 통과 전 금지다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_fragment_position_v275_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactFragmentPositionDiceV276":
        # [AUDIT][Risk:Blocker][Scope:v276_rare_fragment_class_collapse]
        # V275 true-overfit003는 support/Dice는 통과했지만 LeftHip rare local classes를
        # class 1/3으로 merge했다. V276은 decoder와 output을 유지하고 present class별
        # Dice만 추가해 class collapse를 검증한다. contact는 계속 사용하지 않는다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_fragment_position_dice_v276_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorGapV277":
        # [AUDIT][Risk:Blocker][Scope:v277_contact_removed]
        # V247-V276은 contact scalar 또는 arbitrary fragment-ID 분류에서 0.90+ Task1
        # proxy gate에 도달하지 못했다. V277은 active path에서 contact head를 버리고,
        # connected-component decoder가 직접 쓰는 support/separator-gap 2채널만 학습한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_separator_gap_v277_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorEnergyV278":
        # [AUDIT][Risk:Blocker][Scope:v278_dense_separator_energy]
        # V277 true-overfit003는 support를 배웠지만 separator를 0 voxel로 collapse했다.
        # V278은 같은 2채널 decoder를 유지하면서 separator를 희소 BCE가 아닌 dense
        # energy regression으로 바꿔 학습 가능성 하나만 검증한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_separator_energy_v278_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSeparatorSoftmaxV287":
        # [AUDIT][Risk:Blocker][Scope:v287_separator_softmax]
        # V277/V278 threshold forensic에서 separator는 0.50/0.35/0.25에서 모두
        # collapse했고 0.15에서는 over-split됐다. V287은 contact를 계속 버린 채,
        # support body와 separator를 independent sigmoid가 아니라 mutually-exclusive
        # softmax class로 학습해 separator collapse 하나만 검증한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_separator_softmax_v287_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV288":
        # [AUDIT][Risk:Blocker][Scope:v288_abbc_pengwin_2024_1st]
        # V287은 body 안에서 CC을 못 끊었고 separator는 thin 1-voxel band였다.
        # V288은 PENGWIN 2024 CT 1st (MIC-DKFZ) ABBC representation을 채택해
        # background/border/boundary/core 4-class를 학습한다. core가 decoder
        # watershed seed가 되고, boundary는 nearest seed로 흡수된다.
        # voxel-level oracle (Phase 1) Dice 1.0 / HD95 0 PASS 후 진입.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v288_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCSDFV289":
        # [AUDIT][Risk:Blocker][Scope:v289_boundary_sdf_redesign]
        # V288 50e hard-trio Instance F1는 0.5317에서 멈췄고 boundary head precision/recall은
        # 0.066/0.21이었다. V289는 boundary class softmax를 channel 2 SDF regression으로 바꿔
        # fragment-pair surface localization 자체를 회귀로 학습한다. 나머지 ABBC 구조와
        # 후처리 (anatomy_size_ratio_keep)는 V288과 동일하다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_sdf_v289_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCSDFFDMV290":
        # [AUDIT][Risk:Blocker][Scope:v290_fdm_weighted_sdf]
        # V289 50e validation barrier_pred_target_ratio가 0.0이 됐다 (model이 SDF<2.0 영역을
        # 못 잡음). V290은 SDF SmoothL1에 fracture-distance weighting을 추가해 boundary-proximal
        # voxel에 더 큰 gradient를 흘려준다. output/decoder는 V289와 동일하다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_sdf_fdm_v290_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV291":
        # [AUDIT][Risk:Blocker][Scope:v291_boundary_class_weight]
        # V288 50e patch boundary precision/recall은 0.066/0.21로 낮았다. V288 ABBC
        # representation은 그대로 두고 boundary class CE+Dice loss term에만 5x weight를
        # 곱해 sparse boundary class의 학습을 강제한다. SDF/EDT가 없어 V288과 동일
        # throughput을 유지한다 (~38 sec/epoch). decoder/eval은 V288 경로를 재사용한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v291_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakCenterFlowV261":
        # [AUDIT][Risk:High][Scope:v261_task1_proxy_gate]
        # V261은 IoU-F를 올리기 위한 bracket이 아니라, 사용자가 준 Task1 지표
        # Dice/Fracture Dice/Local Dice/HD95/ASSD/Instance/Merge/Split/Topology에
        # 직접 대응하는 dense center-flow decoder contract를 학습 가능한지 검증한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_center_flow_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakCenterPeakFlowV262":
        # [AUDIT][Risk:High][Scope:v262_task1_proxy_gate]
        # V262는 V261의 8채널 center-flow 구조를 유지하되 channel3을 binary component가
        # 아니라 peak-NMS용 center peak/rank/mass contract로 학습한다. fulltrain 판단은
        # IoU-F가 아니라 Task1 official-aligned proxy와 시각화 gate에 고정한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_center_peak_flow_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakCenterPeakFlowCalibratedV263":
        # [AUDIT][Risk:High][Scope:v263_task1_proxy_gate]
        # V263은 V262 forensic에서 확인된 support overpaint와 fracture label2/non2
        # inversion을 직접 고정한다. fulltrain 판단은 hard3 Task1 proxy와 시각화가
        # 먼저 통과해야 하며, IoU-F는 보조 진단으로만 남긴다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_center_peak_flow_calibrated_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakDenseCenterHeatmapFlowCalibratedV264":
        # [AUDIT][Risk:Blocker][Scope:v264_task1_proxy_gate]
        # V263은 support가 맞아도 center peak가 완전히 collapse되어 empty prediction이
        # 나왔다. V264는 변경 변수를 center target density로 제한하고, fulltrain 판단은
        # hard3 Task1 proxy와 시각화 gate에 계속 고정한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_dense_center_heatmap_flow_calibrated_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakTopologyConstrainedCenterHeatmapFlowCalibratedV265":
        # [AUDIT][Risk:Blocker][Scope:v265_task1_proxy_gate]
        # V264는 support geometry를 회복했지만 false center marker가 과다해 Split/Topology가
        # 실패했다. V265는 변경 변수를 fragment-count constrained marker topology로
        # 제한하고, fulltrain 판단은 hard3 Task1 proxy와 시각화에 고정한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_topology_constrained_center_heatmap_flow_calibrated_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactCenterHeatmapFlowV267":
        # [AUDIT][Risk:Blocker][Scope:v267_contact_rejection]
        # V247-V266은 contact/fracture/contact-edge 구조를 바꿔도 Task1 proxy gate를
        # 통과하지 못했다. V267은 contact를 완전히 제거하고 support/center/offset만
        # 학습한다. fulltrain 판단은 hard3 Task1 proxy와 시각화에 고정한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_center_heatmap_flow_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSpatialEmbeddingV270":
        # [AUDIT][Risk:Blocker][Scope:v270_spatial_embedding_gate]
        # V270은 oracle-pass contact-free dense spatial embedding contract다. Fulltrain
        # 판단은 learned hard3 Task1 proxy와 visual audit에 고정한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_spatial_embedding_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactSpatialEmbeddingContrastiveV271":
        # [AUDIT][Risk:Blocker][Scope:v271_fragment_identity_gate]
        # V271은 V270의 4채널 decoder contract를 유지하고, small-fragment merge를 막기
        # 위해 fragment-balanced pull/push metric loss만 추가한다. fulltrain 판단은
        # hard3 Task1 proxy와 시각화 gate에 계속 고정한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_spatial_embedding_contrastive_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactCoordSpatialEmbeddingV279":
        # [AUDIT][Risk:Blocker][Scope:v279_contact_removed_coordconv]
        # Contact/separator scalar 계열은 true-overfit에서 반복적으로 붕괴했다. V279는
        # V270/V271 decoder와 Task1 proxy gate를 유지하고, 모델 입력에 deterministic
        # z/y/x 좌표만 추가해 fragment sink 좌표를 학습 가능한 표현으로 만드는지 검증한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_coord_spatial_embedding_v279_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactQueryMaskV280":
        # [AUDIT][Risk:Blocker][Scope:v280_query_mask]
        # V279 LeftHip-only isolation에서도 sink embedding이 한 점으로 collapse했다. V280은
        # seed/query-conditioned binary mask를 직접 학습해 fragment identity를 seed proposal
        # 문제와 분리한다. fulltrain 판단은 oracle-query Task1 proxy와 시각화 후에만 한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_query_mask_v280_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactQueryMaskPositiveNegativeV281":
        # [AUDIT][Risk:Blocker][Scope:v281_positive_negative_query_mask]
        # V280은 positive query만으로 fragment count는 맞췄지만 GT-support argmax IoU-F도
        # 낮았다. V281은 contact를 계속 배제하고 positive/negative query map만 추가해
        # fragment identity 신호가 학습 가능한지 분리 검증한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_query_mask_pn_v281_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactGlobalCoordQueryMaskV285":
        # [AUDIT][Risk:Blocker][Scope:v285_global_coord_query_mask]
        # V284의 no-contact pairwise graph도 fracture fragments를 main anatomy로
        # 흡수했다. V285는 oracle query seed가 주어졌을 때 fragment mask 자체를 학습할 수
        # 있는지 검증하며, V283에서 확인된 full-ROI coordinate contract를 적용한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_query_mask_v285_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactGlobalCoordQueryMaskPositiveNegativeV286":
        # [AUDIT][Risk:Blocker][Scope:v286_global_coord_positive_negative_query]
        # V285 final에서 small fragment query mask가 주변 anatomy로 과확장됐다.
        # V286은 contact 없이 positive/negative query maps와 full-ROI 좌표만으로
        # 그 과확장을 줄일 수 있는지 검증한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_query_mask_pn_v286_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactFreeEmbeddingV282":
        # [AUDIT][Risk:Blocker][Scope:v282_free_embedding_partition]
        # center/Voronoi oracle과 V280/V281 query-mask가 모두 0.90+ gate에서 멀었다.
        # V282는 support segmentation을 보존하고, contact 없이 support 내부 partition만
        # free discriminative embedding으로 학습 가능한지 검증한다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_free_embedding_v282_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactGlobalCoordFreeEmbeddingV283":
        # [AUDIT][Risk:Blocker][Scope:v283_global_coord_embedding]
        # V282 patch diagnostic은 높았지만 full-volume은 실패했다. V283은 contact를
        # 계속 배제하고, CoordConv tile-local 좌표 대신 full-ROI global coordinate를
        # 입력 채널로 넣어 train crop과 sliding-window inference의 좌표계를 맞춘다.
        return (
            loss_profile or "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_free_embedding_v283_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakCompactV221":
        # [AUDIT][Risk:High][Scope:v221_marker_shaping]
        # V221 brackets V219/V220 compactness strength while keeping contact
        # calibration untouched. This tests whether V220's topology gain survives
        # at a weaker setting before adding any contact-preservation pressure.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025ContactPreservePeakCompactV222":
        # [AUDIT][Risk:High][Scope:v222_contact_preserve_marker_shaping]
        # V222 keeps V220-like compactness and changes only aux contact pressure.
        # It directly tests whether the topology signal can be retained without
        # the measured V220 fixed-threshold contact recall collapse.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_contact_preserve_peak_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakCoverageCompactV223":
        # [AUDIT][Risk:High][Scope:v223_core_coverage]
        # V221 improved topology/multi-fragment seed count but left core coverage
        # low. V223 changes only the core island floor inside GT core voxels.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_coverage_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025OriginalContactMidPeakCompactV224":
        # [AUDIT][Risk:High][Scope:v224_contact_target]
        # V221 over-predicts contact at threshold 0.5. V224 keeps V221 core
        # compactness and removes only the adjacent-shell positive expansion
        # from the aux contact target.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_original_contact_mid_peak_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakContactCapCompactV226":
        # [AUDIT][Risk:High][Scope:v226_contact_overmass]
        # Decoder-only V225 did not preserve V221 topology. V226 keeps V221
        # compactness and adds only a weak one-sided contact mass cap, with no
        # contact floor or positive pressure.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_contact_cap_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakTightContactCapCompactV227":
        # [AUDIT][Risk:High][Scope:v227_contact_overmass_bracket]
        # V227 is the tighter paired bracket for V226. The only changed variable
        # is cap strength; topology compactness, sampler, split, and output
        # contract remain identical.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_tight_contact_cap_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakSupportSeedCompactV230":
        # [AUDIT][Risk:High][Scope:v230_seedless_support]
        # V226/V227 proved contact caps suppress contact recall. V230 returns to
        # V221 and changes only seed presence inside GT support fragments.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_support_seed_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025MidPeakConservativeCoreTailCompactV231":
        # [AUDIT][Risk:High][Scope:v231_core_tail_floor]
        # V223's core-island floor regressed topology. V231 keeps the same
        # class-4-only idea but weakens it to test over-pressure as the cause.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_conservative_coretail_compact_barrier_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateSeedHeadV201":
        # [AUDIT][Risk:High][Scope:v201_fragment_seed_head]
        # V201 keeps V197's dense barrier scaffold and adds only a fragment-scale
        # seed head trained from instance sidecars. This directly tests whether
        # topology should be separated from semantic class-4 core mass.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_seed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCompactSeedHeadV202":
        # [AUDIT][Risk:High][Scope:v202_fragment_seed_compact_bracket]
        # V202 adds the same seed head as V201 but keeps V199's compact semantic
        # core pressure to bracket whether semantic-core compactness helps or
        # hurts fragment-scale seed topology.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_compact_seed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateSparseSeedHeadV203":
        # [AUDIT][Risk:High][Scope:v203_sparse_seed_head]
        # Decoder-only local-max did not recover topology from V201/V202 broad
        # seed responses. V203 keeps the V201 scaffold and changes only the
        # seed-head loss toward sparse mass control.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_sparse_seed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCoreCompactSparseSeedHeadV204":
        # [AUDIT][Risk:High][Scope:v204_sparse_seed_compact_bracket]
        # V204 pairs V203 on the second GPU while preserving V202's compact
        # semantic-core branch. The only shared change from V201/V202 is sparse
        # seed mass control.
        return (
            loss_profile or "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_compact_sparse_seed_head",
            ce_class_weights or "off",
            oversample_profile or "boundary_fragment_v3_sidecar_core_recall_high_edge",
        )
    if trainer == "PengwinTrainerBICMContactV22":
        return (
            loss_profile or "bicm_v22_core_marker_separation",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV23":
        return (
            loss_profile or "bicm_v23_staged_core_marker",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV24":
        return (
            loss_profile or "bicm_v24_contact_preserved_core_marker",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV25":
        return (
            loss_profile or "bicm_v25_coupled_contact_core_marker",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV26":
        return (
            loss_profile or "bicm_v26_contact_tolerant_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV27":
        return (
            loss_profile or "bicm_v26_contact_tolerant_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV28":
        return (
            loss_profile or "bicm_v28_isolated_contact_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV29":
        return (
            loss_profile or "bicm_v29_gentle_isolated_contact_precision",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV30":
        return (
            loss_profile or "bicm_v30_soft_contact_ridge",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV31":
        return (
            loss_profile or "bicm_v31_local_contrastive_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV32":
        return (
            loss_profile or "bicm_v32_memory_efficient_local_contrast",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV34":
        return (
            loss_profile or "bicm_v34_compact_core_marker",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV35":
        return (
            loss_profile or "bicm_v35_contact_precision_compact_core",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV36":
        return (
            loss_profile or "bicm_v36_staged_contact_precision_compact_core",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMContactV37":
        return (
            loss_profile or "bicm_v37_asymmetric_contact_compact_core",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgeAffinityV38":
        return (
            loss_profile or "bicm_v38_edge_affinity",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgePrimaryV39":
        return (
            loss_profile or "bicm_v39_edge_primary",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgeCoreV40":
        return (
            loss_profile or "bicm_v40_edge_core_separation",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMInstanceCoreV41":
        return (
            loss_profile or "bicm_v41_instance_core_edge",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMInstanceCoreV42":
        return (
            loss_profile or "bicm_v42_instance_core_edge_primary",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgeContactV43":
        return (
            loss_profile or "bicm_v43_edge_contact_viability",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgeLocalRankV44":
        return (
            loss_profile or "bicm_v44_edge_local_rank",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgeCandidateV45":
        return (
            loss_profile or "bicm_v45_edge_candidate_save",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMEdgeCurriculumV46":
        return (
            loss_profile or "bicm_v46_staged_edge_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSeparatedContactV47":
        return (
            loss_profile or "bicm_v47_separated_contact_head",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSupportAwareContactV48":
        return (
            loss_profile or "bicm_v48_support_aware_contact_branch",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDecoderFeatureContactV49":
        return (
            loss_profile or "bicm_v49_decoder_feature_contact_branch",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDecoderFeaturePhaseV50":
        return (
            loss_profile or "bicm_v50_decoder_feature_contact_phase",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDenseEdgeCostV51":
        return (
            loss_profile or "bicm_v51_dense_edge_cost",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDenseEdgeCoreV52":
        return (
            loss_profile or "bicm_v52_dense_edge_core_topology",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMFragmentMarkerCoreV53":
        return (
            loss_profile or "bicm_v53_fragment_marker_core",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMSparseHeadBalancedV54":
        return (
            loss_profile or "bicm_v54_sparse_head_balanced",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCalibratedSparseHeadV55":
        return (
            loss_profile or "bicm_v55_calibrated_sparse_head",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMPhasedMarkerContactV56":
        return (
            loss_profile or "bicm_v56_phased_marker_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMDecoderFeatureMarkerPhaseV57":
        return (
            loss_profile or "bicm_v57_decoder_feature_marker_phase",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMHeatmapMarkerContactV58":
        return (
            loss_profile or "bicm_v58_heatmap_marker_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMCorePreservingContactV59":
        return (
            loss_profile or "bicm_v59_core_preserving_contact",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMStrictCoreTopologyV60":
        return (
            loss_profile or "bicm_v60_strict_core_topology",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    if trainer == "PengwinTrainerBICMPeakSeedV61":
        return (
            loss_profile or "bicm_v61_peak_seed",
            ce_class_weights or "off",
            oversample_profile or "bicm_v11_contact_roi",
        )
    return (
        loss_profile or "dc_ce",
        ce_class_weights or "off",
        oversample_profile or "default",
    )


def _dataset_input_channels(ds_id: int) -> int:
    dataset_json = Path(os.environ.get("nnUNet_raw", str(Path("/workspace/nnunet/raw")))) / DATASETS[ds_id]["name"] / "dataset.json"
    if not dataset_json.exists():
        return 1
    try:
        payload = json.loads(dataset_json.read_text())
        return len(payload.get("channel_names", {"0": "CT"}))
    except Exception:
        return 1


def _pelvic_foundation_init_path(target_input_channels: int = 1) -> Path | None:
    """Return the preferred Dataset532 checkpoint for explicit warm-start tests."""
    if PELVIC_FOUNDATION_DEFAULT.exists():
        return _ensure_weights_only_pretrained(PELVIC_FOUNDATION_DEFAULT, target_input_channels=target_input_channels)
    fallback = Path(PRETRAINED_DEFAULT)
    if fallback.exists():
        return _ensure_weights_only_pretrained(fallback, target_input_channels=target_input_channels)
    return None


def _ensure_weights_only_pretrained(src: Path, target_input_channels: int = 1) -> Path:
    """Create a PyTorch-2.6-safe pretrained file containing network weights only.

    nnU-Net 2.5.x calls `torch.load(fname)` in its pretrained loader. In recent
    PyTorch versions that defaults to `weights_only=True`, which rejects older
    nnU-Net checkpoints carrying numpy scalars in optimizer/logger metadata.
    The local Dataset532 checkpoint is trusted, so we load it once with
    `weights_only=False`, strip everything except `network_weights`, and save a
    minimal tensor-only file that nnU-Net can safely consume.
    """
    target_input_channels = int(target_input_channels)
    out_path = (
        PELVIC_FOUNDATION_WEIGHTS_ONLY
        if target_input_channels == 1
        else RESULT_WEIGHT / f"pretrained_ds532_fold0_network_weights_only_in{target_input_channels}.pth"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        out_path.exists()
        and out_path.stat().st_mtime >= src.stat().st_mtime
    ):
        return out_path
    import torch

    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)
    if "network_weights" not in ckpt:
        raise KeyError(f"pretrained checkpoint missing network_weights: {src}")
    weights = dict(ckpt["network_weights"])
    if target_input_channels > 1:
        # nnU-Net stores the first encoder convolution twice for this network
        # wrapper (`encoder.*` and `decoder.encoder.*`). Expand every matching
        # 1-channel stem tensor so the official loader can keep its strict
        # all-non-head-shapes-must-match contract.
        for key, tensor in list(weights.items()):
            if (
                not key.endswith(("stem.convs.0.conv.weight", "stem.convs.0.all_modules.0.weight"))
                or getattr(tensor, "ndim", 0) != 5
                or int(tensor.shape[1]) != 1
            ):
                continue
            expanded = tensor.new_zeros((tensor.shape[0], target_input_channels, *tensor.shape[2:]))
            expanded[:, 0:1] = tensor
            expanded[:, 1:] = tensor.mean(dim=1, keepdim=True) * 0.05
            weights[key] = expanded
    torch.save({"network_weights": weights}, str(out_path))
    return out_path


# =============================================================================
# train — wraps nnUNetv2_train CLI
# =============================================================================
def train(ds_id: int, fold: int = 0, gpu: int = 0,
          plans: str = "nnUNetResEncUNetLPlans",
          config: str = "3d_fullres",
          init_weights: str | None = None,
          full_init_network: bool = False,
          trainer: str | None = None,
          loss_profile: str | None = None,
          ce_class_weights: str | None = None,
          oversample_profile: str | None = None,
          env_profile: str = "speed",
          continue_training: bool = False,
          tag: str | None = None,
          extra_env: dict | None = None,
          background: bool = False) -> int:
    """Launch nnUNetv2_train.

    - Foreground is the default so this process waits and reaps nnU-Net cleanly.
      That matters in this container because PID 1 is `sleep`, not an init
      reaper, so detached orphan training processes can remain as zombies after
      completion.
    - SPEED_ENV (torch.compile, n_proc_DA=16, expandable_segments, OMP=1) is
      applied unless overridden via `extra_env`.
    - GPU selection by setting CUDA_VISIBLE_DEVICES (rather than
      `--device cuda:N`) so nnUNet sees device as cuda:0 internally.
    - `--background` keeps the old detached behavior for manual use, but the
      foreground mode is preferred for long runs that should tail cleanly.

    Args:
        ds_id: active dataset ID in the current registry (532 or 533).
        fold: 0..4 for 5-fold cross-val.
        gpu: CUDA index (0 or 1).
        plans: nnUNet plans file. Default ResEncL = SOTA backbone.
        config: 3d_fullres | 3d_lowres | 2d. Default 3d_fullres for full quality.
        init_weights: pretrained .pth path; None = train from scratch.
        full_init_network: include segmentation heads in pretrained transfer.
            Required for same-architecture refinement runs such as V94.
        extra_env: extra environment vars to override SPEED_ENV defaults.
        trainer: active custom nnU-Net trainer.
        loss_profile: explicit weak-class/boundary ablation from LOSS_PROFILES.
            None lets the selected trainer choose its default.
        ce_class_weights: off | auto. Auto computes clipped median-frequency
            CE weights from rebuilt labels and writes an audit JSON.
        oversample_profile: default | weak0123. None follows selected trainer.
        env_profile: speed | stable | single. Stable disables torch.compile and
            lowers data augmentation workers for memory-constrained resumes.
            Single disables augmentation multiprocessing for tiny overfit/debug
            runs where unpacked sidecars must be read deterministically.
        continue_training: pass nnU-Net `--c` and resume optimizer/EMA from
            checkpoint_latest.pth. Use this only when latest exists.

    Returns: process PID.
    """
    if env_profile not in ENV_PROFILES:
        raise ValueError(f"env_profile must be one of {sorted(ENV_PROFILES)}")
    cfg = DATASETS[ds_id]
    trainer = trainer or cfg["trainer"]
    if trainer not in TRAINERS:
        raise ValueError(f"trainer must be one of {TRAINERS}")
    loss_profile, ce_class_weights, oversample_profile = _resolve_training_profiles(
        trainer, loss_profile, ce_class_weights, oversample_profile
    )
    if loss_profile not in LOSS_PROFILES:
        raise ValueError(f"loss_profile must be one of {LOSS_PROFILES}")
    if ce_class_weights not in CE_WEIGHT_PROFILES:
        raise ValueError(f"ce_class_weights must be one of {CE_WEIGHT_PROFILES}")
    if oversample_profile not in OVERSAMPLE_PROFILES:
        raise ValueError(f"oversample_profile must be one of {OVERSAMPLE_PROFILES}")
    cmd = [
        _nnunet_train_bin(),
        str(ds_id), config, str(fold),
        "-tr", trainer, "-p", plans,
    ]
    if continue_training:
        latest = NN_RES / cfg["name"] / f"{trainer}__{plans}__{config}" / f"fold_{fold}" / "checkpoint_latest.pth"
        if not latest.exists():
            raise FileNotFoundError(
                f"--continue-training requires checkpoint_latest.pth, missing: {latest}"
            )
        cmd.append("--c")
        init_weights = None
    if init_weights is None and not continue_training and ds_id in NEED_INIT_FROM_PELVIC:
        init_path = _pelvic_foundation_init_path(target_input_channels=_dataset_input_channels(ds_id))
        if init_path is not None:
            init_weights = str(init_path)
            print(f"[train] auto-init from {init_weights}")
        else:
            print(
                "[train] no Dataset532 checkpoint_best or fallback pretrained "
                f"weights at {PRETRAINED_DEFAULT}, training from scratch"
            )
    if init_weights:
        cmd += ["-pretrained_weights", str(init_weights)]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # [REPRO][Scope:runtime_env]
    # Training profile defaults are explicit and auditable. `speed` is used for
    # normal full training, `stable` for memory-constrained resumes, and
    # `single` for one-case gates where deterministic sidecar reads matter more
    # than throughput. The selected env vars are written into launch JSON below.
    for k, v in ENV_PROFILES[env_profile].items():
        env[k] = v
    if continue_training:
        # PyTorch 2.6 changed the default torch.load behavior to
        # weights_only=True when the caller does not pass an explicit value.
        # nnU-Net resume checkpoints include optimizer/training metadata, so
        # a plain `--c` resume can fail before writing a traceback to the
        # nnU-Net log. Keep this scoped to resume paths; fresh warm-starts use
        # a tensor-only checkpoint generated by _ensure_weights_only_pretrained.
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["PENGWIN_LOSS_PROFILE"] = loss_profile
    env["PENGWIN_CE_CLASS_WEIGHTS"] = ce_class_weights
    env["PENGWIN_OVERSAMPLE_PROFILE"] = oversample_profile
    if init_weights and trainer in {
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
        "PengwinTrainerBICMJointSupportProductV111",
        "PengwinTrainerBICMEdgeGraphAssignmentV112",
        "PengwinTrainerBICMAdaptiveBoundaryProductV113",
        "PengwinTrainerBICMAdaptiveBoundaryProductV113HighLR",
        "PengwinTrainerBICMEncoderAdapterV114",
        "PengwinTrainerBICMSemanticOracleAdapterV115",
        "PengwinTrainerBICMGraphCostSeparatorV116",
        "PengwinTrainerBICMSupportGateSemanticContactV117",
        "PengwinTrainerBICMWarmSupportGateSemanticContactV118",
        "PengwinTrainerBICMAllNetworkAdaptiveBoundaryV119",
        "PengwinTrainerBICMSameFragmentAffinityV120",
        "PengwinTrainerBICMWarmSameFragmentAffinityV121",
        "PengwinTrainerBICMWarmEdgeProductPrecisionV122",
        "PengwinTrainerBICMHighRecallGateCleanupV123",
        "PengwinTrainerBICMSupportConditionedEdgePrecisionV124",
        "PengwinTrainerBICMSaturatedEdgeDenseGateV125",
        "PengwinTrainerBICMLocalAdjacencyProductV126",
        "PengwinTrainerBICMLocalAdjacencyProductV126FromV96",
        "PengwinTrainerBICMEdgeResetLocalAdjacencyProductV127",
        "PengwinTrainerBICMSupportBridgeSuppressionV128",
        "PengwinTrainerBICMAffinitySharpeningV129",
        "PengwinTrainerBICMSupportTopologyRepairV130",
        "PengwinTrainerBICMAllNetworkSupportTopologyRepairV131",
        "PengwinTrainerBICMSupportVetoGateV132",
        "PengwinTrainerBICMSupportVetoSemanticV133",
        "PengwinTrainerBICMSupportVetoGateHighLRV134",
        "PengwinTrainerBICMSupportVetoGateUltraLRV135",
        "PengwinTrainerBICMSupportTopologyHighLRV136",
        "PengwinTrainerBICMSupportTopologyUltraLRV137",
        "PengwinTrainerBICMSupportTopologyMidLRV138",
        "PengwinTrainerBICMSupportTopologyUpperMidLRV139",
        "PengwinTrainerBICMSupportTopologyLowBracketLRV140",
        "PengwinTrainerBICMSupportTopologyHighBracketLRV141",
        "PengwinTrainerBICMSupportTopologyCoreSeedLowLRV142",
        "PengwinTrainerBICMSupportTopologyCoreSeedMidLRV143",
        "PengwinTrainerBICMStrongCoreSeedLowLRV144",
        "PengwinTrainerBICMStrongCoreSeedMidLRV145",
        "PengwinTrainerBICMFragmentSeedPresenceLowLRV146",
        "PengwinTrainerBICMFragmentSeedPresenceMidLRV147",
        "PengwinTrainerBICMCoreHeatmapSeedLowLRV148",
        "PengwinTrainerBICMCoreHeatmapSeedMidLRV149",
        "PengwinTrainerBICMCoreOnlyHeatmapSeedMidLRV150",
        "PengwinTrainerBICMCoreOnlyHeatmapSeedHighLRV151",
        "PengwinTrainerBICMCoreOnlyCenterSeedMidLRV152",
        "PengwinTrainerBICMCoreHeatmapCenterSeedMidLRV153",
        "PengwinTrainerBICMCoreOnlyCenterSeedSamplerV154",
        "PengwinTrainerBICMCoreHeatmapCenterSeedSamplerV155",
        "PengwinTrainerBICMCoreOnlyCenterSeedAffinitySamplerV158",
        "PengwinTrainerBICMCoreHeatmapCenterSeedAffinitySamplerV159",
        "PengwinTrainerBICMCoreOnlyCenterSeedOffsetSamplerV160",
        "PengwinTrainerBICMCoreHeatmapCenterSeedOffsetSamplerV161",
    }:
        # [AUDIT][Risk:High][Scope:warm_start_head_transfer]
        # nnU-Net's generic pretrained loader skips `.seg_layers.` by default.
        # V94+ are same-architecture V89 checkpoint refinements, so skipping
        # the final 10-channel heads leaves calibrated rows random and
        # invalidates any row-preservation claim. Force full transfer here.
        full_init_network = True
    if full_init_network:
        env["PENGWIN_LOAD_PRETRAINED_SEG_LAYERS"] = "1"
    if extra_env:
        env.update(extra_env)

    print(f"[train] Ds{ds_id} {cfg['name']} (fold {fold}) on GPU {gpu}")
    print(
        f"        trainer={trainer} loss_profile={env['PENGWIN_LOSS_PROFILE']} "
        f"ce_weights={env['PENGWIN_CE_CLASS_WEIGHTS']} "
        f"oversample={env['PENGWIN_OVERSAMPLE_PROFILE']} env_profile={env_profile}"
    )
    mode = "background" if background else "foreground"
    print(f"        mode = {mode}")
    if background:
        print("        logs: disabled (status/checkpoints only)")
        print("        WARN: background mode can leave zombies if PID 1 does not reap children")
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        print("        stdout/stderr: inherited; nnU-Net log also stays in the fold directory")
        proc = subprocess.Popen(cmd, env=env)
    print(f"        PID = {proc.pid}")
    launch_tag = tag or f"ds{ds_id}_fold{fold}_{int(time.time())}"
    status_path = RESULT_REPORT / f"train_launch_{launch_tag}.json"
    launch_payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": ds_id,
        "dataset_name": cfg["name"],
        "fold": fold,
        "gpu": gpu,
        "pid": proc.pid,
        "plans": plans,
        "config": config,
        "trainer": trainer,
        "continue_training": bool(continue_training),
        "init_weights": init_weights,
        "full_init_network": bool(full_init_network),
        "loss_profile": env["PENGWIN_LOSS_PROFILE"],
        "ce_class_weights": env["PENGWIN_CE_CLASS_WEIGHTS"],
        "oversample_profile": env["PENGWIN_OVERSAMPLE_PROFILE"],
        "env_profile": env_profile,
        "env_overrides": {k: env.get(k) for k in ENV_PROFILES[env_profile]},
        "mode": mode,
        "stdout_stderr": "DEVNULL" if background else "inherit",
        "logs": "nnU-Net fold training_log_*.txt",
    }
    _write_json(status_path, launch_payload)
    print(f"        launch_json = {status_path}")
    if not background:
        try:
            return_code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                return_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                return_code = proc.wait()
            launch_payload.update({
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "returncode": int(return_code),
                "interrupted": True,
            })
            _write_json(status_path, launch_payload)
            raise
        launch_payload.update({
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "returncode": int(return_code),
        })
        _write_json(status_path, launch_payload)
        if return_code != 0:
            raise SystemExit(return_code)
    return proc.pid


def _start_train_for_queue(ds_id: int, fold: int, gpu: int,
                           plans: str, config: str,
                           trainer: str,
                           loss_profile: str,
                           ce_class_weights: str,
                           oversample_profile: str,
                           env_profile: str,
                           queue_tag: str) -> subprocess.Popen:
    """Start one nnU-Net train process for the managed queue.

    This intentionally keeps a `Popen` handle instead of using `train()`, whose
    detached launch returns only a PID. The queue needs to know exactly when a
    dataset finished before it assigns the next dataset to the freed GPU.
    """
    cfg = DATASETS[ds_id]
    cmd = [
        _nnunet_train_bin(),
        str(ds_id), config, str(fold),
        "-tr", trainer, "-p", plans,
    ]
    init_weights = None
    if ds_id in NEED_INIT_FROM_PELVIC:
        init_path = _pelvic_foundation_init_path(target_input_channels=_dataset_input_channels(ds_id))
        if init_path is not None:
            init_weights = str(init_path)
    if init_weights:
        cmd += ["-pretrained_weights", init_weights]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if env_profile not in ENV_PROFILES:
        raise ValueError(f"env_profile must be one of {sorted(ENV_PROFILES)}")
    for k, v in ENV_PROFILES[env_profile].items():
        env[k] = v
    env["PENGWIN_LOSS_PROFILE"] = loss_profile
    env["PENGWIN_CE_CLASS_WEIGHTS"] = ce_class_weights
    env["PENGWIN_OVERSAMPLE_PROFILE"] = oversample_profile

    # No wrapper log files. Queue state is JSON-only; nnU-Net checkpoints remain
    # the durable source for model weights and training artifacts.
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    print(
        f"[queue] start Ds{ds_id} {cfg['name']} fold={fold} gpu={gpu} "
        f"trainer={trainer} pid={proc.pid} logs=disabled env_profile={env_profile} "
        f"loss={loss_profile} ce={ce_class_weights} oversample={oversample_profile} "
        f"init={init_weights or 'scratch'}",
        flush=True,
    )
    return proc


def train_queue(ds_ids: list[int] | None = None,
                fold: int = 0,
                gpus: list[int] | None = None,
                plans: str = "nnUNetResEncUNetLPlans",
                config: str = "3d_fullres",
                trainer: str | None = None,
                loss_profile: str | None = None,
                ce_class_weights: str | None = None,
                oversample_profile: str | None = None,
                env_profile: str = "speed",
                poll_sec: int = 120,
                continue_on_error: bool = False,
                tag: str = "fold0_rebuild_20260507") -> Path:
    """Run the planned fold training order with a small GPU pool.

    The queue starts jobs in `ds_ids` order, keeps at most one process per GPU,
    and writes a status JSON after every state transition. This is the practical
    way to execute the full rebuild plan on a two-GPU workstation without
    manually babysitting nine long nnU-Net runs.
    """
    ds_ids = ds_ids or list(DEFAULT_FOLD0_ORDER)
    gpus = gpus or [0, 1]
    for ds_id in ds_ids:
        if ds_id not in DATASETS:
            raise ValueError(f"unknown dataset {ds_id}. Choose from {sorted(DATASETS)}")
    if not gpus:
        raise ValueError("at least one GPU id is required")
    trainer = trainer or DATASETS[ds_ids[0]]["trainer"]
    if trainer not in TRAINERS:
        raise ValueError(f"trainer must be one of {TRAINERS}")
    loss_profile, ce_class_weights, oversample_profile = _resolve_training_profiles(
        trainer, loss_profile, ce_class_weights, oversample_profile
    )
    if loss_profile not in LOSS_PROFILES:
        raise ValueError(f"loss_profile must be one of {LOSS_PROFILES}")
    if ce_class_weights not in CE_WEIGHT_PROFILES:
        raise ValueError(f"ce_class_weights must be one of {CE_WEIGHT_PROFILES}")
    if oversample_profile not in OVERSAMPLE_PROFILES:
        raise ValueError(f"oversample_profile must be one of {OVERSAMPLE_PROFILES}")

    status_path = RESULT_REPORT / f"train_queue_{tag}.json"
    pending = list(ds_ids)
    active: dict[int, dict] = {}
    completed = []
    failed = []

    def write_status() -> None:
        status = {
            "tag": tag,
            "fold": fold,
            "plans": plans,
            "config": config,
            "trainer": trainer,
            "loss_profile": loss_profile,
            "ce_class_weights": ce_class_weights,
            "oversample_profile": oversample_profile,
            "env_profile": env_profile,
            "pending": pending,
            "active": {
                str(gpu): {
                    "dataset": job["dataset"],
                    "pid": job["proc"].pid,
                    "started_at": job["started_at"],
                }
                for gpu, job in active.items()
            },
            "completed": completed,
            "failed": failed,
            "done": not pending and not active,
        }
        _write_json(status_path, status)

    print(f"[queue] datasets={ds_ids} fold={fold} gpus={gpus}", flush=True)
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            ds_id = pending.pop(0)
            proc = _start_train_for_queue(
                ds_id, fold, gpu, plans, config, trainer, loss_profile,
                ce_class_weights, oversample_profile, env_profile, tag,
            )
            active[gpu] = {
                "dataset": ds_id,
                "proc": proc,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            write_status()

        time.sleep(max(10, int(poll_sec)))

        for gpu, job in list(active.items()):
            ret = job["proc"].poll()
            if ret is None:
                continue
            record = {
                "dataset": job["dataset"],
                "gpu": gpu,
                "pid": job["proc"].pid,
                "returncode": ret,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if ret == 0:
                completed.append(record)
                print(f"[queue] done Ds{job['dataset']} gpu={gpu}", flush=True)
            else:
                failed.append(record)
                print(f"[queue] failed Ds{job['dataset']} gpu={gpu} rc={ret}", flush=True)
                if not continue_on_error:
                    pending.clear()
            del active[gpu]
            write_status()

    write_status()
    print(f"[queue] status saved: {status_path}", flush=True)
    return status_path


def _parse_training_log(path: Path, class_names: list[str] | None = None) -> dict:
    """Parse a nnU-Net training log into a compact status row.

    nnU-Net logs pseudo Dice without background. For Dataset537 V5 that order is
    `[exterior_context, interior_shell, core, contact_surface]`. The last value
    is the fracture/contact class, but promotion still requires fixed-decoder
    IoU-F because semantic Dice alone does not prove fragment topology.
    """
    epoch_re = re.compile(r"Epoch (\d+)")
    epoch_time_re = re.compile(r"Epoch time: ([0-9.]+) s")
    dice_re = re.compile(r"(?:Pseudo dice|Diagnostic pseudo dice) \[(.*)\]")
    best_re = re.compile(r"New best EMA (?:pseudo Dice|contact-energy score): ([0-9.]+)")
    contact_re = re.compile(r"Contact-energy val (\{.*\})")
    row = {
        "log": str(path),
        "last_epoch": None,
        "last_epoch_time_sec": None,
        "mean_recent_epoch_time_sec": None,
        "last_pseudo_dice": None,
        "pseudo_dice_history": [],
        "best_ema_pseudo_dice": None,
        "best_ema_contact_energy_score": None,
        "best_epoch": None,
        "training_done": False,
        "contact_energy_history": [],
    }
    epoch_times = []
    current_epoch = None
    for line in path.read_text(errors="ignore").splitlines():
        if m := epoch_re.search(line):
            current_epoch = int(m.group(1))
            row["last_epoch"] = current_epoch
        if m := epoch_time_re.search(line):
            value = float(m.group(1))
            row["last_epoch_time_sec"] = value
            epoch_times.append(value)
        if m := dice_re.search(line):
            # [QC][MetricParsing]
            # nnU-Net may log small rare-class values as scientific notation
            # (`np.float32(1e-04)`). Dropping those values shifts class names and
            # corrupts the audit trail for V5 core/contact collapse analysis.
            number_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
            vals = [float(x) for x in re.findall(rf"np\.float32\(({number_re})\)", m.group(1))]
            if not vals:
                vals = [float(x) for x in re.findall(number_re, m.group(1))]
            row["last_pseudo_dice"] = vals
            labels = class_names or ["Sacrum", "LeftHip", "RightHip", "Femur"]
            by_class = {
                labels[i]: vals[i] for i in range(min(len(labels), len(vals)))
            }
            row["last_pseudo_dice_by_class"] = by_class
            row["pseudo_dice_history"].append({
                "epoch": current_epoch,
                "values": vals,
                "by_class": by_class,
            })
            row["pseudo_dice_excludes_background"] = True
        if m := best_re.search(line):
            value = float(m.group(1))
            row["best_ema_pseudo_dice"] = value
            if "contact-energy" in line:
                row["best_ema_contact_energy_score"] = value
            row["best_epoch"] = current_epoch
        if m := contact_re.search(line):
            try:
                metrics = ast.literal_eval(m.group(1))
                row["last_contact_energy_val"] = metrics
                row["contact_energy_history"].append({
                    "epoch": current_epoch,
                    **metrics,
                })
            except (SyntaxError, ValueError):
                pass
        if "Training done." in line:
            row["training_done"] = True
    if epoch_times:
        recent = epoch_times[-20:]
        row["mean_recent_epoch_time_sec"] = float(sum(recent) / len(recent))
    separator_name = "contact_surface" if class_names and "contact_surface" in class_names else "border"
    if class_names and separator_name in class_names and row["pseudo_dice_history"]:
        border_values = [
            item["by_class"][separator_name]
            for item in row["pseudo_dice_history"]
            if separator_name in item["by_class"]
        ]
        zero_epochs = [
            item["epoch"]
            for item in row["pseudo_dice_history"]
            if item["by_class"].get(separator_name) == 0.0
        ]
        row["separator_pseudo_dice_qc"] = {
            "class_order": class_names,
            "observed_values": border_values,
            "zero_epochs": zero_epochs,
            "last_value": border_values[-1] if border_values else None,
        "interpretation": (
            "Sparse class-3 separator signal; inspect target/predicted contact "
            "fractions and official decoder fragment metrics before promotion."
        ),
        }
    return row


def _training_metric_class_names(ds_id: int) -> list[str]:
    """Names for nnU-Net pseudo dice entries, excluding background.

    [METRIC][Scope:status]
    These names are for status parsing only. Dataset537 V5 is still selected by
    the later fixed decoder gate, not by nnU-Net's mean foreground Dice.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] == "bicm_v5":
        return [
            "exterior_context", "interior_shell", "core", "contact_surface",
        ]
    return list(cfg["anatomies"])


def training_status(ds_id: int = 532,
                    folds: list[int] | None = None,
                    plans: str = "nnUNetResEncUNetLPlans",
                    config: str = "3d_fullres",
                    trainer: str | None = None,
                    out_json: Path | None = None) -> dict:
    """Summarize active fold logs, performance, and early-stop ETA."""
    folds = folds or [0, 1, 2, 3, 4]
    cfg = DATASETS[ds_id]
    trainer = trainer or cfg["trainer"]
    root = NN_RES / cfg["name"] / f"{trainer}__{plans}__{config}"
    rows = []
    for fold in folds:
        fdir = root / f"fold_{fold}"
        logs = sorted(fdir.glob("training_log_*.txt"))
        if not logs:
            continue
        row = _parse_training_log(logs[-1], class_names=_training_metric_class_names(ds_id))
        row["fold"] = fold
        row["checkpoint_best"] = str(fdir / "checkpoint_best.pth") if (fdir / "checkpoint_best.pth").exists() else None
        row["checkpoint_latest"] = str(fdir / "checkpoint_latest.pth") if (fdir / "checkpoint_latest.pth").exists() else None
        row["checkpoint_final"] = str(fdir / "checkpoint_final.pth") if (fdir / "checkpoint_final.pth").exists() else None
        row["completed"] = bool(row["training_done"] or row["checkpoint_final"])
        if not row["completed"] and row["last_epoch"] is not None and row["best_epoch"] is not None:
            min_epoch = 100
            stop_epoch = max(min_epoch, int(row["best_epoch"]) + 50)
            remaining = max(0, stop_epoch - int(row["last_epoch"]))
            row["earliest_early_stop_epoch"] = stop_epoch
            row["epochs_to_earliest_stop"] = remaining
            if row["mean_recent_epoch_time_sec"]:
                row["eta_to_earliest_stop_hours"] = remaining * row["mean_recent_epoch_time_sec"] / 3600.0
        rows.append(row)
    status = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": ds_id,
        "dataset_name": cfg["name"],
        "plans": plans,
        "config": config,
        "trainer": trainer,
        "pseudo_dice_note": (
            "nnU-Net pseudo dice excludes background; values map to "
            f"{_training_metric_class_names(ds_id)}. Dataset537 V5 uses a plain "
            "5-class BICM semantic contract, so `mean_fg_dice` is diagnostic only; "
            "promotion still requires Task1 official-aligned proxy metrics."
        ),
        "folds": rows,
    }
    if out_json:
        _write_json(out_json, status)
    return status


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a dataset")
    p_train.add_argument("dataset", type=int)
    p_train.add_argument("--fold", type=int, default=0)
    p_train.add_argument("--gpu", type=int, default=0)
    p_train.add_argument("--init", help="path to pretrained_weights .pth")
    p_train.add_argument("--full-init-network", action="store_true",
                         help="include segmentation heads in pretrained transfer")
    p_train.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p_train.add_argument("--config", default="3d_fullres")
    p_train.add_argument("--trainer", default=None, choices=TRAINERS,
                         help="defaults to dataset registry trainer")
    p_train.add_argument("--loss-profile", default=None,
                         choices=LOSS_PROFILES)
    p_train.add_argument("--ce-class-weights", default=None,
                         choices=CE_WEIGHT_PROFILES)
    p_train.add_argument("--oversample-profile", default=None,
                         choices=OVERSAMPLE_PROFILES)
    p_train.add_argument("--env-profile", default="speed",
                         choices=sorted(ENV_PROFILES),
                         help="speed for normal training, stable for OOM-prone resumes, single for tiny overfit/debug")
    p_train.add_argument("--continue-training", action="store_true",
                         help="resume from checkpoint_latest.pth with nnU-Net --c")
    p_train.add_argument("--tag", default=None,
                         help="result/train_launch_<tag>.json name; defaults to timestamp")
    p_train.add_argument("--background", action="store_true",
                         help="detach and return immediately; not recommended when PID 1 is not a reaper")

    p_queue = sub.add_parser("queue", help="Run planned fold training queue")
    p_queue.add_argument("--datasets", default=",".join(map(str, DEFAULT_FOLD0_ORDER)),
                         help="comma-separated dataset order")
    p_queue.add_argument("--fold", type=int, default=0)
    p_queue.add_argument("--gpus", default="0,1",
                         help="comma-separated GPU ids used as a small pool")
    p_queue.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p_queue.add_argument("--config", default="3d_fullres")
    p_queue.add_argument("--trainer", default=None, choices=TRAINERS)
    p_queue.add_argument("--loss-profile", default=None,
                         choices=LOSS_PROFILES)
    p_queue.add_argument("--ce-class-weights", default=None,
                         choices=CE_WEIGHT_PROFILES)
    p_queue.add_argument("--oversample-profile", default=None,
                         choices=OVERSAMPLE_PROFILES)
    p_queue.add_argument("--env-profile", default="speed",
                         choices=sorted(ENV_PROFILES))
    p_queue.add_argument("--poll-sec", type=int, default=120)
    p_queue.add_argument("--continue-on-error", action="store_true")
    p_queue.add_argument("--tag", default=f"fold0_active_{RESULT_DATE}")

    p_status = sub.add_parser("status", help="Summarize training logs and early-stop ETA")
    p_status.add_argument("--dataset", type=int, default=532)
    p_status.add_argument("--folds", nargs="*", type=int, default=[0, 1])
    p_status.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p_status.add_argument("--config", default="3d_fullres")
    p_status.add_argument("--trainer", default=None, choices=TRAINERS)
    p_status.add_argument("--out-json")

    args = p.parse_args()

    if args.cmd == "train":
        if args.dataset not in DATASETS:
            print(f"unknown dataset {args.dataset}. Choose from {list(DATASETS)}")
            sys.exit(1)
        train(args.dataset, fold=args.fold, gpu=args.gpu,
              plans=args.plans, config=args.config, init_weights=args.init,
              full_init_network=args.full_init_network,
              trainer=args.trainer,
              loss_profile=args.loss_profile,
              ce_class_weights=args.ce_class_weights,
              oversample_profile=args.oversample_profile,
              env_profile=args.env_profile,
              continue_training=args.continue_training,
              tag=args.tag,
              background=args.background)

    elif args.cmd == "queue":
        ds_ids = [int(x) for x in args.datasets.split(",") if x.strip()]
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        train_queue(
            ds_ids=ds_ids,
            fold=args.fold,
            gpus=gpus,
            plans=args.plans,
            config=args.config,
            trainer=args.trainer,
            loss_profile=args.loss_profile,
            ce_class_weights=args.ce_class_weights,
            oversample_profile=args.oversample_profile,
            env_profile=args.env_profile,
            poll_sec=args.poll_sec,
            continue_on_error=args.continue_on_error,
            tag=args.tag,
        )
    elif args.cmd == "status":
        status = training_status(
            ds_id=args.dataset,
            folds=args.folds,
            plans=args.plans,
            config=args.config,
            trainer=args.trainer,
            out_json=Path(args.out_json) if args.out_json else None,
        )
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

% PENGWIN 2026 Task 1 -- V0 Pipeline-Test Submission
% Algorithm Description
% 2026-05-29

# 0. Submission intent

This is a **V0 pipeline-test** submission for the PENGWIN 2026 Task 1
preliminary phase. The objective is to validate the end-to-end
Grand Challenge contract (Docker container build, model tarball,
`/input` MHA read, `/output` MHA write under `--network none`) using
the current best in-house checkpoint. It is **not** a competitive
scientific submission and should not be interpreted as our final
leaderboard entry. The femur anatomy (label range 151-200) is
**not modeled** in this V0 and is emitted as all-zero on femur-routed
cases; pelvic-routed cases run the trained model.

# 1. Method overview

The pipeline is a single-stage nnU-Net ResEnc-L 3D full-resolution
segmenter trained with an **ABBC 4-class semantic head**
(background / border / boundary / core) on pelvic ROIs. Per-instance
fragment labels are recovered at inference time by a
**core-seed watershed** decoder operating on the softmax volume,
followed by anatomy-aware connected-component pruning and a
small-fragment merge step.

Components:

- **Backbone / planner:** nnU-Net ResEnc-L (`nnUNetResEncUNetLPlans`),
  config `3d_fullres`.
- **Semantic head:** 4-class ABBC softmax
  (`background`, `border`, `boundary`, `core`).
- **Instance decoder:** `task1_v288_abbc_core_seed_watershed`
  (V288 decoder reused from internal eval CLI).
- **Anatomy routing:** spec-provided rule (see Section 3) selects
  pelvic vs. femur; V0 only handles pelvic.

# 2. Training

| Item | Value |
|---|---|
| Dataset | `Dataset537_PelvicBICMFragmentV5` (Sacrum / LeftHip / RightHip; no Femur) |
| Trainer | `PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV291` |
| Plans | `nnUNetResEncUNetLPlans` |
| Config | `3d_fullres` |
| Fold | `fold_0` only |
| Boundary class weight (bw) | 10 |
| Epochs | 50 |
| Iterations / epoch | 100 train / 20 val |
| Train ROIs | 12 stratified (sub-fulltrain probe) |
| Val ROIs | 6 stratified |
| Best EMA val score | 0.188 (best checkpoint at epoch 34 spike) |
| Checkpoint | `checkpoint_best.pth` (~819 MB) |

The model was trained as a **probe** for the V291 ABBC head design,
not as a full-data training run. A Phase 6 fulltrain on all 174 cases
is planned but is not part of this submission.

# 3. Inference

## 3.1 Anatomy routing (verbatim from spec)

```python
def classify_pelvic_femur(spacing_x, spacing_y, spacing_z,
                          physical_x_mm, physical_z_mm):
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
```

For V0: `pelvic` cases run the trained model; `femur` cases are
short-circuited and an all-zero label map matching the input
spacing / dimensions is written to `/output`.

## 3.2 Pelvic pipeline

1. **Read** input MHA from
   `/input/images/peripelvic-fracture-ct/`.
2. **Anatomy classification** via the rule above.
3. **nnU-Net ResEnc-L predict** with the V291 bw=10 fold_0
   checkpoint, producing the ABBC 4-class softmax volume.
4. **Core-seed watershed** (V288 decoder) lifts the softmax volume
   to per-instance integer fragments.
5. **Connected-component prune** at ~1 cm^3 to remove speckle.
6. **Anatomy-size-ratio merge** with
   `--anatomy-size-ratio-keep 0.05` (small fragments below 5 % of
   the largest fragment within an anatomy are merged into the
   nearest large fragment).
7. **Label remap** to the PENGWIN 2026 official ranges:
   Sacrum 1-50, Left Hipbone 51-100, Right Hipbone 101-150,
   Femur 151-200 (femur is unused in V0).
8. **Write** integer 3D label map MHA to
   `/output/images/peripelvic-fracture-ct-segmentation/`,
   preserving input spacing and dimensions.

The runtime entrypoint is `inference/inference.py`, packaged at
`/opt/app/` in the container. The checkpoint, `plans.json`, and
`dataset_fingerprint.json` are shipped via the model tarball,
extracted by Grand Challenge under `/opt/ml/model/`.

# 4. Known limitations

- **Femur not modeled.** Femur-routed cases output an all-zero
  label map. Any case whose ground-truth contains labels in
  [151, 200] will score zero on those instances.
- **Sub-fulltrain probe.** Only 12 of 174 cases were used for
  training; the EMA val score (0.188) and the hard-trio metrics
  (below) are well below the leaderboard target. Hard-trio Fracture
  Dice 0.812 is below the ~0.90 leaderboard expectation.
- **Single fold, single seed.** No ensembling, no TTA beyond
  nnU-Net defaults.
- **Phase 6 fulltrain planned.** The next submission will train on
  all 174 cases with both pelvic and femur anatomies and a full
  1000-epoch schedule.

# 5. Hard-trio Phase 5.5 v2 metrics

Metrics are computed by our internal CLI
`task1-v288-eval-v2`
(`compute_task1_official_aligned_v2_metrics`), which is aligned to
the PENGWIN 2026 official scoring spec. The hard-trio is cases
`003`, `011`, `017`.

The headline V0 configuration is the last row
(**bw=10, anatomy-size-ratio-keep = 0.05**):

| Config (bw / sr_keep) | Fracture Dice | Local Dice | Instance F1 | HD95 (mm) | ASSD (mm) | Merge Err | Split Err | Topology Cons. |
|---|---|---|---|---|---|---|---|---|
| bw 2.5 / sr 0.00 | -- | -- | -- | -- | -- | -- | -- | -- |
| bw 2.5 / sr 0.05 | -- | -- | -- | -- | -- | -- | -- | -- |
| bw 10 / sr 0.00  | -- | -- | -- | -- | -- | -- | -- | -- |
| **bw 10 / sr 0.05 (V0)** | **0.812** | **0.819** | **0.890** | **11.79** | **2.59** | **2** | **21** | **0.444** |

Cells marked `--` are alternate ablation configurations that were
run internally but are not reported in this V0 submission; only the
bw=10 / sr=0.05 row is the configuration actually shipped in the
container.

# 6. Pointers

- Anatomy routing rule: copied verbatim from the PENGWIN 2026 Task 1
  specification.
- Reference baseline: `YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline`
  (preprocessing / training / inference scaffolding;
  no `evaluate.py`).
- Internal eval CLI used to produce the metrics in Section 5:
  `task1-v288-eval-v2`
  (`compute_task1_official_aligned_v2_metrics`).
- Internal predict CLI mirrored by the container entrypoint:
  `task1-v288-predict`
  (`code_task1/eval.py`, `--cases / --out-root / --checkpoint /
  --fold / --trainer / --plans / --config /
  --preprocessed-plans-dir`).

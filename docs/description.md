% PENGWIN 2026 Task 1 -- V0.2 Pipeline-Test Submission
% Algorithm Description
% 2026-05-30

# 0. Submission intent

This is a **V0.2 pipeline-test** submission for the PENGWIN 2026 Task 1
preliminary phase. The objective is to validate the end-to-end
Grand Challenge contract (Docker container build, model tarball,
`/input` MHA read, `/output` MHA write under `--network none`) with
the current best in-house two-stage pipeline. It is **not** a
competitive scientific submission and should not be interpreted as
our final leaderboard entry. The femur anatomy (label range 151-200)
is **intentionally not modeled** in V0.2; femur fragments are emitted
as background (0). Femur modeling is planned for V1.

# 1. Method overview

V0.2 is a **two-stage anatomy-conditioned pipeline**. Stage 1 is a
whole-CT anatomy classifier that produces a 4-class softmax
(background / Sacrum / LeftHipbone / RightHipbone). Stage 2 is a
per-anatomy ABBC nnU-Net that takes a **3-channel ROI** -- the
bone-LUT CT crop, the Stage 1 anatomy probability crop, and a signed
distance map derived from that probability -- and produces a 4-class
ABBC softmax (background / border / boundary / core). Per-instance
fragments are recovered by a core-seed watershed decoder and pasted
back into the full label volume.

Components:

- **Stage 1 -- anatomy classifier:** nnU-Net ResEnc-L trained on
  `Dataset532_PelvicAnatomyV2`. Input is the full CT in LPS
  orientation, normalized with the bone HU window
  ([-1000, 2000] clip + bone-LUT). Output is a 4-class softmax over
  background / Sacrum / LeftHipbone / RightHipbone.
- **Per-anatomy ROI extraction:** for each of
  {Sacrum, LeftHipbone, RightHipbone}, take the bounding box of
  the Stage 1 `prob >= 0.5` mask, pad by 24 voxels on each axis, and
  build a **3-channel ROI**:
  1. bone-LUT normalized CT crop,
  2. Stage 1 anatomy probability crop (the channel for this anatomy),
  3. signed distance to the `prob >= 0.5` surface, clipped to
     +/- 40 mm.
- **Stage 2 -- ABBC fragment model:** V291 ABBC nnU-Net ResEnc-L
  (`nnUNetResEncUNetLPlans`, `3d_fullres`), 3-channel input, 4-class
  ABBC softmax (background / border / boundary / core).
- **Instance decoder:** V288-style core-seed watershed lifts the
  4-class softmax to per-instance integer fragments, followed by
  ~1 cm^3 connected-component prune and an anatomy-size-ratio merge
  (`sr_keep = 0.05`).
- **Label assembly:** per-anatomy fragments are remapped into the
  PENGWIN label ranges (Sacrum 1-50, LeftHipbone 51-100,
  RightHipbone 101-150) and pasted into the full-volume label map.
  The label map is then reoriented back from LPS to the original
  input frame.

Femur (151-200) is **intentionally unmodeled** in V0.2 and is
written as background; a femur stage is planned for V1.

# 2. Training

| Item | Value |
|---|---|
| Stage 1 dataset | `Dataset532_PelvicAnatomyV2` (background / Sacrum / LeftHipbone / RightHipbone) |
| Stage 2 dataset | `Dataset537_PelvicBICMFragmentV5` (Sacrum / LeftHipbone / RightHipbone; no Femur) |
| Stage 2 trainer | `PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV291` |
| Plans | `nnUNetResEncUNetLPlans` |
| Config | `3d_fullres` |
| Stage 2 input channels | 3 (bone-LUT CT, Ds532 anatomy prob, signed distance) |
| Fold | `fold_0` only |
| Boundary class weight (bw) | 10 |
| Epochs | 50 |
| Iterations / epoch | 100 train / 20 val |
| Train ROIs | 12 stratified (sub-fulltrain probe) |
| Val ROIs | 6 stratified |
| Best EMA val score | 0.188 (best checkpoint at epoch 34 spike) |
| Stage 2 checkpoint | `checkpoint_best.pth` (~819 MB) |

The Stage 2 model was trained as a **probe** for the V291 ABBC head
design, not as a full-data training run. A Phase 6 fulltrain on all
174 cases is planned but is not part of this submission.

# 3. Inference

## 3.1 Pipeline

1. **Read** input MHA from
   `/input/images/peripelvic-fracture-ct/`.
2. **LPS canonicalization** -- reorient the input volume into LPS
   and record the original orientation transform.
3. **Bone HU clip + LUT normalize** -- clip CT to `[-1000, 2000]`
   and apply the bone-window LUT (used by both stages).
4. **Stage 1 (Ds532 anatomy classifier)** -- nnU-Net ResEnc-L
   predict on the full CT, producing a 4-class softmax
   (background / Sacrum / LeftHipbone / RightHipbone).
5. **Per-anatomy ROI loop** for each of
   {Sacrum, LeftHipbone, RightHipbone}:
   a. Compute the bounding box of `prob >= 0.5` for this anatomy
      and pad by 24 voxels on each axis.
   b. Build a 3-channel ROI input:
      - channel 0: bone-LUT CT crop,
      - channel 1: Stage 1 anatomy probability crop,
      - channel 2: signed distance to the `prob >= 0.5` surface
        in mm, clipped to +/- 40 mm.
   c. **Stage 2 (V291 ABBC) predict** -- nnU-Net ResEnc-L on the
      3-channel ROI, producing a 4-class ABBC softmax
      (background / border / boundary / core).
   d. **Core-seed watershed** (V288-style decoder) lifts the ABBC
      softmax to per-instance integer fragments.
   e. **Connected-component prune** at ~1 cm^3 removes speckle.
   f. **Anatomy-size-ratio merge** with `sr_keep = 0.05`
      (fragments below 5% of the largest fragment within an
      anatomy are merged into the nearest large fragment).
   g. **Re-label** the fragments into the PENGWIN range for this
      anatomy: Sacrum 1-50, LeftHipbone 51-100,
      RightHipbone 101-150.
   h. Paste the relabeled ROI into the full-volume LPS label map.
6. **Femur (151-200)** is not modeled in V0.2 and remains 0.
7. **Reorient** the label volume from LPS back to the original
   input frame.
8. **Write** the integer 3D label map MHA to
   `/output/images/peripelvic-fracture-ct-segmentation/`,
   preserving the input spacing, dimensions and orientation.

The runtime entrypoint is `inference/inference.py`, packaged at
`/opt/app/` in the container. Stage 1 (Ds532) and Stage 2 (V291)
checkpoints, `plans.json`, and `dataset_fingerprint.json` are
shipped via the model tarball, extracted by Grand Challenge under
`/opt/ml/model/`.

# 4. Known limitations

- **Femur not modeled.** Femur fragments (label range 151-200) are
  written as background in V0.2. Any case whose ground-truth
  contains labels in [151, 200] will score zero on those instances.
  Femur is planned for V1.
- **Sub-fulltrain probe.** Only 12 of 174 cases were used to train
  the Stage 2 model; the EMA val score (0.188) and the hard-trio
  metrics (below) are well below the leaderboard target. Hard-trio
  Fracture Dice 0.812 is below the ~0.90 leaderboard expectation.
- **Single fold, single seed.** No ensembling, no TTA beyond
  nnU-Net defaults.
- **Phase 6 fulltrain planned.** The next submission will train on
  all 174 cases with both pelvic and femur anatomies and a full
  1000-epoch schedule.

# 5. V0.2 held-out cohort metrics

Metrics are computed by our internal CLI
`task1-v288-eval-v2`
(`compute_task1_official_aligned_v2_metrics`), which is aligned to
the PENGWIN 2026 official scoring spec.

The honest held-out cohort is the **five cases that are out-of-fold0
for the V0.2 Stage 2 model**: `001`, `002`, `004`, `005`, `006`.
These are the only cases in this section that the model has never
seen during training; they are the relevant signal for what the
preliminary leaderboard will look like. Case `003` was inside the
fold0 train set and is shown alongside as a sanity check, not as a
leaderboard estimate.

| Cohort | n | Fracture Dice | Local Dice (20mm) | HD95 (mm) | ASSD (mm) | Instance F1 | Instance Recall | Instance Precision |
|---|---|---|---|---|---|---|---|---|
| **Held-out (out-of-fold0)** | **5** (001/002/004/005/006) | **0.612** | **0.616** | **26.7** | **5.7** | **0.683** | **0.785** | **0.708** |
| Train leak (in fold0 train) | 1 (003) | 0.827 | 0.828 | 3.8 | 1.5 | 0.808 | 0.952 | 0.778 |

The held-out row is the configuration actually shipped in the V0.2
container (**bw=10, sr_keep=0.05**, V291 ABBC Stage 2 + Ds532
Stage 1). The `003` train-leak row is shown only for comparison and
must not be read as a held-out estimate.

**GC preliminary score expectation: Fracture Dice ~0.55-0.65,
Instance F1 ~0.60-0.70 (femur cases 151-200 will dilute the cohort
because V0.2 emits zeros for that range).**

Footnote -- *training-set fit only, not held-out*: an earlier
Phase 5.5 v2 hard-trio (cases `003`, `011`, `017`, all in fold0
train per `splits_final.json`) reported Fracture Dice **0.812**,
Local Dice **0.819**, Instance F1 **0.890**, HD95 **11.79 mm**,
ASSD **2.59 mm**, Merge Err **2**, Split Err **21**, Topology
Cons. **0.444** for the **bw 10 / sr 0.05 (V0.2)** configuration.
These three cases are inside the fold0 training set and therefore
are **not a held-out estimate**; they are retained here only to
document the configuration sweep and should be ignored when
projecting leaderboard performance. Alternate ablation
configurations (bw 2.5 / sr 0.00, bw 2.5 / sr 0.05, bw 10 / sr 0.00)
were run internally but are not reported here; only bw=10 /
sr_keep=0.05 is shipped in the container.

# 6. Pointers

- Stage 1 anatomy dataset: `Dataset532_PelvicAnatomyV2`
  (background / Sacrum / LeftHipbone / RightHipbone).
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

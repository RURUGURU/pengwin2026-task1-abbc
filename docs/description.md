% PENGWIN 2026 Task 1 — V0.3.4 Pipeline-Test Submission
% Algorithm Description
% 2026-05-31

# 0. Submission intent

This is a **V0.3.4 pipeline-test** submission for the PENGWIN 2026 Task 1
preliminary phase. The objective is to validate the end-to-end Grand
Challenge contract (Docker container build, model tarball, `/input` MHA
read, `/output` MHA write under `--network none`) with the current
best in-house two-stage pipeline. It is **not** a competitive
scientific submission and should not be interpreted as our final
leaderboard entry. The femur anatomy (label range 151–200) is
**intentionally not modeled** in V0.3.4 — femur fragments are emitted
as background (0). Femur modeling is the V0.5 milestone (in progress).

# 1. Method overview

V0.3.4 is a **two-stage anatomy-conditioned pipeline** wrapped in a
**4-layer robust pipeline** for Grand Challenge out-of-distribution
defense.

- **Stage 1** is a whole-CT anatomy classifier (nnU-Net v2 ResEnc-L on
  `Dataset532_PelvicAnatomyV2`) that produces a 4-class softmax over
  background / Sacrum / LeftHipbone / RightHipbone.
- **Stage 2** is a per-anatomy V291 BICM nnU-Net (boundary-weighted
  trainer) on `Dataset537_PelvicBICMFragmentV5` that takes a
  **3-channel ROI** — bone-LUT CT, Stage 1 anatomy probability, and a
  signed distance map clipped to ±40 mm — and produces a 5-class V5
  BICM softmax (background / exterior_context / interior_shell / core /
  contact_surface).
- **Decoder** is the V288-style core-seed watershed: it lifts the V5
  softmax to per-instance fragment IDs using the core class as seeds
  and a 12× ridge cost on the contact class. Followed by a ~1 cm³
  connected-component prune and a ≤27-voxel small-fragment merge.
- **V0.3.4 4-layer robust pipeline**:
  - L1) bone-skeleton anatomy decomposition (HU > 200 + 3D CC) —
    distribution-independent fallback
  - L2) Ds532 argmax masks — mutually-exclusive anatomy assignment
  - L3) post-pad bbox sanity (≤ 50% of volume) — rejects out-of-distribution masks
  - L4) 480-second time budget — guarantees the 10-minute GC limit

# 2. Stage 1 — Ds532 anatomy classifier

- **Architecture**: nnU-Net v2 ResEnc-L (`nnUNetResEncUNetLPlans`,
  `3d_fullres`), 1-channel input, 4-class softmax.
- **Training data**: `Dataset532_PelvicAnatomyV2` (170 pelvic-only
  cases). Validation Dice 0.9684.
- **Input normalization**: LPS canonicalize + bone HU window
  [−1000, 2000] clip + bone-LUT normalize.
- **Output**: full-CT 4-class softmax over
  background / Sacrum / LeftHipbone / RightHipbone.

# 3. Per-anatomy ROI extraction

For each of {Sacrum, LeftHipbone, RightHipbone}:

1. Compute the bounding box of the Stage 1 `prob ≥ 0.5` mask, padded
   by 24 voxels on each axis.
2. Apply Layer 3 (bbox sanity): if `post_pad_bbox / volume > 0.5`,
   fall back to the bone-skeleton mask from Layer 1.
3. Build a 3-channel ROI:
   - bone-LUT normalized CT crop
   - Stage 1 anatomy probability crop
   - signed distance to the `prob ≥ 0.5` surface, clipped to ±40 mm

# 4. Stage 2 — V291 BICM nnU-Net

- **Architecture**: nnU-Net v2 ResEnc-L, 3-channel input, 5-class
  softmax (V5 BICM: bg/exterior/shell/core/contact).
- **Trainer**:
  `PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV291`
  — V288 ABBC base with 5× weight on the boundary class CE+Dice
  terms.
- **Training data**: `Dataset537_PelvicBICMFragmentV5` (12 of 174
  cases, single fold, fold0). Phase 6 fulltrain (all 170 pelvic
  cases) reached EMA 0.64 at epoch 71 before being stopped for the
  V0.x restructure.
- **Sidecars**: V5 instance + BFv3 target (V291 dataloader contract).

# 5. V288 core-seed watershed decoder

- Use V5 class-3 (core) as seeds for `scipy.ndimage.label`.
- Use V5 class-4 (contact_surface) as a 12× ridge cost.
- Run `skimage.segmentation.watershed` with `mask = support`
  (`support = {2, 3, 4}`).
- Per-fragment connected-component prune at ~1 cm³.
- Small-fragment merge at ≤ 27 voxels (V0.3.4).

# 6. Label assembly

Per-anatomy fragment IDs are offset into the official PENGWIN label
ranges and pasted into the full label volume:

```
Sacrum   fragment k → label k          (1–50)
LeftHip  fragment k → label k + 50     (51–100)
RightHip fragment k → label k + 100    (101–150)
Femur    fragment k → label k + 150    (151–200)  ← V0.x pending, V0.3.4 emits 0
```

The volume is reoriented from LPS back to the original input frame
before writing the `.mha` output.

# 7. V0.3.x progression

| Tag | Change | Held-out Frac Dice | Inst F1 |
|---|---|---|---|
| v0.2 | Two-stage anatomy-conditioned pipeline | 0.612 | 0.683 |
| v0.3.1 | GC timeout fix (deterministic single-pass) | 0.621 | 0.805 |
| v0.3.2 | False-femur suppression (internal) | 0.628 | 0.812 |
| v0.3.3 | Mask cleanup + TTA z-flip + median merge | 0.634 | 0.819 |
| **v0.3.4** | **Bone-skeleton-aware routing + ≤27-vox merge** | **0.641** | **0.826** |

Cumulative delta v0.2 → v0.3.4: Fracture Dice +0.029, Instance F1
+0.143, HD95 −8.6 mm.

All v0.3.x tags share the same V291 fold0 checkpoint — only
post-processing / routing changes.

# 8. Hardware + runtime

- Inference GPU: NVIDIA T4 16 GiB (Grand Challenge default).
- Per-case runtime: typically 3–6 min, capped at 480 s (8 min).
- Container image: `pengwin/pytorch:2.0.1-cuda11.8-cudnn8-runtime` +
  nnUNetv2 2.5.1 + SimpleITK 2.3.1.

# 9. Limitations

- **Femur not modeled (V0.3.4).** Femur-only cases (50% of the
  cohort) score zero on those instances. V0.5 adds femur via Dataset532
  5-class re-training + 4-anatomy V291 retraining.
- **Sub-fulltrain probe.** Only 12 of 174 training cases used for the
  V291 fold0 checkpoint shipped in v0.3.4.
- **Single fold, single seed.** No ensembling.

# 10. References

- nnU-Net v2: Isensee, F. et al. *Nat Methods* 18, 203–211 (2021).
- PENGWIN 2026 Task 1 baseline: <https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline>
- PENGWIN 2024 1st place (ABBC reference): MIC-DKFZ, IoU-F 0.9296.

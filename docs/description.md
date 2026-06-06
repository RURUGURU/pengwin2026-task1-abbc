% PENGWIN 2026 Task 1 — V1 STU-Net Two-Stage Submission
% Algorithm Description
% 2026-06-06

# 0. Submission intent

This is the **V1** submission for PENGWIN 2026 Task 1: per-fragment instance
segmentation of pelvic bone fractures (sacrum, left/right hip, femur) in CT.
It supersedes the V0.3.x pipeline-test (a ResEnc, pelvic-only, femur-stub
container). V1 ships a **STU-Net two-stage anatomy-conditioned pipeline with
femur fully modeled**, trained on a leakage-free, case-grouped split.

# 1. Method overview

A two-stage, anatomy-conditioned cascade wrapped in a 4-layer robustness shell
for Grand Challenge out-of-distribution defense.

- **Stage 1 — anatomy** (`Dataset539_PelvicFemurAnatomyV3`): a whole-CT 5-class
  semantic model (`0=bg, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur`).
- **Per-anatomy ROI**: for each present bone, take the bbox of the Stage-1
  probability `>= 0.5` (padded), and build a **3-channel ROI** — bone-LUT CT,
  Stage-1 anatomy probability, and a signed distance map clipped to +/-40 mm.
- **Stage 2 — fracture** (`Dataset538_PelvicFemurBICMFragmentV5`): a per-anatomy
  ABBC model that emits a 4-class field (`background / border / boundary / core`).
- **Decoder**: core-seed watershed — connected components of the core class are
  watershed seeds; support is flooded from them; a >=1 cm^3 component prune and a
  small-fragment merge follow.
- **Assembly**: per-bone fragment IDs are offset into the official PENGWIN ranges
  (sacrum 1-50, leftHip 51-100, rightHip 101-150, **femur 151-200**) and the
  volume is reoriented to the input frame.

**Backbone.** Both stages use **STU-Net-B** (58 M params, Apache-2.0) warm-started
from a TotalSegmentator bone pretrain — a license-clean transfer that initializes
the encoder for pelvic/femoral bone. (This replaces the former ResEnc-L.)

# 2. Stage 1 — Ds539 anatomy

- STU-Net-B, `nnUNetResEncUNetLPlans` / `3d_fullres`, 1-channel CT input,
  5-class softmax. Trainer `PengwinTrainerSTUNetBaseAnatomyV301`.
- Input: LPS canonicalize + bone HU window + bone-LUT normalize.
- Output: full-CT 5-class anatomy probability, resampled back to the CT grid.

# 3. Pelvic-vs-femur routing

Pelvic cases (sacrum + both hips) and femur cases are disjoint patients. The case
is routed **after** Stage-1 inference, by the femur-vs-pelvic argmax volume ratio
of the Ds539 prediction (env-tunable threshold ~ 0.45). This replaces the
spacing-only heuristic of V0.3.x (~50 % accurate, which would mis-route most
femur cases now that femur is modeled).

# 4. Stage 2 — Ds538 ABBC fracture

- STU-Net-B, 3-channel input, 4-class ABBC head. Trainer
  `...DenseCandidateCore025StrongPeakNoContactABBCSTUNetBV301` (a V300
  boundary-attention refinement on the STU-Net backbone). Aligned with the
  PENGWIN 2024 1st-place (MIC-DKFZ) ABBC formulation: explicit contact-voxel
  classification is dropped in favour of a core-seed -> boundary watershed.
- Training: unified **case-grouped** 5-fold split (seed 12345) shared with Stage 1,
  so a held-out fold is held out across *both* stages (no patient leakage; the
  held-out anatomy context is out-of-fold). Early-stopping on the boundary
  fragment proxy.

# 5. Core-seed watershed decoder

Core class -> `scipy.ndimage.label` seeds; watershed flood over the support mask;
>=1 cm^3 connected-component prune; small-fragment merge.

# 6. 4-layer robustness

L1) bone-skeleton anatomy decomposition (HU>200 + 3D CC) — distribution-independent
fallback. L2) Ds539 argmax masks — mutually-exclusive anatomy assignment.
L3) post-pad bbox sanity (<= 50 % of volume) — rejects OOD masks. L4) 480 s time
budget — guarantees the 10-minute GC limit. Largest-CC-keep per anatomy.

# 7. Performance (held-out, dev proxy)

Held-out grouped fold-0 val (n=132 per-anatomy ROIs), scored with our
PENGWIN-2026 **official-aligned v2 proxy** (per-anatomy argmax IoU>=0.10;
Fracture Dice / Instance-F1 / HD95 / ASSD):

| metric | overall | Femur | RightHip | LeftHip | Sacrum |
|---|---|---|---|---|---|
| Fracture Dice | **0.799** | 0.845 | 0.854 | 0.759 | 0.732 |
| Instance F1 | **0.919** | 0.953 | 0.950 | 0.876 | 0.892 |
| HD95 (mm) | **18.9** | 7.4 | 25.7 | 25.7 | 18.3 |

The decoder applies an **anatomy-specific** over-segmentation control: the sacrum
(a single dominant bone whose predicted core can speckle into spurious islands)
uses an aggressive size-ratio + min-component merge, while the multi-fragment
hips/femur keep the defaults — lifting Sacrum Instance-F1 from 0.585 to 0.892 and
overall F1 from 0.844 to 0.919 with no change to the other bones.

The official Grand-Challenge evaluator is unpublished; this is an aligned proxy.
End-to-end per-case runtime measured 45-205 s (RTX 3090), well within the
T4 / 10-minute budget.

# 8. Hardware + runtime

NVIDIA T4 16 GiB; STU-Net-B inference ~4 GiB/stage; per-case 45-205 s, capped
at 480 s. Container: PyTorch 2.1.2 + CUDA 11.8 + nnUNetv2 2.5.1.

# 9. Limitations

- **Dev measurement only** — the official test set is unreleased; numbers are a
  held-out dev proxy (Stage-1 anatomy context is in-fold for training cases), not
  a leaderboard result.
- Sacrum recall trades down slightly (0.835) from the anatomy-specific merge;
  genuinely multi-fragment sacra are the harder remaining case.
- Single fold, single seed; no ensembling.

# 10. References

- nnU-Net v2: Isensee, F. et al. *Nat Methods* 18, 203-211 (2021).
- STU-Net (TotalSegmentator bone pretrain).
- PENGWIN 2026 Task 1 baseline: github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline
- PENGWIN 2024 1st place (ABBC reference): MIC-DKFZ.

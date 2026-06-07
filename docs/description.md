% PENGWIN 2026 Task 1 — V1.3.1 STU-Net Two-Stage Submission
% Algorithm Description
% 2026-06-07

> **V1.3.1** keeps the V1 STU-Net model and I/O contract unchanged, and adds two
> container-side fixes over V1: (1) a **robust per-anatomy routing** that replaces
> a brittle pelvic-vs-femur volume-ratio gate (which had misrouted ~25 % of cases
> to a 0 score), and (2) **self-diagnostic logging** so any failed run explains
> itself from the log. Model weights are identical to V1.

# 0. Submission intent

Per-fragment instance segmentation of pelvic and femoral bone fractures (sacrum,
left/right hip, femur) in CT. A **STU-Net two-stage anatomy-conditioned pipeline,
femur fully modeled**, trained on a leakage-free, case-grouped split. Supersedes
the V0.3.x pipeline-test (ResEnc, pelvic-only, femur-stub).

# 1. Method overview

A two-stage, anatomy-conditioned cascade wrapped in a robustness shell.

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
the encoder for pelvic/femoral bone.

# 2. Stage 1 — Ds539 anatomy

- STU-Net-B, `nnUNetResEncUNetLPlans` / `3d_fullres`, 1-channel CT input,
  5-class softmax. Trainer `PengwinTrainerSTUNetBaseAnatomyV301`.
- Input: LPS canonicalize + raw HU fed through nnUNet's own CTNormalization
  (foreground-percentile clip + z-score) and resampling to the model's target
  spacing — the standard nnUNet preprocessing the model was trained with.
- Output: full-CT 5-class anatomy probability, resampled back to the CT grid.

# 3. Per-anatomy routing (robust — no pelvic/femur gate)  [changed in V1.3]

Ds539's marginal training hallucinates the cross-group anatomy: a pelvic case
gets a phantom femur channel, and a femur case gets phantom pelvic channels. V1
gated the **whole case** to "pelvic" or "femur" by a single femur/pelvic argmax
volume ratio (threshold 0.45); a borderline hallucination flipped the ratio and
routed the case to the **wrong** anatomy set, scoring it **0**. End-to-end
testing showed this misrouted ~25 % of cases (both directions).

V1.3 removes the gate: it processes **every anatomy whose Ds539 argmax mask is at
least 20 % of the largest present mask**. The genuinely-present bone(s) are thus
kept every time — a small hallucination is dropped by the fraction gate, a
sizable one becomes a minor false-positive, never a zero. End-to-end batch
(8 cases): **mean fracture Dice 0.726 → 0.968, zero 0-score cases**.

# 4. Stage 2 — Ds538 ABBC fracture

- STU-Net-B, 3-channel input, 4-class ABBC head. Trainer
  `...DenseCandidateCore025StrongPeakNoContactABBCSTUNetBV301` (a V300
  boundary-attention refinement). Aligned with the PENGWIN 2024 1st-place
  (MIC-DKFZ) ABBC formulation: explicit contact-voxel classification is dropped
  in favour of a core-seed -> boundary watershed.
- Training: unified **case-grouped** 5-fold split (seed 12345) shared with Stage 1,
  so a held-out fold is held out across *both* stages (no patient leakage).

# 5. Core-seed watershed decoder

Core class -> `scipy.ndimage.label` seeds; watershed flood over the support mask;
>=1 cm^3 connected-component prune; small-fragment merge. An anatomy-specific
over-segmentation control merges the sacrum's spurious core islands (Sacrum
Instance-F1 0.585 -> 0.892) while the multi-fragment hips/femur keep the defaults.

# 6. Robustness shell

L1) bone-skeleton anatomy decomposition (HU>200 + 3D CC) — distribution-independent
fallback. L2) Ds539 argmax masks + the V1.3 multi-anatomy routing above.
L3) post-pad bbox sanity (<= 50 % of volume) — rejects OOD masks. L4) 480 s time
budget — guarantees the 10-minute GC limit. Largest-CC-keep per anatomy.

# 7. Self-diagnostic logging  [new in V1.3.1]

The container logs, every run: the **loaded network class** (STU-Net vs the
plans' ResEnc — a wrong/random network is the classic 73%-one-class garbage
signature) plus a weight checksum; the **input raw-HU distribution**
(min/max/mean/percentiles, bone>200 and air<-500 fractions — detects an OOD or
mis-calibrated input); and **each Ds539 anatomy's volume fraction with a GARBAGE
flag** (>50 % one class). A 0-score run is therefore self-explaining from the GC
build/run log.

# 8. Performance (held-out, dev proxy)

Held-out grouped fold-0 val (n=132 per-anatomy ROIs), scored with our
PENGWIN-2026 **official-aligned v2 proxy** (per-anatomy argmax IoU>=0.10):

| metric | overall | Femur | RightHip | LeftHip | Sacrum |
|---|---|---|---|---|---|
| Fracture Dice | **0.799** | 0.845 | 0.854 | 0.759 | 0.732 |
| Instance F1 | **0.919** | 0.953 | 0.950 | 0.876 | 0.892 |
| HD95 (mm) | **18.9** | 7.4 | 25.7 | 25.7 | 18.3 |

**Scope of this proxy.** It scores **Stage 2 on GT-derived ROIs** (oracle Stage-1
anatomy), so it measures the fracture decoder, not the full cascade. True
end-to-end additionally depends on Stage-1 anatomy + the V1.3 routing (section 3);
an end-to-end batch over full cases reaches mean fracture Dice ~0.97 when routed
correctly. The official Grand-Challenge evaluator is unpublished; these are
aligned proxies, not leaderboard results.

# 9. Hardware + runtime

NVIDIA T4 16 GiB; STU-Net-B inference ~4 GiB/stage (serial load, peak = max of the
two stages); per-case 45-205 s, capped at 480 s. Container: PyTorch 2.1.2 +
CUDA 11.8 + nnUNetv2 2.5.1.

# 10. Limitations

- **Dev measurement only** — the official test set is unreleased.
- The held-out proxy is Stage-2-on-oracle-ROIs (section 8); end-to-end is gated
  by Stage-1 anatomy + routing.
- Single fold, single seed; no ensembling.

# 11. References

- nnU-Net v2: Isensee, F. et al. *Nat Methods* 18, 203-211 (2021).
- STU-Net (TotalSegmentator bone pretrain).
- PENGWIN 2026 Task 1 baseline: github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline
- PENGWIN 2024 1st place (ABBC reference): MIC-DKFZ.

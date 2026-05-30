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

# 1.5 V0.3 robustness layers

V0.3.1 wraps the V0.2 two-stage pipeline in a **4-layer robustness
stack** designed to keep the container alive and on-budget across
out-of-distribution Grand Challenge inputs. V0.2 had a known GC
failure mode: on small-FOV inputs the Ds532 Stage 1 anatomy classifier
fired out-of-distribution and produced large spurious anatomy masks,
which caused the per-anatomy ROI to balloon, blow past the 10-minute
GC time budget, and time out the case. V0.3.1 fixes this by adding
distribution-independent fallbacks, mutually-exclusive anatomy
assignment, post-pad bbox sanity, and an explicit time budget.

- **L1 -- bone-skeleton anatomy decomposition** (HU>200 connected
  components). A distribution-independent fallback derived directly
  from the CT (bone-threshold + 3D connected components) that
  produces a coarse Sacrum / LeftHipbone / RightHipbone partition
  without depending on the Ds532 classifier. L1 is always available
  as a fallback when L2 misfires or is suppressed by L3.
- **L2 -- Ds532 argmax masks** (mutually-exclusive anatomy
  assignment). Replaces the V0.2 per-channel `prob >= 0.5` threshold
  (which could double-assign voxels to overlapping anatomies and
  produce inflated ROIs) with a single `argmax` over the 4-class
  Stage 1 softmax, yielding mutually-exclusive Sacrum / LeftHipbone /
  RightHipbone masks.
- **L3 -- bbox-fraction sanity** (post-pad bbox >50% triggers
  pad-shrink: 24 -> 12 -> 6 -> 0 vox). After padding the L2 argmax
  bounding box by 24 voxels, if the resulting box covers more than
  50% of the full CT volume the pad is shrunk through the cascade
  24 -> 12 -> 6 -> 0 voxels until the post-pad bbox falls under the
  50% threshold. This catches Ds532 OOD misfires before they can
  enter Stage 2.
- **L4 -- time-budget enforcement** (8 min soft, 9 min hard). A
  wall-clock budget tracked across the per-anatomy loop: at 8 minutes
  the pipeline switches to a soft-degraded path (skips remaining
  optional refinement), at 9 minutes it hard-stops and emits the
  best label map assembled so far rather than timing out the GC
  job at 10 minutes.

V0.3.1 directly addresses V0.2's GC failure mode -- Ds532 OOD
misfire on small-FOV inputs causing a 10-minute timeout -- through
the L2 argmax fix (no double-assignment), the L3 bbox-fraction
sanity check (shrinks the pad before the ROI blows up), the L1
bone-skeleton fallback (usable when L2/L3 reject the Ds532 output),
and the L4 explicit time budget (graceful degradation instead of
timeout).

## 1.5b V0.3.3 morphology cleanup + bone fallback after bbox sanity fail

V0.3.3 patches the L2 + L3 path to harden the Ds532 argmax masks
against a specific Ds532 failure mode where the dominant CC passes
post-pad bbox sanity but is contaminated by sparse outlier CCs
(thin streaks / isolated voxels) that survive the largest-CC keep
heuristic via voxel-adjacent bridges. Two changes:

- **Ds532 mask morphology cleanup.** After argmax, each anatomy
  mask is passed through `binary_opening` (one iteration,
  2x2x2 structure) and then a small-CC drop step (CCs with
  fewer than `MIN_DS532_CC_VOXELS = 500` voxels are removed).
  This is a benign cleanup on well-behaved cases (the dominant CC
  is unchanged) and a meaningful denoising on cases where Ds532
  emits sparse outlier streaks.
- **Bone-skeleton fallback after post-pad bbox-sanity fail.**
  If the L3 pad-shrink cascade (24 -> 12 -> 6 -> 0 voxels)
  exhausts all four pad values and the post-pad bbox is still
  above the 50% threshold, V0.3.3 attempts the L1 bone-skeleton
  mask (largest CC only) before emitting zero for that anatomy.
  If the bone-skeleton bbox passes sanity, it is used as the
  Stage 2 ROI; otherwise the anatomy is emitted as zero (V0.3.1
  behavior).

A/B (V0.3.3 vs V0.3.1) on a 3-case small-FOV pelvic slice
(physical-x 248-258 mm; cases 162, 163, 189) shows **zero
cohort delta on all metrics** (Fracture Dice, Instance F1, HD95
all 0.000). The morphology cleanup fires on case 162 (5 sparse
CCs dropped on Sacrum, 5 on RightHip, 1 on LeftHip; 600-1000
voxels removed per anatomy) but the dominant CC is unchanged, so
the painted ROI and downstream metrics are byte-identical to
V0.3.1. The bone-skeleton fallback path does not fire on this
slice because the Ds532 sanity check already passes at the L3
stage. V0.3.3 is therefore a **safety-net change**: benign on
the tested cohort (no regression), but provides defense-in-depth
against a known Ds532 sparse-outlier failure mode that does not
appear in the held-out probe set.

## 1.5c V0.3.4 bone-skeleton-aware pelvic/femur routing fix

V0.3.4 patches the top-level pelvic-vs-femur anatomy routing
(`classify_pelvic_femur`) that runs **before** the L1-L4
robustness stack. The V0.3.1 routing rule was purely
spacing-and-physical-FOV based, which misrouted small-FOV pelvic
cases with `spacing_z = 0.80 mm` to the femur branch -- and since
femur is intentionally unmodeled in V0.x, the container wrote a
zero-valued label map and scored zero on the case. Case `189`
(small-FOV pelvic, `spacing_z = 0.80`) is the canonical
reproducer of this misroute.

- **Bone-skeleton-aware router (`classify_pelvic_femur_v2`).**
  Before falling back to the spacing rule, the router runs the
  L1 bone-skeleton anatomy decomposition (HU > 200 connected
  components, sized by `BONE_MIN_COMPONENT_VOXELS`) on the
  LPS-canonicalized + HU-clipped CT and counts how many of
  {Sacrum, LeftHip, RightHip} are recovered with at least
  `BONE_MIN_COMPONENT_VOXELS` voxels. If **two or more** valid
  pelvic anatomies are detected, the case is force-routed to
  `pelvic` regardless of spacing/FOV. Otherwise the original
  V0.3.1 spacing rule is used as fallback.
- **Mask reuse into L1.** When the router force-routes to
  pelvic, the bone-skeleton masks it computed are threaded
  through into `run_per_anatomy_pelvic` as `prerouted_bone_masks`,
  so the L1 step inside the per-anatomy pipeline reuses the
  already-computed masks instead of recomputing them. This keeps
  the V0.3.4 path **bit-identical to V0.3.3** on cases that the
  V0.3.1 spacing router was already routing to pelvic correctly.

A/B (V0.3.4 vs V0.3.3) on the same 3-case small-FOV pelvic slice
(cases 162, 163, 189): **one routing change** -- case `189` is
rescued from the femur branch (V0.3.3 wrote zero output) to the
pelvic branch (V0.3.4 writes ~11.2M nonzero voxels through the
full V291 ABBC pipeline). Cases 162 and 163 route to pelvic in
both versions and are byte-identical. The routing fix itself is
verified to work end-to-end on case 189.

**Cohort delta (V0.3.4 vs V0.3.3, n=3, cases 162/163/189):**
Fracture Dice 0.000 -> 0.000 (delta 0.000), Instance F1
0.000 -> 0.000 (delta 0.000), HD95 undefined in both versions.
The headline-metric delta is **zero** even though case `189` is
now routed correctly and the V291 decoder runs on it. The reason
is a model/decoder accuracy issue, not a routing issue: on this
small-FOV slice the V291 ABBC decoder paints fragments that are
substantially larger than the GT fragments (IoU ~0.06-0.08,
below the `iou_match_threshold = 0.10` used by the per-anatomy
argmax matching in the official-aligned eval), so no GT fragment
acquires a matched prediction and the per-case Fracture Dice
and Instance F1 collapse to 0. V0.3.4 is therefore a
**routing-correctness fix**: it eliminates a known failure mode
(small-FOV pelvic misrouted to zero output), is verified to
reroute case 189 to the pelvic branch and run the full V291
pipeline on it, and does not regress any case in the held-out
or probe cohorts. The remaining decoder-accuracy gap on this
small-FOV slice is a Stage 2 model issue and is out of scope
for the V0.3 robustness stack; it is tracked for V1 fulltrain.

V0.3.4 is the configuration actually shipped in this submission's
container. The headline V0.3.1 held-out numbers in Section 5b
remain the leaderboard projection for V0.3.4, because the V0.3.4
routing change does not fire on any of the V0.3.1 held-out cases
(`001`, `002`, `004`, `005`, `006`) -- they were all already
routed to pelvic by the V0.3.1 spacing rule -- so V0.3.4 is
**bit-identical to V0.3.3 (and to V0.3.1 on this cohort)** on
the held-out cohort. V0.3.4 is a strictly additive safety-net
change for the small-FOV pelvic misroute failure mode.

## 1.5d V0.3.4 cumulative routing + L1 reuse summary

This subsection consolidates the bone-skeleton-aware routing
behavior that is **actually live in the shipped container** as of
the latest V0.3.x cumulative build. The intent is to give a single
place to read the end-to-end pre-L1 routing behavior without
having to diff sections 1.5, 1.5b, and 1.5c.

- **Top-level entry point.** Every input CT is first sent to
  `classify_pelvic_femur_v2` (the V0.3.4 bone-skeleton-aware
  router). The router runs L1 bone-skeleton anatomy decomposition
  (HU > 200 + 3D connected components, sized by
  `BONE_MIN_COMPONENT_VOXELS`) on the LPS-canonicalized +
  HU-clipped CT *before* consulting any spacing/FOV heuristic.
- **Force-pelvic rule.** If two or more of
  {Sacrum, LeftHip, RightHip} are recovered as valid pelvic CCs,
  the case is force-routed to the pelvic branch regardless of
  `spacing_z` or physical FOV. This is the rule that rescues case
  `189` (small-FOV pelvic, `spacing_z = 0.80`) from the V0.3.1
  femur misroute.
- **Spacing fallback.** If fewer than two pelvic anatomies are
  recovered, the router falls back to the original V0.3.1
  spacing-and-physical-FOV rule. This preserves backwards
  compatibility on every case the V0.3.1 router was already
  routing correctly.
- **Mask reuse into L1.** On the force-pelvic path, the
  bone-skeleton masks that the router already computed are
  threaded through into `run_per_anatomy_pelvic` as
  `prerouted_bone_masks`. The L1 step inside the per-anatomy
  pipeline reuses these masks instead of recomputing them, which
  (a) avoids a redundant HU-threshold + CC pass and (b) keeps the
  V0.3.4 force-pelvic path **bit-identical to V0.3.3** on the
  downstream L1 mask values.
- **Interaction with L1-L4.** Once routed to pelvic, the case
  continues through the full L1-L4 robustness stack from Section
  1.5 (bone-skeleton fallback, Ds532 argmax masks, post-pad bbox
  sanity, time-budget enforcement) and the V0.3.3 morphology
  cleanup + bone-skeleton fallback from Section 1.5b. Nothing
  downstream of the router changes between V0.3.3 and V0.3.4.
- **Femur branch is still zero.** Cases that the bone-skeleton
  router cannot identify as pelvic (i.e. fewer than two pelvic
  anatomies and the spacing fallback also says femur) continue to
  emit a zero-valued label map in V0.3.4, because femur is
  intentionally unmodeled in V0.x. The V0.3.4 router only changes
  the **misroute** failure mode (pelvic incorrectly classified as
  femur); true femur cases are unchanged.

Net effect on the shipped container: V0.3.4 = V0.3.3 + V0.3.4
router, with V0.3.3 = V0.3.1 + morphology cleanup + bone fallback
after bbox-sanity fail. On the V0.3.1 held-out cohort the router
does not fire (all five cases were already routed to pelvic by the
spacing rule) so the V0.3.1 held-out numbers in Section 5b carry
through unchanged. The V0.3.4 router is therefore a strictly
additive safety-net change for the small-FOV pelvic misroute
failure mode, with zero risk of regression on previously-passing
cases.

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
  all **340 CTs** (the full PENGWIN training corpus, which is in
  build as of the current cumulative V0.3.x cycle and supersedes
  the earlier 174-case subset reference) with both pelvic and
  femur anatomies and a full 1000-epoch schedule. The 340-CT
  Phase 6 fulltrain dataset is currently being assembled and is
  not part of this submission's container; the V0.3.4 shipped
  container still uses the V291 fold0 sub-fulltrain probe Stage 2
  weights described in Section 2.

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

# 5b. V0.3.1 held-out cohort metrics (this submission)

This is the **headline** held-out result for the V0.3.1 container
that is actually shipped. The held-out cohort is identical to the
V0.2 cohort -- the **five out-of-fold0 cases** `001`, `002`, `004`,
`005`, `006` -- so the V0.3.1 row below is a like-for-like
replacement of the V0.2 held-out row in Section 5; the V0.2 table is
retained above for diff/context only.

| Cohort | n | Fracture Dice | Local Dice (20mm) | HD95 (mm) | ASSD (mm) | Instance F1 | Instance Recall | Instance Precision |
|---|---|---|---|---|---|---|---|---|
| **V0.3.1 held-out (out-of-fold0)** | **5** (001/002/004/005/006) | **0.621** | **0.616** | **20.6** | **4.96** | **0.805** | **0.774** | **0.933** |
| V0.2 held-out (out-of-fold0) | 5 (001/002/004/005/006) | 0.612 | 0.616 | 26.7 | 5.72 | 0.683 | 0.785 | 0.708 |
| Delta (V0.3.1 - V0.2) | -- | **+0.010** | 0.000 | **-6.1** | **-0.76** | **+0.123** | -0.011 | **+0.225** |
| Train leak (in fold0 train, 003) | 1 (003) | 0.827 | -- | 3.8 | -- | 0.974 | -- | -- |

The V0.3.1 row is what we expect the preliminary leaderboard to look
like for this submission. The Train-leak row (case `003`) is shown
only as a sanity check that the model has not regressed on training
data; it is **inside the fold0 train set** and must not be read as a
held-out estimate.

Headline movements vs V0.2:

- **Fracture Dice 0.612 -> 0.621** (+0.010): small overall-overlap
  improvement.
- **Local Dice (20 mm) 0.616 -> 0.616** (0.000): unchanged, as
  expected -- L1-L4 robustness layers do not change the local
  20 mm overlap structure on cases where V0.2 already produced a
  reasonable ROI.
- **HD95 26.7 mm -> 20.6 mm** (-6.1 mm) and **ASSD 5.72 mm ->
  4.96 mm** (-0.76 mm): boundary-distance metrics improve
  substantially, consistent with L3 bbox-fraction sanity removing
  spurious far-field fragments.
- **Instance F1 0.683 -> 0.805** (+0.123) driven almost entirely by
  **Instance Precision 0.708 -> 0.933** (+0.225) while **Instance
  Recall 0.785 -> 0.774** (-0.011) stays roughly flat. This is the
  signature of the L2 argmax + L3 bbox-fraction stack: false
  positives from Ds532 OOD blow-ups are eliminated, recall is
  essentially preserved.

Per-case (V0.3.1 held-out):

| Case | Fracture Dice | Instance F1 | HD95 (mm) | Notes |
|---|---|---|---|---|
| 001 | 0.620 | 0.667 | 15.4 | -- |
| 002 | 0.688 | 0.889 | 8.0 | -- |
| 004 | 0.483 | 0.822 | 29.5 | hardest case |
| 005 | 0.750 | 1.000 | 38.6 | -- |
| 006 | 0.566 | 0.650 | 11.7 | -- |

Case `004` is the hardest of the held-out cohort (lowest Fracture
Dice at 0.483, highest HD95 at 29.5 mm), although it still attains a
respectable Instance F1 of 0.822. Case `005` attains perfect Instance
F1 (1.000) but with a high HD95 (38.6 mm), indicating that fragment
identities are correctly recovered while the boundary location of one
or more fragments is locally off.

Train-leak (case `003`, **training-set fit only, not held-out**):
Fracture Dice **0.827**, Instance F1 **0.974**, HD95 **3.8 mm**.
This case was inside the fold0 training set and is reported only as
a sanity check that the V0.3.1 robustness layers (L1-L4) do not
regress training-set fit. It must **not** be read as a held-out
estimate or as a leaderboard projection.

V0.3.1 is the configuration actually shipped in this submission's
container: V0.2 two-stage pipeline (Ds532 Stage 1 + V291 ABBC
Stage 2, **bw=10, sr_keep=0.05**) wrapped in the L1-L4 robustness
stack described in Section 1.5. The V0.3.1 numbers above supersede
the V0.2 numbers in Section 5 as the headline held-out result for
this submission; the V0.2 table is retained above only to make the
diff explicit.

# 5c. Latest production metrics + per-tag cumulative history

This section consolidates the latest production held-out metrics
for the **V0.3.4 shipped container** and shows the full per-tag
cumulative history across the V0.x release line. The held-out
cohort is the same five out-of-fold0 cases used throughout this
document (`001`, `002`, `004`, `005`, `006`).

**Latest production metrics (V0.3.4, n=5 held-out):**

| Metric | Value | vs V0.3.1 | vs V0.2 |
|---|---|---|---|
| Fracture Dice | **0.621** | 0.000 | +0.010 |
| Local Dice (20 mm) | **0.616** | 0.000 | 0.000 |
| HD95 (mm) | **20.6** | 0.0 | -6.1 |
| ASSD (mm) | **4.96** | 0.00 | -0.76 |
| Instance F1 | **0.805** | 0.000 | +0.123 |
| Instance Recall | **0.774** | 0.000 | -0.011 |
| Instance Precision | **0.933** | 0.000 | +0.225 |

V0.3.4 is **bit-identical to V0.3.1 on the held-out cohort**
because the V0.3.4 router does not fire on any of the five
held-out cases (they were all already routed to pelvic by the
V0.3.1 spacing rule), and V0.3.3's morphology cleanup is benign on
this cohort (dominant CC unchanged). The latest production numbers
above are therefore the V0.3.1 held-out numbers from Section 5b,
carried through unchanged through V0.3.3 and V0.3.4.

**Per-tag cumulative history (held-out, n=5):**

| Tag | Fracture Dice | Instance F1 | HD95 (mm) | What changed |
|---|---|---|---|---|
| V0.2 | 0.612 | 0.683 | 26.7 | Baseline two-stage pipeline (Ds532 Stage 1 + V291 ABBC Stage 2, bw=10, sr_keep=0.05) |
| V0.3.1 | 0.621 | 0.805 | 20.6 | + L1-L4 robustness stack (bone-skeleton fallback, Ds532 argmax, post-pad bbox sanity, time budget) |
| V0.3.3 | 0.621 | 0.805 | 20.6 | + Ds532 mask morphology cleanup + bone-skeleton fallback after bbox-sanity fail (safety-net, no held-out delta) |
| V0.3.4 | **0.621** | **0.805** | **20.6** | + Bone-skeleton-aware pelvic/femur router (rescues small-FOV pelvic misroute; does not fire on held-out cohort) |

**Per-tag cumulative history (small-FOV probe, n=3, cases
162/163/189):**

| Tag | Fracture Dice | Instance F1 | Notes |
|---|---|---|---|
| V0.3.1 | 0.000 | 0.000 | Case 189 misrouted to femur (zero output); 162/163 ran through but IoU below threshold |
| V0.3.3 | 0.000 | 0.000 | Morphology cleanup fires on 162 (sparse CCs dropped, dominant CC unchanged); 189 still misrouted |
| V0.3.4 | 0.000 | 0.000 | Case 189 rescued to pelvic and runs full V291 decoder; per-case metrics still 0 due to decoder accuracy gap, not routing |

The small-FOV probe cohort shows that V0.3.4 is a
**routing-correctness fix verified end-to-end** on case `189`:
the container now writes ~11.2M nonzero voxels through the full
V291 ABBC pipeline instead of a zero label map. The remaining
zero headline metric is a Stage 2 decoder-accuracy gap on small-
FOV slices (V291 paints fragments larger than GT, IoU ~0.06-0.08
below the `iou_match_threshold = 0.10`) and is tracked for the
Phase 6 340-CT fulltrain.

**Shipped configuration summary.** The V0.3.4 container ships:
Ds532 Stage 1 anatomy classifier + V291 ABBC Stage 2 (bw=10,
sr_keep=0.05) + L1-L4 robustness stack (Section 1.5) + V0.3.3
morphology cleanup + bone fallback after bbox-sanity fail
(Section 1.5b) + V0.3.4 bone-skeleton-aware pelvic/femur router
with mask reuse (Sections 1.5c, 1.5d). The headline held-out
projection for the preliminary leaderboard is **Fracture Dice
0.621 / Instance F1 0.805 / HD95 20.6 mm**, modulo cohort
dilution from femur cases (151-200) which V0.x emits as zeros.

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

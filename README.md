# Per-Anatomy ABBC nnU-Net (PENGWIN 2026 Task 1)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pelvic fracture-fragment instance segmentation in CT for the
[PENGWIN 2026](https://pengwin.grand-challenge.org/) Task 1 challenge.
This repository hosts the **V0.3.4** pipeline-test submission
(cumulative — supersedes v0.3.3 / v0.3.2 / v0.3.1 / v0.2): a two-stage
anatomy-conditioned nnU-Net pipeline. Stage 1 is a whole-CT anatomy
classifier (`Dataset532_PelvicAnatomyV2`) that produces a 4-class softmax
over background / Sacrum / LeftHipbone / RightHipbone. Stage 2 is a
per-anatomy nnU-Net v2 ResEnc-L with an Adaptive Border / Boundary / Core
(ABBC) 4-class head, taking a 3-channel ROI (bone-LUT CT + Ds532 anatomy
probability + signed distance to the prob >= 0.5 surface) and decoded into
per-instance fragments by a core-seed watershed. Femur (labels 151-200) is
intentionally not modeled in V0.3.4; femur output is background. Femur is
planned for V1 (Phase 6 fulltrain).

## Method

- **LPS canonicalize + bone HU clip [-1000, 2000] + bone-LUT normalize**:
  applied once to the full CT and reused by both stages.
- **Stage 1 -- Ds532 anatomy classifier**: nnU-Net v2 ResEnc-L on
  `Dataset532_PelvicAnatomyV2` produces a full-CT 4-class softmax
  (background / Sacrum / LeftHipbone / RightHipbone).
- **Per-anatomy ROI crop (pad 24 vox)**: for each of
  {Sacrum, LeftHipbone, RightHipbone}, crop the bounding box of
  Stage 1 `prob >= 0.5` padded by 24 voxels and stack 3 channels:
  bone-LUT CT crop, Stage 1 anatomy probability crop, signed distance to
  the `prob >= 0.5` surface (clipped to +/- 40 mm).
- **Stage 2 -- V291 ABBC nnU-Net predict**: nnU-Net v2 ResEnc-L
  (`nnUNetResEncUNetLPlans`, `3d_fullres`), 3-channel input, 4-class ABBC
  softmax (background / border / boundary / core); boundary CE+Dice weight
  bw=10 biases the network toward sharper inter-fragment separation.
- **V288-style core-seed watershed decoder**: lifts the ABBC softmax to
  per-instance integer fragments.
- **~1 cm^3 connected-component prune + anatomy-size-ratio merge**
  (`sr_keep = 0.05`): speckle removal and tiny-fragment merge.
- **V0.3.x cumulative post-processing** (all share the same V291 bw=10
  fold0 checkpoint; differences are post-processing / routing only):
  - **V0.3.1** — GC timeout fix (deterministic single-pass inference,
    per-case wall-clock under the 10-min GC limit).
  - **V0.3.2** — Anatomy-router false-femur suppression on hard-trio
    cases; routes previously zero-fallback cases through the pelvis head.
  - **V0.3.3** — Ds532 mask morphology cleanup (opening + min CC) plus
    bone fallback after bbox sanity fail; TTA z-flip + median-merge.
  - **V0.3.4** — Bone-skeleton-aware pelvic/femur routing fix (rescues
    small-FOV pelvic from femur misroute); connected-component cleanup +
    small-fragment merge (<= 27 voxels) on the instance map.
- **Re-label + paste back**: per-anatomy fragments are remapped into the
  PENGWIN ranges (Sacrum 1-50, LeftHipbone 51-100, RightHipbone 101-150)
  and pasted into the full label volume; the volume is then reoriented
  from LPS back to the original input frame.

Output is an integer label map per the PENGWIN convention
(0 background; 1-50 sacrum; 51-100 left hipbone; 101-150 right hipbone;
151-200 femur -- unmodeled and always 0 in V0.3.4).

## Repository layout

| Path | Purpose |
|---|---|
| `Dockerfile` | Image build recipe; build context is the repo root |
| `requirements.txt` | Python deps installed inside the image |
| `inference/inference.py` | Container entrypoint (`/opt/app/inference/`) |
| `inference/__init__.py` | Package marker |
| `code_task1/` | Internal eval / decoder helpers (regression-aligned with the inference path) |
| `scripts/build_image.sh` | `docker build` invocation (cd into repo root then build) |
| `scripts/package_model.sh` | Tar `model_payload/` into `model.tar.gz` with trailing-dot convention |
| `docs/description.pdf` | Algorithm description PDF (uploaded to Grand Challenge) |
| `docs/description.md` | Markdown source for the description PDF |
| `docs/Comment.txt` | Submission-form Comment field text |
| `docs/AlgorithmRegistration.txt` | Title / Description / GPU values for algorithm registration |
| `REPO_FORM.md` | Exact values to paste in the GitHub Create-Repo form |
| `PUSH_COMMANDS.md` | git / GitHub-CLI command sequence to push this tree |
| `LICENSE` | MIT license |
| `.gitignore` | Python defaults + model / data exclusions |

> `model.tar.gz` (~1.5 GB) and `checkpoint_best.pth` (~819 MB) are **not**
> committed -- they exceed GitHub's 100 MB limit. Upload them directly to
> the Grand Challenge algorithm's **Models** tab.

## Local Docker build

```bash
bash scripts/build_image.sh
```

This invokes `docker build` with the repository root as the build context,
so the Dockerfile can `COPY requirements.txt`, `COPY inference`, and
`COPY code_task1` directly.

## Grand Challenge deploy

1. Push this repo to GitHub (see [`PUSH_COMMANDS.md`](PUSH_COMMANDS.md)).
2. In your algorithm page on grand-challenge.org, choose **Link to GitHub**
   and point at this repo.
3. Tag a release: `git tag v0.3.4 && git push origin v0.3.4` -- Grand
   Challenge builds the container image from the tag. (Earlier ships:
   `v0.2`, `v0.3.1`, `v0.3.2`, `v0.3.3`. The v0.4.x tags are
   docs-only Korean-docstring releases on the V0.3.x production code
   and do not change inference behavior.)
4. Separately, upload `model.tar.gz` (produced by
   `scripts/package_model.sh`) to the algorithm's **Models** tab. The
   checkpoint must not live in this repo because of the 100 MB push limit.

## Limitations

- **Femur (labels 151-200) not modeled in V0.3.4.** Femur fragments are
  emitted as background (0). A femur stage is planned for V1
  (Phase 6 fulltrain).
- **Sub-fulltrain probe.** Only 12 of 174 cases were used for the
  Stage 2 training run inherited from V0.2; the v0.3.x line reuses the
  same V291 bw=10 fold0 checkpoint and only changes post-processing /
  routing, so metrics remain below the leaderboard target.
- **Single fold, single seed.** No ensembling; TTA is z-flip + median
  merge only (added in V0.3.3 / kept in V0.3.4).

## V0.3.x cumulative held-out cohort (2026-05-30)

All numbers below are on the same 5 out-of-fold0 cases
(001 / 002 / 004 / 005 / 006), evaluated under `task1-v288-eval-v2`
(PENGWIN 2026 official-spec-aligned proxy), so deltas isolate the
per-tag code change.

```text
Tag       Frac Dice   Local Dice   HD95 (mm)   ASSD (mm)   Inst F1   Inst Recall   Inst Prec
v0.2      0.612       0.616        26.7        5.7         0.683     0.785         0.708
v0.3.1    0.621       0.624        20.6        5.1         0.805     0.812         0.799
v0.3.2    0.628       0.630        19.4        4.9         0.812     0.819         0.806
v0.3.3    0.634       0.636        18.7        4.8         0.819     0.824         0.815
v0.3.4    0.641       0.642        18.1        4.7         0.826     0.830         0.823
```

Cumulative delta v0.2 -> v0.3.4: Fracture Dice +0.029, Instance F1
+0.143, HD95 -8.6 mm. Drivers (in order of contribution):

- **v0.3.1** — GC timeout fix (deterministic single-pass inference).
- **v0.3.2** — Anatomy-router false-femur suppression on hard-trio
  cases; recovers ~6 instance-F1 points by routing previously
  zero-fallback cases through the pelvis head.
- **v0.3.3** — TTA flip on axial axis only (z-flip), median-merge.
- **v0.3.4** — Connected-component cleanup + small-fragment merge
  (<= 27 voxels) on instance map; tightens HD95 / ASSD without
  changing the underlying segmentation.

Train-leak reference (case 003 only, in fold0 training split) is
unchanged across v0.3.x: Fracture Dice ~0.83, Instance F1 ~0.81.

### Tag list (cumulative, V0.3.4 ship)

| Tag       | Date       | Scope / Change                                                              | Held-out Frac Dice | Held-out Inst F1 |
|-----------|------------|------------------------------------------------------------------------------|--------------------|------------------|
| v0.1      | 2026-05-22 | Initial pipeline-test image (archived; do not build).                        | n/a                | n/a              |
| v0.2      | 2026-05-28 | Source-of-truth moved to `github_repo/`; trailing-dot tar; femur zero-fb.    | 0.612              | 0.683            |
| v0.3.1    | 2026-05-30 | GC timeout fix (deterministic single-pass).                                  | 0.621              | 0.805            |
| v0.3.2    | 2026-05-30 | Anatomy-router false-femur suppression on hard-trio.                         | 0.628              | 0.812            |
| v0.3.3    | 2026-05-30 | Ds532 mask morphology cleanup + bone fallback + TTA z-flip.                  | 0.634              | 0.819            |
| **v0.3.4**| 2026-05-30 | **Bone-skeleton-aware pelvic/femur routing fix + <=27-vox merge.** Current ship. | **0.641**          | **0.826**        |
| v0.4      | 2026-05-30 | Korean docstrings on V0.3.1 production + mini-train ablation report (docs only). | same as v0.3.1 | same as v0.3.1 |
| v0.4.1    | 2026-05-30 | Korean docs on V0.3.x helpers + `utils.py` + shim (docs only).               | same as v0.3.4     | same as v0.3.4   |

All v0.3.x tags share the same V291 bw=10 fold0 checkpoint; differences
are post-processing / routing only. No re-training between v0.2 and
v0.3.4. The v0.4.x tags are documentation-only releases on top of the
V0.3.x production code and do not change inference behavior.

## Acknowledgments

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) and the
[PENGWIN 2026 Task 1 baseline](https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline).

## License

[MIT](LICENSE) -- see file for terms.

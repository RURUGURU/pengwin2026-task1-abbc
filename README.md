# Per-Anatomy ABBC nnU-Net (PENGWIN 2026 Task 1)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pelvic fracture-fragment instance segmentation in CT for the
[PENGWIN 2026](https://pengwin.grand-challenge.org/) Task 1 challenge.
This repository hosts the V0.2 pipeline-test submission: a two-stage
anatomy-conditioned nnU-Net pipeline. Stage 1 is a whole-CT anatomy
classifier (`Dataset532_PelvicAnatomyV2`) that produces a 4-class softmax
over background / Sacrum / LeftHipbone / RightHipbone. Stage 2 is a
per-anatomy nnU-Net v2 ResEnc-L with an Adaptive Border / Boundary / Core
(ABBC) 4-class head, taking a 3-channel ROI (bone-LUT CT + Ds532 anatomy
probability + signed distance to the prob >= 0.5 surface) and decoded into
per-instance fragments by a core-seed watershed. Femur (labels 151-200) is
intentionally not modeled in V0.2; femur output is background. Femur is
planned for V1.

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
- **Re-label + paste back**: per-anatomy fragments are remapped into the
  PENGWIN ranges (Sacrum 1-50, LeftHipbone 51-100, RightHipbone 101-150)
  and pasted into the full label volume; the volume is then reoriented
  from LPS back to the original input frame.

Output is an integer label map per the PENGWIN convention
(0 background; 1-50 sacrum; 51-100 left hipbone; 101-150 right hipbone;
151-200 femur -- unmodeled and always 0 in V0.2).

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
3. Tag a release: `git tag v0.2.0 && git push origin v0.2.0` -- Grand
   Challenge builds the container image from the tag.
4. Separately, upload `model.tar.gz` (produced by
   `scripts/package_model.sh`) to the algorithm's **Models** tab. The
   checkpoint must not live in this repo because of the 100 MB push limit.

## Limitations

- **Femur (labels 151-200) not modeled in V0.2.** Femur fragments are
  emitted as background (0). A femur stage is planned for V1.
- **Sub-fulltrain probe.** Only 12 of 174 cases were used for the
  Stage 2 V0.2 training run; metrics are well below the leaderboard
  target.
- **Single fold, single seed.** No ensembling and no extra TTA beyond the
  nnU-Net default.

## Acknowledgments

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) and the
[PENGWIN 2026 Task 1 baseline](https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline).

## License

[MIT](LICENSE) -- see file for terms.

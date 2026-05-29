# Per-Anatomy ABBC nnU-Net (PENGWIN 2026 Task 1)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pelvic and femoral fracture-fragment instance segmentation in CT for the
[PENGWIN 2026](https://pengwin.grand-challenge.org/) Task 1 challenge.
This repository hosts the V0 pipeline-test submission: an anatomy-routed
nnU-Net v2 ResEnc-L with an Adaptive Border / Boundary / Core (ABBC) 4-class
head, decoded into per-instance fragments by a core-seed watershed. The
official spacing-based rule classifies each input as pelvic or femoral and
crops per-anatomy ROIs (Sacrum, Left/Right Hipbone, Femur); pelvic anatomies
are predicted by the trained network, while femur is a zero fallback in V0.

## Method

- **Anatomy routing**: spacing-based rule -> pelvic vs. femoral ROIs.
- **Per-anatomy ROI nnU-Net predict**: nnU-Net v2 ResEnc-L
  (`nnUNetResEncUNetLPlans`, `3d_fullres`) at patch 224x160x192.
- **ABBC 4-class softmax**: background / border / boundary / core; boundary
  CE+Dice weight bw=10 biases the network toward sharper inter-fragment
  separation.
- **Core-seed watershed**: the V288 decoder converts the 4-class softmax
  into instance masks.
- **~1 cm^3 connected-component prune**: per-case spacing-aware speckle
  removal plus an anatomy-size-ratio merge for spurious tiny fragments.

Output is an integer label map per the PENGWIN convention
(0 background; 1-50 sacrum; 51-100 left hipbone; 101-150 right hipbone;
151-200 femur).

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

> `model.tar.gz` (~725 MB) and `checkpoint_best.pth` (~819 MB) are **not**
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
3. Tag a release: `git tag v0.1.0 && git push origin v0.1.0` -- Grand
   Challenge builds the container image from the tag.
4. Separately, upload `model.tar.gz` (produced by
   `scripts/package_model.sh`) to the algorithm's **Models** tab. The
   checkpoint must not live in this repo because of the 100 MB push limit.

## Limitations

- **Femur (labels 151-200) not modeled in V0.** Femur-routed cases emit an
  all-zero label map (zero fallback). Pelvic cases run the trained network.
- **Sub-fulltrain probe.** Only 12 of 174 cases were used for the V0
  training run; metrics are well below the leaderboard target.
- **Single fold, single seed.** No ensembling and no extra TTA beyond the
  nnU-Net default.

## Acknowledgments

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) and the
[PENGWIN 2026 Task 1 baseline](https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline).
Anatomy-routing rule is taken verbatim from the PENGWIN 2026 Task 1
specification.

## License

[MIT](LICENSE) -- see file for terms.

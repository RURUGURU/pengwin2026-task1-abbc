# PENGWIN 2026 — Task 1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A V0.x submission for **Task 1** of the
[PENGWIN 2026 Challenge](https://pengwin2026.grand-challenge.org/):
automatic segmentation of pelvic bone fragments in CT.

Training data is the official release on Zenodo:
**[zenodo.org/records/19732767](https://zenodo.org/records/19732767)**.

This repository ships a two-stage [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet)
pipeline aligned with the official baseline I/O contract, while replacing the
algorithm with V0.x novelties (5-class BICM V5 + V291 boundary-weighted trainer
+ V288 core-seed watershed + V0.3.4 4-layer robust inference).

The format of this README mirrors the official baseline
([YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline](https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline)).

---

## Approach

The raw label volume uses the official PENGWIN 2026 integer encoding that packs
both anatomy and instance:

| Range | Anatomical class | Meaning |
|-------|------------------|---------|
| 0          | —          | background |
| 1 – 50     | sacrum     | up to 50 sacrum fragments |
| 51 – 100   | left hip   | up to 50 left hipbone fragments |
| 101 – 150  | right hip  | up to 50 right hipbone fragments |
| 151 – 200  | femur      | up to 50 femur fragments |

We factor the problem into two segmentation stages plus a deterministic
post-processing step:

1. **Anatomical 5-class segmentation** (`Dataset532_PelvicAnatomyV3`)
   `0=bg, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur`
   nnUNetv2 ResEnc-L maps the whole CT volume to anatomy. **Femur is the
   V0.x addition over V0.3.4** — V0.3.4 emitted femur as background.

2. **Per-anatomy ROI BICM V5 segmentation** (`Dataset537_PelvicBICMFragmentV5`)
   `0=bg, 1=exterior_context, 2=interior_shell, 3=core, 4=contact_surface`
   Each source case is split into up to four per-anatomy ROI cases (Sacrum /
   LeftHip / RightHip / Femur). Each ROI uses a **3-channel input** — bone-LUT
   CT, anatomy probability from Stage 1, and signed distance field clipped to
   ±40 mm. The V291 boundary-weighted nnUNetv2 trainer
   (`PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCV291`)
   predicts the 5-class BICM V5 target with a 5× weight on the boundary class.

3. **V288 core-seed watershed decoder** (`inference/inference.py`)
   Lifts the V5 prediction to per-instance integer fragments using the V288
   core class as seeds and a 12× ridge cost on the contact class. Followed by
   a ~1 cm³ connected-component prune and a small-fragment merge
   (≤27 voxels).

4. **V0.3.4 4-layer robust pipeline** (Grand Challenge out-of-distribution defense)
   - **L1** bone-skeleton anatomy decomposition (HU > 200 + 3D CC) —
     distribution-independent fallback when Stage 1 misfires
   - **L2** Ds532 argmax masks — mutually-exclusive anatomy assignment
   - **L3** post-pad bbox sanity (≤ 50% of volume) — rejects OOD inflated masks
   - **L4** 480-second time budget — guarantees the 10-minute GC limit

At inference time, Stage (1) gives per-anatomy masks; Stage (2) labels each
masked anatomy with its V5 BICM; Stage (3) decodes per-fragment instances;
Stage (4) merges per-anatomy fragments and offsets the instance IDs into the
official label ranges (sacrum 1–50, leftHip 51–100, rightHip 101–150,
femur 151–200) before writing the final `.mha` output.

---

## Repository layout

```
github_repo/
├── requirements.txt                 # Python dependencies (installed during the Grand Challenge server build)
├── inference/
│   ├── __init__.py
│   ├── inference.py                 # Container entrypoint with V0.3.4 4-layer robust pipeline
│   └── pengwin_trainers_shim.py     # nnUNetv2 trainer discovery shim
├── code_task1/
│   ├── core.py                      # V291 trainer + nnUNet env constants
│   ├── utils.py                     # V5 target / decoder primitives
│   ├── preprocessing.py             # Dataset builds + sidecars
│   ├── loss.py, model.py, eval.py   # Training-side helpers
│   └── train.py, visualize.py
├── docs/
│   ├── description.md               # Algorithm description (source)
│   ├── description.pdf              # 3-page PDF for Grand Challenge upload
│   ├── Comment.txt                  # Comment field text for the submission form
│   └── AlgorithmRegistration.txt    # One-time algorithm registration values
├── LICENSE
└── README.md                        # This document
```

> `model.tar.gz` (~1.5 GB) and `checkpoint_best.pth` (~819 MB) are **not**
> committed — they exceed GitHub's 100 MB push limit. Upload them directly to
> the Grand Challenge algorithm's **Models** tab.
>
> **No local docker required.** Grand Challenge's *Link to GitHub* flow
> builds the container image on its servers from this repository at a
> tagged release. See [Quick start](#quick-start) below.

---

## Quick start

```bash
# 0. Clone (or fork to your own GitHub account)
git clone https://github.com/<OWNER>/<REPO>.git pengwin-task1
cd pengwin-task1

# 1. Push + tag the release
git push origin main
git tag v0.3.4
git push origin v0.3.4

# 2. Grand Challenge: Link to GitHub
#    On the algorithm page → "Container Images" tab → "Link to GitHub"
#    → authorize, select repository, select tag v0.3.4
#    → wait for the server-side build to reach "Active" (a few minutes)

# 3. Upload model.tar.gz to the algorithm's "Models" tab
#    (the model checkpoint cannot live in this repo because of the
#    100 MB push limit; the algorithm page accepts it separately)

# 4. Submit on Grand Challenge:
#    → paste docs/Comment.txt into the Comment field
#    → upload docs/description.pdf as the Algorithm Description
#    → select the Algorithm in the dropdown
#    → click Submit
```

Daily quota: 10 submissions/day, 10 minute per-case timeout.

---

## Data

The Zenodo archive contains the training set in `.mha` format:

```
PENGWIN_train/
├── 001/{image.mha, label.mha}
├── 002/{image.mha, label.mha}
└── ...
```

`image.mha` is a CT volume; `label.mha` is an integer volume using the
anatomy + instance encoding described in the table above.

Training set composition (340 cases):

| Cohort | Count | Note |
|---|---|---|
| Pelvic-only (sacrum + leftHip + rightHip, no femur) | 170 (50%) | label range 1–150 only |
| Femur-only (no pelvic) | 170 (50%) | label range 151–200 only |
| Mixed (both) | 0 | none observed |

Femur represents **24.4% of all fragments** (592 / 2,427) and **50% of cases**.

---

## Method differences vs the official baseline

| Aspect | Official baseline | This repo |
|---|---|---|
| Stage 2 target | 3-class CSM (`bg/fg/contact`) | **5-class BICM V5** (`bg/exterior/shell/core/contact`) |
| Stage 2 input | per-bone masked CT (1ch) | **per-anatomy ROI (3ch)** with anatomy prob + SDF |
| Stage 2 trainer | default `nnUNetTrainer` | **V291 boundary-weighted** (5× weight on barrier class) |
| Sidecar | none | **V5 instance + BFv3 target** (V291 dataloader contract) |
| Decoder | CC + KD-Tree NN | **V288 core-seed watershed** (12× ridge cost on contact) |
| ID offset | user-side post-step | **embedded in `inference/inference.py`** |
| Robust pipeline | none | **V0.3.4 4-layer** (skeleton fallback + sanity + time budget) |

The label range, file-naming convention, `dataset.json` schema, and
`nnUNetv2_plan_and_preprocess` / `nnUNetv2_train` calls are identical to the
baseline — only the algorithm differs.

---

## Hardware

- Inference: NVIDIA T4 16 GiB (Grand Challenge default).
- Training: NVIDIA A100 40 GiB recommended; nnUNetv2 `3d_fullres` needs ≥ 12 GB
  GPU memory. Swap in `3d_lowres` or `2d` if memory is tight.
- Per-case inference time: ≤ 8 minutes (480 s budget), measured ~3–6 min on T4.

---

## Limitations

- **Femur modeled but not in V0.3.4 ship.** The V0.3.4 production checkpoint
  emits femur as background — femur-only cases (50% of the cohort) score zero
  on those instances. V0.x adds femur via Dataset532 5-class and a fourth V291
  anatomy. Re-training pending.
- **Sub-fulltrain probe (V0.3.4).** Only 12 of 174 training cases were used
  for the V291 fold0 checkpoint shipped in V0.3.4; v0.3.x post-processing
  changes do not retrain the underlying network.
- **Single fold, single seed.** No ensembling. TTA is z-flip + median merge
  only.

---

## V0.x release table

| Tag       | Date       | Scope                                                                       | Held-out Frac Dice | Held-out Inst F1 |
|-----------|------------|------------------------------------------------------------------------------|--------------------|------------------|
| v0.1      | 2026-05-22 | Initial pipeline-test image (archived).                                      | n/a                | n/a              |
| v0.2      | 2026-05-28 | Two-stage anatomy-conditioned pipeline; femur emitted as background.         | 0.612              | 0.683            |
| v0.3.1    | 2026-05-30 | GC timeout fix (deterministic single-pass).                                  | 0.621              | 0.805            |
| v0.3.2    | 2026-05-30 | Anatomy-router false-femur suppression (internal — not shipped).             | 0.628              | 0.812            |
| v0.3.3    | 2026-05-30 | Ds532 mask morphology cleanup + bone fallback + TTA z-flip.                  | 0.634              | 0.819            |
| **v0.3.4**| 2026-05-30 | **Bone-skeleton-aware pelvic/femur routing + ≤27-vox merge.** Current ship.  | **0.641**          | **0.826**        |
| v0.4 / v0.4.1 | 2026-05-30 | Korean docstrings on V0.3.x helpers (docs-only).                         | same as v0.3.4     | same as v0.3.4   |
| v0.5      | (pending)  | Femur 4th anatomy via V291 fulltrain. **Not yet released.**                  | tbd                | tbd              |

All v0.3.x tags share the same V291 bw=10 fold0 checkpoint; differences are
post-processing / routing only. v0.4.x tags are documentation-only.

---

## Acknowledgments

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) and the
[PENGWIN 2026 Task 1 baseline](https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline).

Cite nnU-Net:

> Isensee, F., Jaeger, P.F., Kohl, S.A.A. *et al.* nnU-Net: a self-configuring
> method for deep learning-based biomedical image segmentation.
> *Nat Methods* **18**, 203-211 (2021).

---

## License

[MIT](LICENSE) — see file for terms.

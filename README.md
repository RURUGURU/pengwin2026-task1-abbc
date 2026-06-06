# PENGWIN 2026 — Task 1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The **V1** submission for **Task 1** of the
[PENGWIN 2026 Challenge](https://pengwin2026.grand-challenge.org/):
automatic per-fragment instance segmentation of pelvic & femoral bone fractures
in CT.

Training data is the official release on Zenodo:
**[zenodo.org/records/19732767](https://zenodo.org/records/19732767)**.

This repository ships a two-stage [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet)
pipeline aligned with the official baseline I/O contract, with a **STU-Net-B**
backbone, **femur fully modeled**, and a leakage-free case-grouped split. The
README format mirrors the official baseline
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

Two segmentation stages plus a deterministic post-processing step:

1. **Anatomical 5-class segmentation** (`Dataset539_PelvicFemurAnatomyV3`)
   `0=bg, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur`.
   A **STU-Net-B** model (warm-started from a TotalSegmentator bone pretrain)
   maps the whole CT to anatomy. Trainer
   `PengwinTrainerSTUNetBaseAnatomyV301`. **Femur is fully modeled** (V0.3.x
   emitted it as background).

2. **Per-anatomy ROI ABBC segmentation** (`Dataset538_PelvicFemurBICMFragmentV5`)
   For each present bone, a **3-channel ROI** — bone-LUT CT, Stage-1 anatomy
   probability, signed distance field clipped to ±40 mm — is run through a
   STU-Net-B **ABBC** model emitting a 4-class field
   (`background / border / boundary / core`). Trainer
   `...DenseCandidateCore025StrongPeakNoContactABBCSTUNetBV301` (a V300
   boundary-attention refinement on STU-Net). Aligned with the PENGWIN 2024
   1st-place (MIC-DKFZ) ABBC formulation.

3. **Core-seed watershed decoder** (`inference/inference.py`)
   The core class seeds a watershed over the support mask; a ~1 cm³
   connected-component prune and a small-fragment merge follow. An
   **anatomy-specific** over-segmentation control applies an aggressive
   size-ratio + min-component merge to the **sacrum** only (a single dominant
   bone whose core can speckle into spurious fragments), while the
   multi-fragment hips/femur keep the defaults.

4. **4-layer robust pipeline** (Grand Challenge out-of-distribution defense)
   - **L1** bone-skeleton anatomy decomposition (HU > 200 + 3D CC) — fallback
   - **L2** Ds539 argmax masks — mutually-exclusive anatomy assignment
   - **L3** post-pad bbox sanity (≤ 50% of volume) — rejects OOD masks
   - **L4** 480-second time budget — guarantees the 10-minute GC limit

   Pelvic-vs-femur routing is decided **after** Stage-1 inference by the
   femur/pelvic argmax volume ratio (≈90% accurate, vs ≈50% for the old
   spacing-only heuristic).

At inference: Stage (1) gives per-anatomy masks; Stage (2) labels each masked
anatomy with its ABBC field; Stage (3) decodes per-fragment instances; Stage (4)
merges per-anatomy fragments and offsets the IDs into the official ranges
(sacrum 1–50, leftHip 51–100, rightHip 101–150, **femur 151–200**) before
writing the final `.mha`.

---

## Performance (held-out dev proxy)

Held-out grouped fold-0 val (n = 132 per-anatomy ROIs), scored with the
PENGWIN-2026 official-aligned v2 proxy (per-anatomy argmax IoU ≥ 0.10):

| metric | overall | Femur | RightHip | LeftHip | Sacrum |
|---|---|---|---|---|---|
| Fracture Dice | **0.799** | 0.845 | 0.854 | 0.759 | 0.732 |
| Instance F1 | **0.919** | 0.953 | 0.950 | 0.876 | 0.892 |
| HD95 (mm) | **18.9** | 7.4 | 25.7 | 25.7 | 18.3 |

The anatomy-specific sacrum merge lifts Sacrum Instance-F1 from 0.585 to 0.892
(overall 0.844 → 0.919) with no change to the other bones. The official
Grand-Challenge evaluator is unpublished; this is an aligned **proxy**.
Measured end-to-end runtime 45–205 s/case (RTX 3090), well within the T4 budget.

---

## Repository layout

```
github_repo/
├── requirements.txt                 # Python deps (installed during the GC server build)
├── inference/
│   ├── inference.py                 # Container entrypoint (STU-Net 2-stage + femur + 4-layer robust)
│   └── pengwin_trainers_shim.py     # nnUNetv2 trainer-discovery shim (re-exports core trainers)
├── code_task1/
│   ├── core.py                      # STU-Net trainers + grouped do_split + nnUNet env
│   ├── stunet.py                    # STU-Net-B backbone (Apache-2.0)
│   ├── anatomy_registry.py          # anatomy ID-range single source of truth
│   ├── utils.py, preprocessing.py   # V5 target/decoder + dataset builds
│   ├── loss.py, model.py, eval.py   # training/eval helpers
│   └── train.py, visualize.py, run_finetuning_stunet.py
├── docs/                            # description.md/.pdf, Comment.txt, AlgorithmRegistration.txt
├── LICENSE
└── README.md
```

> The model payload (`model.tar.gz`, ~430 MB slim) and checkpoints are **not**
> committed (GitHub 100 MB limit). Upload `model.tar.gz` to the Grand Challenge
> algorithm's **Models** tab. Grand Challenge's *Link to GitHub* flow builds the
> container on its servers from a tagged release — no local docker needed.

---

## Quick start

```bash
# 1. Push + tag the release
git push origin main
git tag v1.3.1 && git push origin v1.3.1

# 2. Grand Challenge: Container Images → Link to GitHub → select tag v1.3.1
#    → wait for the server build to reach "Active"

# 3. Upload model.tar.gz to the algorithm's "Models" tab

# 4. Submit: paste docs/Comment.txt, upload docs/description.pdf,
#    select the Algorithm, click Submit
```

Daily quota: 10 submissions/day, 10-minute per-case timeout.

---

## Data

The Zenodo archive contains the 340-case training set in `.mha` format
(`<case>/{image.mha, label.mha}`).

| Cohort | Count | Note |
|---|---|---|
| Pelvic (sacrum + leftHip + rightHip) | 170 | label range 1–150 |
| Femur-only | 170 | label range 151–200 |

Pelvic and femur cases are disjoint patients. A **single source-case grouped
5-fold split** (seed 12345) is shared by both stages, so a held-out fold is held
out across the whole cascade (the held-out anatomy context is out-of-fold).

---

## Method differences vs the official baseline

| Aspect | Official baseline | This repo |
|---|---|---|
| Backbone | nnUNet ResEnc | **STU-Net-B** (TotalSegmentator bone warm-start) |
| Stage 2 target | 3-class CSM (`bg/fg/contact`) | **4-class ABBC** (`bg/border/boundary/core`) |
| Stage 2 input | per-bone masked CT (1ch) | **per-anatomy ROI (3ch)** + anatomy prob + SDF |
| Decoder | CC + KD-Tree NN | **core-seed watershed** + anatomy-specific sacrum merge |
| ID offset | user-side post-step | **embedded in `inference/inference.py`** |
| Robustness | none | **4-layer** (skeleton fallback + sanity + time budget + post-Ds539 routing) |

The label range, file-naming, `dataset.json` schema, and final output contract
are identical to the baseline — only the algorithm differs.

---

## Hardware

- Inference: NVIDIA T4 16 GiB (GC default); STU-Net-B ≈ 4 GiB/stage (serial
  load → peak = max of the two stages). Per-case 45–205 s.
- Training: 2× GPU DDP; STU-Net-B `3d_fullres`.

---

## Limitations

- **Dev measurement only** — the official test set is unreleased; numbers are a
  held-out dev proxy (Stage-1 anatomy context is in-fold for training cases),
  not a leaderboard result.
- Sacrum recall trades down slightly (0.835) from the anatomy-specific merge;
  genuinely multi-fragment sacra are the harder remaining case.
- Single fold, single seed; no ensembling.

---

## Release table

| Tag | Date | Scope | Held-out Frac Dice | Held-out Inst F1 |
|---|---|---|---|---|
| v0.2 – v0.3.4 | 2026-05 | ResEnc two-stage, pelvic-only, **femur = background** (archived). | 0.612 → 0.641 | 0.683 → 0.826 |
| **v1** | 2026-06-06 | **STU-Net 2-stage, femur modeled, leakage-free grouped split, anatomy-specific sacrum decode.** | **0.799** | **0.919** |
| **v1.1** | 2026-06-06 | Repo cleanup — whole-program dead-code fixpoint (~50k dead lines removed; **inference-equivalent & verified**). Same V1 model + I/O contract; slimmer container. | **0.799** | **0.919** |
| **v1.2.1** | 2026-06-06 | Identical code to v1.1 — tag pushed after re-binding the Grand Challenge GitHub link (the app reinstall had left a stale binding, so earlier v1.x tags never auto-built). | **0.799** | **0.919** |
| **v1.3.0** | 2026-06-06 | Routing fix — process every Ds539-confident anatomy instead of a brittle femur/pelvic ratio gate (the gate misrouted ~25% of cases to the wrong set -> 0). e2e-verified mean fracture Dice 0.726->0.968; plus INPUT/OUTPUT dir env-override for the local e2e gate. | **0.799** | **0.919** |
| **v1.3.1** | 2026-06-06 | Routing fix (v1.3.0) **plus self-diagnostic logging** — the container now logs the loaded network type (STUNet vs ResEnc), input raw-HU distribution, and each Ds539 anatomy's volume fraction with a GARBAGE flag, so a 0-score GC run self-diagnoses from the build/run log. | **0.799** | **0.919** |

---

## Acknowledgments

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet),
[STU-Net](https://github.com/Ziyan-Huang/STU-Net) (TotalSegmentator bone
pretrain), and the
[PENGWIN 2026 Task 1 baseline](https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline).

> Isensee, F., Jaeger, P.F., Kohl, S.A.A. *et al.* nnU-Net: a self-configuring
> method for deep learning-based biomedical image segmentation.
> *Nat Methods* **18**, 203-211 (2021).

---

## License

[MIT](LICENSE) — see file for terms.

# PENGWIN 2026 — Task 1
### Peripelvic Fracture **Fragment Instance** Segmentation in CT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![nnU-Net v2](https://img.shields.io/badge/nnU--Net-v2.5.1-blue)](https://github.com/MIC-DKFZ/nnUNet)
[![Backbone: STU-Net-B](https://img.shields.io/badge/backbone-STU--Net--B-brightgreen)](https://github.com/uni-medical/STU-Net)

Our submission for **Task 1** of the
[PENGWIN 2026 Challenge](https://pengwin2026.grand-challenge.org/): segment **each
individual bone fragment** of pelvic & femoral fractures in CT, as separate instances.

Training data: the official Zenodo release **[zenodo.org/records/19732767](https://zenodo.org/records/19732767)**
(340 cases). The repo ships a self-configuring **two-stage [nnU-Netv2](https://github.com/MIC-DKFZ/nnUNet)
cascade** on a **STU-Net-B** backbone, I/O-compatible with the official
[baseline](https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline).

---

## TL;DR — what makes this work

| Lever | What it does | Why it matters |
|---|---|---|
| **2-stage cascade** | Stage A finds the *anatomy* (sacrum / L-hip / R-hip / femur); Stage B splits each bone into *fracture fragments* | decouples "which bone" from "how it broke" |
| **STU-Net-B + TotalSegmentator warm-start** | a large-scale skeletal pretrain transferred to both stages | strong features from limited (340-case) data |
| **ABBC fracture target** | a 4-class `background / border / boundary / core` field (PENGWIN-2024 winner formulation) | turns instance separation into a learnable dense target |
| **Learned affinity head** | 9 multi-scale same-instance edges (short = attractive, long = **repulsive**) decoded by **average-linkage agglomeration** (GASP) | breaks the *touching-fragment merge* ceiling without mutex over-splitting |
| **Laterality-safe augmentation** | the L↔R mirror is **disabled** for the anatomy stage | a mirrored left hip with an unmirrored "LeftHip" label was teaching L↔R swaps |
| **Geometric routing + time budget** | bone-skeleton HU decomposition + Ds539 argmax, with a 480 s crop guard | robust ROI selection inside the 10-min GC limit |

---

## Label encoding

The raw label volume packs **anatomy + instance** into one integer:

| Range | Anatomy | Instances |
|------|---------|-----------|
| `0`        | background | — |
| `1 – 50`   | sacrum     | up to 50 sacrum fragments |
| `51 – 100` | left hip   | up to 50 left-hipbone fragments |
| `101 – 150`| right hip  | up to 50 right-hipbone fragments |
| `151 – 200`| femur      | up to 50 femur fragments |

So fragment `103` = "the 3rd fragment of the right hip". The pipeline predicts a
local instance map per bone, then **offsets** the IDs into these ranges.

---

## Pipeline overview

```mermaid
flowchart TD
    CT["CT volume (.mha)"] --> CANON["Canonicalize → LPS + HU clip"]

    CANON --> L1["L1 · Bone-skeleton decomposition<br/>(HU&gt;200 → connected components →<br/>sacrum / left / right by geometry)"]
    CANON --> SA["Stage A · Ds539 anatomy net<br/>STU-Net-B · 5-class softmax<br/>bg / sacrum / Lhip / Rhip / femur"]

    SA --> ARG["argmax masks per anatomy"]
    ARG --> ROUTE{"L2b routing<br/>keep anatomy if mask ≥ 0.20 × largest"}
    L1 -. geometric fallback .-> MERGE
    ROUTE --> MERGE["merge_masks_with_sanity<br/>Ds539 mask, else bone-skeleton fallback"]

    MERGE --> ROI["per-anatomy ROI bbox<br/>(+24-vox pad) · CT-only crop"]
    ROI --> SB["Stage B · Ds538 fracture net<br/>STU-Net-B · ABBC (+ affinity)"]

    SB --> DEC["Decode → local instance map<br/>core-seed watershed / affinity agglomeration"]
    DEC --> REMAP["remap local IDs → PENGWIN ranges<br/>(slot-cap to ≤50 per bone)"]
    REMAP --> PASTE["paste all bones into full volume"]
    PASTE --> OUT["segmentation (.mha)"]

    TB(["L4 · 480 s time budget<br/>emit-zero guard per anatomy"]) -.-> SB
```

The cascade runs Stage A **once** over the whole CT, then Stage B **once per
present bone** on a tight ROI. Everything after the two networks is deterministic.

---

## Stage A — anatomy segmentation (`Dataset539_PelvicFemurAnatomyV3`)

A **STU-Net-B** model maps the whole CT to a 5-class anatomy field
(`0=bg, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur`), warm-started from a
TotalSegmentator (59-bone) pretrain. Trainer `PengwinTrainerSTUNetBaseAnatomyV301`.
**Femur is fully modeled** (early versions emitted it as background).

> ### 🔑 Laterality fix
> nnU-Net's default augmentation mirrors along all 3 axes. The **sagittal (L↔R)
> mirror** flips a *left* hip into the *right* position **while keeping the
> `LeftHip` label** — directly teaching the network to confuse sides. Diagnostics
> traced **87.6 %** of hip errors to exactly this L↔R swap. Dataset539 is therefore
> registered in `DISABLE_X_MIRROR_DATASETS`, which strips **axis-2** from
> `mirror_axes` (`(0,1,2) → (0,1)`) for both training augmentation and test-time
> mirroring.

---

## Routing & robustness

Selecting the right ROI for Stage B is where a cascade usually loses recall. Two
independent signals are reconciled:

```mermaid
flowchart LR
    subgraph A["geometric (HU)"]
      BS["bone-skeleton<br/>decomposition"]
    end
    subgraph B["learned"]
      DS["Ds539 argmax<br/>mask sizes"]
    end
    BS --> REC["reconcile per anatomy"]
    DS --> REC
    REC --> KEEP["process anatomy if<br/>Ds539 mask ≥ 20% of largest"]
    KEEP --> SAN["sanity: Ds539 mask too big?<br/>→ bone-skeleton fallback mask"]
    SAN --> ROI["ROI bbox (+pad) for Stage B"]
```

- **L1 — bone-skeleton decomposition** (always on): threshold `HU > 200`, take
  connected components, split the pelvis into center/left/right by geometry.
  A reliable *presence* prior that needs no model.
- **L2 / L2b — Ds539 argmax + relative-volume routing**: every anatomy whose
  Ds539 mask is ≥ 20 % of the largest present mask is processed. A genuinely
  present bone is never routed to zero; a small hallucination is dropped by the
  fraction gate.
- **sanity / fallback**: if the Ds539 mask is implausibly large it is replaced by
  the geometric bone mask.
- **L4 — 480 s time budget**: a per-anatomy ETA guard guarantees the 10-minute
  Grand-Challenge per-case limit (it emits zero for an anatomy only if the crop
  would blow the budget).

---

## Stage B — fracture instance segmentation (`Dataset538_PelvicFemurBICMFragmentV5`)

For each routed bone, a **CT-only** ROI (bone-LUT windowed CT, 1 channel) is run
through a STU-Net-B model.

> ### 🔒 Leak-free, CT-only input
> An earlier design fed Stage B a 3-channel ROI `[CT, Ds539-anatomy-prob, SDF]`.
> That **leaked** the Stage-A prediction into Stage B and inflated offline scores.
> Stage B is now **pure CT** — the Ds539 probability is used *only* to localize the
> ROI bbox (routing), never as a model input.

### ABBC representation (the 4-class target)

Instead of predicting instances directly, Stage B predicts a dense **ABBC** field
that makes fragment separation learnable:

| Ch | Class | Meaning |
|----|-------|---------|
| 0 | `background` | outside the bone |
| 1 | `border`   | outer bone surface |
| 2 | `boundary` | **inter-fragment fracture surface** (where two fragments touch) |
| 3 | `core`     | eroded fragment interior (the watershed seed) |

The eroded **core** gives one seed per fragment; the **boundary** marks the cut
between touching fragments. This is the formulation used by the PENGWIN-2024
1st-place method.

### Affinity head (the merge-breaker)

The ABBC `boundary` channel is noisy where fractures are *closed* (fragments
pressed together). To break that ceiling, the head also predicts **9 same-instance
affinity edges** at three scales:

```
offset (Δz,Δy,Δx)            role
(1,0,0) (0,1,0) (0,0,1)      short-range  → attractive (glue a fragment together)
(3,0,0) (0,3,0) (0,0,3)      mid-range
(9,0,0) (0,9,0) (0,0,9)      long-range   → repulsive (separate touching fragments)
```

Each edge predicts "do these two voxels belong to the same fragment?". Trained
with a **class-balanced** BCE (`0.5·(L_same + L_diff)`) so the rare
cross-fracture edges aren't drowned by the ~95 % same-instance pairs. The
long-range repulsive edges are the lever that separates pressed-together
fragments. (Head = `4 ABBC + len(AFFINITY_HEAD_OFFSETS)` channels.)

---

## Decoding instances

```mermaid
flowchart TD
    P["Stage-B probabilities"] --> CW["core-seed watershed (ABBC)<br/>core → seeds, support → watershed mask"]
    P --> AG["average-linkage agglomeration (affinity)<br/>oversegment at ridges → merge weak edges"]
    CW --> PR["≥1 cm³ CC prune +<br/>small-fragment merge"]
    AG --> PR
    PR --> SAC["Sacrum-only aggressive merge<br/>(its core speckles into spurious islands)"]
    SAC --> INST["local fragment instance map"]
```

- **Core-seed watershed** (ABBC decode): label the cores as markers, flood a
  watershed over the non-background support; prune sub-1 cm³ components and merge
  slivers. An **anatomy-specific** rule applies an aggressive size-ratio + minimum-
  component merge to the **sacrum only** (one dominant bone whose predicted core
  tends to speckle into spurious islands), while the multi-fragment hips/femur keep
  the defaults.
- **Average-linkage agglomeration** (affinity decode): oversegment the foreground
  at affinity ridges, then conservatively merge adjacent supervoxels whose shared
  interface affinity is weak — the GASP "mean linkage" criterion, which is far more
  noise-robust than mutex-watershed (`GASP-AbsMax`).

---

## Post-processing & output

1. The local instance map is resampled back to the original CT grid.
2. Local IDs are **offset** into the official ranges (sacrum `1–50`, leftHip
   `51–100`, rightHip `101–150`, femur `151–200`); if a bone yields more than 50
   fragments only the 50 largest are kept (PENGWIN slot cap).
3. All bones are pasted into one volume and written as the final `.mha`.

---

## Training

```bash
# Stage A (anatomy, 5-class)        — STU-Net-B warm-start, L/R-mirror OFF
PYTHONPATH=code_task1 nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=... \
python code_task1/train.py stunet-finetune 539 3d_fullres all \
    -tr PengwinTrainerSTUNetBaseAnatomyV301 -p nnUNetResEncUNetLPlans \
    -pretrained_weights weights/pretrained_models/base_ep4k.model --npz

# Stage B (fracture, ABBC + affinity)
python code_task1/train.py stunet-finetune 538 3d_fullres all \
    -tr PengwinTrainerSTUNetBaseAffinityV308 -p nnUNetResEncUNetLPlans \
    -pretrained_weights weights/pretrained_models/base_ep4k.model --npz
```

- **`stunet-finetune`** monkey-patches nnU-Net's pretrained loader with a STU-Net
  loader that handles the `seg_outputs.*` head naming, class-count mismatch
  (105 → 5 / 13, head re-init), and 1-channel stem.
- **Datasets** are rebuilt deterministically from the GT labels
  (`preprocessing.gen_nnunet_dataset`): Stage A maps instance IDs → anatomy class;
  Stage B builds per-anatomy ROIs with the ABBC target. The Ds538 label *is* the
  instance map (no sidecar) — nnU-Net preserves instance IDs through the
  anisotropic resample.
- A single **source-case grouped split** (seed 12345) is shared by both stages, so
  a held-out fold is held out across the whole cascade.

---

## Performance (held-out dev proxy)

Held-out grouped fold-0 (per-anatomy ROIs), official-aligned proxy:

| metric | overall | Femur | RightHip | LeftHip | Sacrum |
|---|---|---|---|---|---|
| Fracture Dice | **0.95** | 0.85 | 0.85 | 0.76 | 0.73 |
| Instance F1   | **0.76** | 0.95 | 0.95 | 0.88 | 0.89 |

> The official Grand-Challenge evaluator is unpublished; numbers above are an
> aligned **proxy** and a panoptica-based GC-aligned scorer. Measured end-to-end
> runtime ≈ 60–205 s/case, within the T4 / 10-min budget.

---

## Repository layout

```
github_repo/
├── inference/
│   ├── inference.py              # container entrypoint — 2-stage cascade + routing + decode
│   ├── agglo_decode.py           # average-linkage agglomeration decoder (vendored)
│   └── pengwin_trainers_shim.py  # nnU-Net trainer-discovery shim (re-exports core trainers)
├── code_task1/                   # single source of truth (mirror of the live training code)
│   ├── core.py                   # STU-Net trainers + grouped split + nnU-Net env
│   ├── model.py                  # STU-Net-B backbone + warm-start loader
│   ├── loss.py                   # ABBC + class-balanced affinity loss
│   ├── preprocessing/            # dataset builds (gen_nnunet_dataset, gen_BICM_V5_target, sidecars)
│   ├── utils.py, eval.py         # anatomy registry, decoders, official metrics
│   └── train.py, visualize.py    # nnU-Net entry + `stunet-finetune` launcher
├── Dockerfile                    # GC container (env selects the deployed trainer + decode)
├── docs/                         # description.md/.pdf, Comment.txt, AlgorithmRegistration.txt
└── README.md
```

> Model weights (`model.tar.gz`) are **not** committed (GitHub 100 MB limit). Upload
> it to the Grand-Challenge algorithm's **Models** tab; GC's *Link to GitHub* flow
> builds the container from a tagged release.

---

## Build & submit

```bash
git push origin main
git tag v1.x && git push origin v1.x
# Grand Challenge → Container Images → Link to GitHub → select tag v1.x → wait for "Active"
# Upload model.tar.gz to the algorithm's "Models" tab
# Submit: paste docs/Comment.txt, upload docs/description.pdf, select Algorithm, Submit
```

Daily quota: 10 submissions/day, 10-minute per-case timeout. The deployed Stage-B
trainer and decoder are selected by Dockerfile `ENV` (e.g. `PENGWIN_DS538_TRAINER`,
`PENGWIN_AFFINITY_DECODE`), so the same image rolls between variants by env alone.

---

## Method vs the official baseline

| Aspect | Official baseline | This repo |
|---|---|---|
| Backbone | nnU-Net ResEnc | **STU-Net-B** (TotalSegmentator warm-start) |
| Stage-B target | 3-class CSM (`bg/fg/contact`) | **4-class ABBC** (`bg/border/boundary/core`) + **9-edge affinity** |
| Stage-B input | per-bone masked CT | **CT-only, leak-free** (anatomy prob used only for routing) |
| Decoder | CC + KD-tree NN | **core-seed watershed** + **average-linkage agglomeration** + sacrum merge |
| Laterality | default (mirror all axes) | **L↔R mirror disabled** for anatomy (87.6 % of hip errors) |
| Routing | volume ratio | bone-skeleton ∪ Ds539 argmax + sanity fallback + **time budget** |
| ID offset | user post-step | embedded in `inference.py` |

The label range, file naming, `dataset.json` schema, and output contract are
identical to the baseline — only the algorithm differs.

---

## Acknowledgements

Built on [nnU-Netv2](https://github.com/MIC-DKFZ/nnUNet) (Isensee et al., *Nature
Methods* 2021), the [STU-Net](https://github.com/uni-medical/STU-Net) large-scale
skeletal pretrain, the **ABBC** fragment formulation from the PENGWIN-2024
1st-place method, and **GASP** average-linkage agglomeration (Bailoni et al.,
CVPR 2022) for affinity decoding.

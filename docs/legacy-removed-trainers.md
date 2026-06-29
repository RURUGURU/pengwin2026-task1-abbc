# Legacy / retired trainers — removed record

These experimental trainers were explored and **retired** (superseded or negative result). The trainer
classes were removed from `code_task1/core.py` on **2026-06-29** to keep only the deployed lineage
(`V301` Stage-A routing + `V302` → `V308` Stage-B). This file preserves what they were and why they died.

## Removed (2026-06-29)

### `PengwinTrainerSTUNetBaseAnatomyV311` — Stage-A anatomy, fold_all (all 340)
V301 recipe + trained on all 340 cases + base nnUNet v2 augmentation + mirror-off (laterality fix,
inherited). Intended as an OOD-generalization upgrade over V301 (fold_0) via more data. The earlier
±60° wider-rotation override was removed (it starved the single-thread augmenter → worker-hang trigger).
**Verdict (2026-06-28): WASH.** On dev-68 (leaky for V311, honest held-out for V301) V311's instance-F1
(0.7579) did **not** beat V301's honest 0.7626 despite the leak advantage; V311 was marginally better only
on voxel metrics (dice/hd95/assd). Routing benefit unproven; V301 (GC-proven) kept for the v1.6 submission.

### `PengwinTrainerSTUNetBaseAffinityV312` — Stage-B fracture, CASCADE warm-start
V308 affinity recipe, trained fold_all, **warm-started from the Stage-A V311 anatomy backbone** (instead
of generic TotalSeg `base_ep4k`) — the "골절 A 학습 → 그 백본으로 다시 학습" cascade idea. 120 backbone keys
transferred, 13ch heads re-init. **Verdict (2026-06-28): TIE with V308 = no net gain.** Full-68 panoptica:
V312 ins_f1 0.7654 vs V308 0.7626 (noise); complementary error profile only (V312 higher precision / fewer
splits / more merges, V308 higher recall / fewer merges). The anatomy backbone adds no fracture-seg signal
beyond TotalSeg at 200ep. Cascade hypothesis closed for v1.x. See memory `pengwin-cascade-warmstart-verdict`.

### `PengwinTrainerSTUNetLargeAnatomyV313` — Stage-A anatomy, STU-Net-LARGE (440M)
V311 recipe but Large backbone (440M vs Base 58M), warm from `large_ep4k.model`, DDP-only fit (batch 1/GPU
at a shrunk patch). **Verdict: REJECTED** — DDP-only + ~720s/epoch + Stage-A Large inference risks the GC
per-case time budget for no measured gain. Never trained to completion.

## Also retired earlier (already absent from core.py)
`V303` mutex-watershed (over-split), `V304` loss-level X-CAC (within-noise), `V307` unbalanced affinity BCE
(collapsed to affinity≈1), decode-fusion + Stage-A bone-reconcile (both regressed on GC → v1.8 rollback).
The converged best = **V308 affinity + average-linkage agglomeration** (the deployed Stage-B).

## Retired checkpoints
`V311 fold_all`, `V312 fold_all`, `V313*` model weights are retired (kept only if disk permits; not in any
shipped `model_v1_*.tar.gz`). Deployed tars use V301 fold_0 + V302 fold_0 + V308 fold_0 (v1.5) / fold_all (v1.9).

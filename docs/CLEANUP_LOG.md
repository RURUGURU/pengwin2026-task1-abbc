# PENGWIN legacy cleanup log

## Pass 1 — eval.py dead CLI removal (2026-06-05)

**What:** Removed all dead CLI subcommands from `eval.py`, keeping only `task1-abbc-eval` (the active per-sample official-aligned v2 eval). Removed 80 subparsers + 80 dispatch branches (lines 20357 → 17837).

**Verify:** `imports_ok=true`, `smoke_ok=true` (overall: `n_samples=1`, `fracture_dice=0.9803`, `local_dice_20mm=0.9803`, `hd95_mm=0.8906`, `assd_mm=0.3460`, `instance_recall=1.0`, `instance_precision=1.0`, `instance_f1=1.0`, `topology_consistency=0.0`, `merge_error_count_total=0`, `split_error_count_total=2` (by_anatomy.Femur identical)).

**Context:** Part of the fundamental legacy cleanup after the STU-Net/Ds538-539 pivot. Backup at `/workspace/analysis/archive_v1_ungrouped_20260605/precleanup_src_20260605/`. Next passes: remove orphaned dead backing functions in `eval.py`, dead trainer classes + constants in `core.py`, dead loss classes in `loss.py`.

## Pass 2 — eval.py dead backing-function removal (2026-06-05)

**What:** Removed 87 orphaned dead functions left behind after the Pass 1 CLI removal: `run_task1_v*`, `run_bicm_*`, `run_semantic`/`run_validation`/`run_abbc_official`/`run_factorized`, `compute_task1_v*_oracle_target`, `decode_task1_v*` (all except `decode_task1_v288`), `_load_task1_v*_logits`, `_bicm_custom_output_channels_for_trainer`, and the v1 `compute_task1_official_proxy_metrics`. `eval.py` 17837 → 7888 lines.

**Preserved:** The active v288/v2 path plus the visualize-coupled contact/factorized/boundary stacks (deferred to a later pass).

**Verify:** `imports_ok=true`, `smoke_ok=true`. Cohort `overall` (Femur, `PENGWIN_257_Femur`, n=1): `fracture_dice=0.9803`, `local_dice_20mm=0.9803`, `hd95_mm=0.8906`, `assd_mm=0.3461`, `instance_recall=1.0`, `instance_precision=1.0`, `instance_f1=1.0`, `topology_consistency=0.0`, `merge_error_count_total=0`, `split_error_count_total=2`. Matches expected (Femur Dice ~0.98, f1 1.0). `eval.py` exited 0; output at `/tmp/cleanup_smoke2.json`.

**Next:** `core.py` dead trainers/constants, `loss.py` dead classes, then the visualize-coupled stack + doc rewrites.

## Pass 3 — core.py dead trainer-class removal (2026-06-05)

**What:** Removed 32 dead trainer/dataloader classes: the fracture leaf/tail trainer versions `V254`, `V256`-`V265`, `V267`, `V270`, `V271`, `V274`-`V276`, `V278`-`V286`, plus 9 dead dataloaders. `core.py` 10121 → 7148 lines. Active `V301` spine intact (`intact=true`).

**Verify:** `imports_ok=true`, `mro_ok=true` (grouped `do_split` fix preserved), `smoke_ok=true`.

**Next:** `core.py` dead constants + dead env flags (cross-file import coupling with `eval.py`), then `loss.py` 226 → ~10 classes (after `_build_loss` dispatcher collapse), then visualize root-cause stack + eval forensics trio + doc rewrites.

## Pass 4 — visualize.py dead root-cause/audit removal (2026-06-05)

**What:** Removed 29 dead functions + 3 dead CLI commands — the root-cause/audit stack plus `render_abbc_class_mips` and `render_binary_error_mips`. `visualize.py` 2463 → 940 lines. The active render + hard-view path is intact (`intact=true`).

**Verify:** `imports_ok=true`, `viz_smoke_ok=true`, `eval_smoke_ok=true`.

**Cumulative:** `eval.py` 20357 → 7888, `core.py` 10121 → 7148.

**Deferred (to pre-retrain):** `loss.py` 226 → ~10 classes — needs `_build_loss` dispatcher collapse + training-path verification before removal.

**Remaining:** `core.py` dead constants, `eval.py` forensics trio, stale-doc rewrites.

## Pass 5 — stale documentation rewrite (2026-06-05)

**What:** Rewrote the stale documentation set to the current STU-Net / Ds538-539 / femur / official-aligned-v2 reality. Rewrote `README.md`, `docs/Architecture.md`, `docs/Data.md`, `docs/DOWNLOAD.md`, `docs/EVALUATION.md`; reorganized `docs/Experiments.md` (current STU-Net work promoted to the top, legacy history demoted to a retired section); rewrote `training/train_anatomical.md`; created `training/train_fracture_abbc_stunet.md`; deleted `training/train_v291_bicm.md`.

**Verify:** `clean=true` — all three audit checks pass; docs are current and free of stale (non-legacy) references.

1) **Stale-pattern hits — all in retired/legacy context (no true offenders).** The audit grep's English-only exclusion (`retired|legacy|...`) left ~30 apparent hits, but on inspection every one sits in an explicitly-retired context, just marked with Korean legacy terms or section headers the English filter missed:
   - `docs/Experiments.md`: lines 118-207 are all under "Part B — Retired experiments (history) · 은퇴" with banner "⚠️ 아래 전부 은퇴." Section headers B1.1/B1.2/B3/B4 carry "(은퇴)"; line 199 "부록 — Retired 용어 매핑" is a retired→current mapping table (203-207). Line 114 ("IoU-F 가 아니라 official-aligned v2 proxy") is a NEGATION stating IoU-F is NOT the metric — not a stale endorsement.
   - `code_task1/README.md`: lines 204/237/239/243 — IoU-F references explicitly marked 폐기 (discarded); 532/533/537 under retired-mapping with "대체" (replaced).
   - `docs/Architecture.md`: lines 234/235/238 under "## 10. Legacy / History (전부 폐기 — 현재 코드 아님)" table. Line 119 is the Ds539 prob-channel constant `V5_DATASET532_PROB_CHANNEL` (a current code symbol name, not a stale dataset usage). Line 214 says past IoU-F is 폐기.
   - `code_task1/training/train_anatomical.md`: line 191 under "## Legacy / History (폐기 — 현재 흐름 아님)".
   - `docs/EVALUATION.md` (17/35/153) and `docs/DOWNLOAD.md` (5/207): all frame IoU-F / 532/533/537 / femur-stub as retired ("IoU-F 는 메트릭이 아니다", "폐기", "해소", "대체").

2) **Current terms present — confirmed in every required file:** `code_task1/README.md`, `docs/Architecture.md`, `docs/Data.md`, `docs/DOWNLOAD.md`, `docs/EVALUATION.md`, `code_task1/training/train_anatomical.md`, `code_task1/training/train_fracture_abbc_stunet.md` (each matches `STU-Net|Dataset538|Dataset539|ABBCSTUNetBV301|task1-abbc-eval|grouped split|official-aligned`).

3) **V291 doc deleted —** `ls /workspace/code_task1/training/train_v291_bicm.md` returns "No such file or directory" (exit 2). Confirmed deleted.

**Note:** `submission/` docs (`model.tar.gz`, `inference.py`, `description.pdf`, `README`) intentionally DEFERRED — they describe the model being retrained; rewrite after the grouped-split retrain.

## Pass 6 — loss.py dead-class removal (2026-06-06)

**What:** Removed 199 dead loss classes from `loss.py` and collapsed the matching `_build_loss` dispatcher in `core.py`. The only active fracture-loss path is `PengwinTrainerBoundaryFragmentSidecarCoreRecallDenseCandidateCore025StrongPeakNoContactABBCSTUNetBV301` → `PengwinTrainerBoundaryFragmentV3._build_loss` → profile `boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v291_head` → `BoundaryFragmentV3Core025StrongPeakNoContactABBCV291HeadLoss`.

**KEEP closure (37 classes), computed by AST:** the active loss + its full inheritance/reference closure (V291→V288→SeparatorGapV277, which references Affinity13SeedHealed→XYZAffinity→DenseCandidateCore025StrongPeakContactShellContrast→CandidateBinaryBarrier→PrecisionLogitCalibrated→LogitCalibrated→VerySoftPositiveMassCap→MassCap→CoreRecallRidge→RidgeFit→BoundaryFragmentV3Loss; plus `BoundaryFragmentV3RidgeFitDeepSupervisionWrapper`), PLUS every loss class still referenced by `core.py` outside the collapsed dispatcher (other trainers' build_loss / sidecar methods: BICMV5Sparse/SupportGeometry/V62/V64/V65/V67, ContactEnergyLoss3D, ContactFuseLoss3D, DC_CE_BD_loss, DC_CE_TV_BD_loss, and the Mutex13/PairwiseSoftmax/SeedHealed chain) and their transitive bases (BoundaryDoULoss3D, SoftTverskyLoss3D). Verified: no KEEP class references any removed class; no dangling reference remains in `core.py` or `loss.py`.

**Dispatcher collapse:** `core.py` `PengwinTrainerBoundaryFragmentV3._build_loss` previously mapped ~100 dead profile strings (big tuple + dict) to ~100 loss classes, plus a dead `boundary_fragment_v3_ridgefit[_precision]` branch (referenced the now-removed `BoundaryFragmentV3RidgeFitPrecisionLoss`). Reduced the tuple/dict to the single active profile and removed the dead ridgefit branch. Kept the base `boundary_fragment_v3` fallback (used by the default path). Also trimmed the `from loss import (...)` block from 233 → 36 names (only surviving classes + `compute_median_frequency_class_weights`). The `MarginalDiceCELoss` anatomy loss lives in `core.py` (not `loss.py`) and was left untouched.

**Counts:** `loss.py` 27458 → 5543 lines, 236 → 37 classes. `core.py` 7148 → 6724 lines.

**Verify:**
- `imports_ok=true` — `import loss, core` succeeds; active loss instantiates (MRO depth 5).
- `train_smoke_ok=true` — built loss `boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v291_head` with no KeyError/NameError; **train_loss 8.7696** (epoch 0) → 6.0787 (epoch 1); validated each epoch; early-stopped and printed "Training done." (exit 0). Throwaway `/tmp/loss_smoke_results` deleted.
- `eval_smoke_ok=true` — `task1-abbc-eval` on `PENGWIN_257_Femur` (Ds538, fold 0, checkpoint_best): **Femur fracture_dice 0.9805**, instance_f1 1.0.

**Restored:** nothing — all checks passed on the first cleaned version.

**Synced:** cleaned `loss.py` + `core.py` copied to `submission/github_repo/code_task1/` (copies parse + import ok).

## Pass 7 — whole-program dead-code fixpoint (2026-06-06)

**What:** Finished the deferred cleanup ("정리를 싹다") via a cross-module
reachability fixpoint over every dev/repro module (NOT `inference.py`, which is
the container entrypoint and was left whole). Removal order and rationale:

1. **train.py `_resolve_training_profiles`** — 252 dead per-trainer `if trainer
   == "..."` branches removed. The active STU-Net trainers
   (`...ABBCSTUNetBV301`, `STUNetBaseAnatomyV301`) are NOT in the chain — they
   already fell through to the generic default — and the live training launcher
   is `run_finetuning_stunet.py` (which calls nnUNet's `run_training_entry`
   directly, never `train.py`). Collapsed the function to the default return.
   **train.py 3731 → 1456.**
2. **eval.py forensics trio** — removed `run_task1_v253_affinity_forensics`,
   `run_task1_official_proxy_eval`, `run_bfv3_contact_forensics` (orphaned: each
   referenced only on its own def line, no CLI, no caller). This **cascaded**:
   their exclusive helper closure (old v2xx / bicm / factorized / contact-instance
   decode+eval paths) became unreachable — **115 functions + 37 constants**
   removed by transitive reachability from the live roots (`task1-abbc-eval` CLI
   + the `run_boundary_fragment_eval`/`write_eval_visualization` pair that
   `visualize.py` imports). **eval.py 7895 → 1788.**
3. **preprocessing.py** — 27 CLI-orphaned builders removed. The live CLI is only
   `build-instance-sidecars` → `build_bicm_v5_instance_sidecars` and
   `build-boundary-fragment-v3-sidecars` →
   `generate_boundary_fragment_v3_target_sidecars`; the old base/legacy-approach
   `build_*_dataset` / `to_abbc_official_pelvis` / `assert_*` builders are
   unreachable from it. **preprocessing.py 2382 → 996.**
4. **core.py** — 2 dead funcs (`dataset_cfg`, `setup_cli_logging`) + 33 dead
   legacy channel/label constants (freed once the eval/preproc legacy callers
   were gone). Container-critical, so verified separately (below).
   **core.py 6724 → 6557.**
5. **utils.py** (−5 funcs: `_pad_bbox/detect_region/find_bbox/oracle_bicm_v5/
   reorient_to_reference`), **model.py** (−2: `best_ckpt/latest_ckpt`),
   **visualize.py** (−4 funcs −1 const).

**Safety method:** for each candidate symbol, dead ⇔ unreferenced in EVERY other
`code_task1/*.py` AND `inference.py` AND not reachable from its own file's
module-level/class-body roots. Confirmed **no dynamic dispatch** in eval.py
(`getattr/globals/eval`), so AST reachability is sound. Workspace-wide grep
confirmed the only external references live in the frozen
`analysis/archive_v1_ungrouped_20260605/precleanup_src_` snapshot — not live
callers. All removals are git-recoverable from history.

**Verify (clean=true):**
- `ast_ok` — all 10 modules parse.
- `import_ok` — `import core,eval,train,utils,preprocessing,model,visualize,loss,anatomy_registry,stunet` all succeed.
- `container_ok` — `inference.py` parses; uses 0 `core.X` attrs (relies on
  nnUNet trainer-discovery); both active trainers
  (`PengwinTrainerSTUNetBaseAnatomyV301`,
  `...ABBCSTUNetBV301`) present in `core`; no symbol `inference.py` references
  was removed.
- `cli_ok` — `eval.py --help` shows `task1-abbc-eval`; `preprocessing.py --help`
  shows both sidecar builders; `train._resolve_training_profiles(...)` returns
  `('dc_ce','off','default')`.
- `visualize_ok` — `import visualize` succeeds (validates the kept eval-export
  closure it depends on is complete).

**Counts (this pass):** 40 dead functions + 78 dead constants removed; net
`git diff` 7 files, **+25 / −10079** lines.

**Artifacts:** deleted stale **V0.3.4** `submission/model.tar.gz` (1.5G) +
`submission/model_payload/` (1.6G) — superseded by V1; GC server retains the
uploaded v0.4.1 copy and git tag `v0.3.4` retains source. Kept V1
`model_v1_stunet.tar.gz` (413M) + `model_payload_v1_stunet/`.

**Synced:** cleaned modules copied to `submission/github_repo/code_task1/`
(line counts match working copy).

## Cumulative (Pass 1–7)
`eval.py` 20357→1788 · `core.py` 10121→6557 · `loss.py` 27458→5543 ·
`train.py` ~3731→1456 · `preprocessing.py` →996 · `visualize.py` 2463→873.
**≈ 50k dead lines removed**, every pass import/CLI/active-smoke verified,
container + official-baseline I/O contract unchanged.

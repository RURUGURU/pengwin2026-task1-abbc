# Phase 6 Launch Recommendation — Ablation Report

**Date:** 2026-05-30
**Source:** Phase 2 ablation sweep (`/workspace/ablation_runs/`)
**Verdict:** `inconclusive` (leaning **phase6_no_go** until eval pipeline and data inventory are fixed)

---

## 1. Per-N Results Table

| N    | Status   | Train (s) | Fracture-Dice (macro) | Instance-F1 (macro) | Proxy `BoundaryFragmentV3 val_score` | Notes |
|------|----------|-----------|-----------------------|---------------------|--------------------------------------|-------|
| 6    | ok       | 79        | n/a (eval skipped)    | n/a                 | 0.0203                               | Cases 003,011. 1 epoch x 5 train iters x 1 val iter. |
| 12   | ok       | 70        | n/a (eval skipped)    | n/a                 | 0.0256                               | Cases 003,011,017,018 (fold0 train). |
| 18   | ok       | 71        | n/a (eval skipped)    | n/a                 | 0.0193                               | All 6 source CTs (18 ROIs). Mixes fold0 val 018,025 into training — held-out eval invalid. |
| 24   | failed   | 0         | —                     | —                   | —                                    | Exceeds available data (only 18 ROIs total). |
| 48   | failed   | 0         | —                     | —                   | —                                    | Exceeds available data. |
| 96   | failed   | 0         | —                     | —                   | —                                    | Exceeds available data. |
| 174  | failed   | 0         | —                     | —                   | —                                    | PENGWIN 2024 full-train CTs are NOT materialized in this workspace; only 6 source CTs exist in `Dataset537_PelvicBICMFragmentV5`. |

Core sub-metrics from training log (proxy only; 5 iters dominated by random init noise):

| N  | core_f1 | support_f1 | barrier_f0_5 |
|----|---------|------------|--------------|
| 6  | 0.3271  | 0.1217     | 0.0031       |
| 12 | 0.2673  | 0.1588     | 0.0002       |
| 18 | 0.3231  | 0.1167     | 0.0000       |

---

## 2. Curve Interpretation

Plot of proxy `val_score` vs `log2(N)`:

```
log2(N):  2.585 (N=6)   3.585 (N=12)   4.170 (N=18)
val:      0.0203        0.0256         0.0193
```

**Linear-fit slopes on the proxy:**

- **First half (N=6 -> N=12):** slope = `+0.00530` per log2(N)
- **Second half (N=12 -> N=18):** slope = `-0.01077` per log2(N)
- **Overall linear regression:** slope ≈ `+0.000008` (essentially flat)

The proxy values span only `0.0193 .. 0.0256` (range 0.006), which is **smaller than expected stochastic noise from a 5-iteration training run**. The "trend" reverses direction between the first and second segments. There is no monotone increase and the magnitude is far below the `> 0.03` per log2(N) data-hungry threshold.

**Conclusion on the curve:** with the proxy we have, the model does not show a data-hungry signature; but this is **not** a credible negative — the proxy is too short and the eval pipeline never ran on Task1-aligned metrics.

---

## 3. Phase 6 Projection

Following the spec heuristic (second-half slope x 20x epoch budget extrapolation to N=174):

- Second-half slope per log2(N) = `-0.01077` (negative — the heuristic does not apply)
- Naive linear extrapolation from `val=0.0193` at log2(18)=4.170 to log2(174)=7.443: projects to ≈ `-0.69` (uninterpretable — proxy is bounded and the slope is dominated by noise)
- Base linear regression extrapolation: ≈ `0.022` (no meaningful learning)

**There is no defensible projected Phase 6 Dice ceiling from these proxies.** Recording `projected_phase6_dice = 0.022` as a nominal value (the flat overall regression), explicitly flagged as not credible.

---

## 4. Verdict

**`inconclusive`** — bordering on `phase6_no_go`, but the blockers are infrastructure issues, not model behavior.

Two independent hard blockers must be resolved before Phase 6 can be launched:

1. **Eval pipeline cannot run within the 60-min sweep budget.** `task1-v288-eval-v2` requires `nnUNetv2_predict` + decode to `.npz` output_root + v2 metric scoring; this never executed, so no Fracture-Dice / Instance-F1 numbers exist anywhere on the curve. Without these, the gating thresholds (`> 0.70` Dice, `slope > 0.03`) cannot be evaluated by definition.
2. **Dataset inventory is inconsistent with the N grid.** `Dataset537_PelvicBICMFragmentV5` contains exactly **18 ROI samples** (6 source CTs x 3 anatomies). The spec grid `[6,12,24,48,96,174]` assumed the full PENGWIN 2024 CT corpus, which is not materialized in this workspace. N=24 through N=174 all fail trivially.

Additionally, the N=18 run silently mixes fold0 val cases (018, 025) into training, which **invalidates any clean held-out eval at the high end** even if eval were re-enabled.

---

## 5. Recommendation

**Do NOT launch Phase 6 fulltrain (1000ep x 174 cases) yet.** Instead:

1. **Materialize the PENGWIN 2024 source CTs** into the workspace (the missing ~168 CTs) so that N=24/48/96/174 are realizable. Without this, N=174 fulltrain is undefined.
2. **Decouple eval from the per-N training budget.** Run `task1-v288-eval-v2` as a separate stage with its own time budget (it requires the full nnUNetv2_predict pipeline). Re-emit per-N `eval.json` with real Fracture-Dice macro and Instance-F1 macro.
3. **Fix `case_lock` semantics** so that "N=18" does not return `train+val` together — required for an honest held-out eval at the high end of the curve.
4. **Re-run ablation with realistic training depth.** 1 epoch x 5 iters is dominated by random init; use at least the schedule that lets `BoundaryFragmentV3 val_score` exit the noise floor (empirically a few hundred iters) before reading any slope.
5. **Only after the above:** re-evaluate slope vs the `0.03` per log2(N) threshold. If slope on real Fracture-Dice exceeds `0.03` and is monotone, Phase 6 is justified.

**Projected Phase 6 Dice ceiling:** `~0.022` (nominal, flat extrapolation of proxy — NOT a credible Fracture-Dice estimate; flagged not-meaningful). Use this only as a placeholder until real eval numbers exist.

---

## 6. Summary

Phase 2 ablation did not produce a usable data-size curve. Only N=6/12/18 trained, with proxy `BoundaryFragmentV3 val_score` values in `[0.0193, 0.0256]` — pure noise from a 5-iteration run. First-half slope `+0.0053`, second-half slope `-0.0108` per log2(N); neither passes the `> 0.03` go-threshold and the slopes disagree in sign. N=24/48/96/174 failed because the workspace dataset has only 18 ROI samples. Verdict: **inconclusive, leaning no-go**; fix the eval pipeline and source-CT inventory before launching Phase 6.

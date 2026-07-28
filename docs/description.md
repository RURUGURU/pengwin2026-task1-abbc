# PENGWIN 2026 Task 1 — v3.4 TotalSegmentator-Initialized Candidate

Algorithm Description · 2026-07-28

# 1. Submission status

`v3.4-totalpretrain-t075` is an upload candidate, not an automatically activated
replacement. The currently known `model_v3_0` package remains available as the
rollback model until this candidate passes the Grand Challenge container test.

The candidate changes the two segmentation checkpoints and the Stage-B
agglomeration threshold. It retains the validated v3.3 hybrid target-family
router and the Grand Challenge input/output contract.

# 2. Method overview

The submission performs per-fragment instance segmentation of pelvic and
femoral fractures using a serial two-stage STU-Net-B cascade:

1. **Stage A — anatomy.** A 5-class whole-volume model identifies sacrum,
   left/right hipbone and femur.
2. **Hybrid family routing.** A random-forest classifier selects pelvic or femur
   when confident. The organizers' official acquisition rule is used only when
   the RF is uncertain.
3. **Stage B — fracture.** A per-anatomy model emits four ABBC channels and nine
   learned same-instance affinity channels.
4. **Instance decoding.** Average-linkage agglomeration converts the affinity
   field into fragment instances and removes components below about 1 cm3.
5. **Assembly.** Instances are mapped to the official label ranges: sacrum
   1-50, left hip 51-100, right hip 101-150 and femur 151-200.

# 3. Deployed model contract

| Component | Deployed value |
|---|---|
| Stage A dataset | `Dataset539_PelvicFemurAnatomyV3` |
| Stage A trainer | `PengwinTrainerSTUNetBaseAnatomyV301` |
| Stage A fold/checkpoint | fold 0 / `checkpoint_best.pth` |
| Stage A initialization | from scratch on refreshed PENGWIN anatomy data |
| Router | v3.3 hybrid RF, confidence margin 0.15 |
| Stage B dataset | `Dataset538_PelvicFemurBICMFragmentV5` |
| Stage B trainer | `PengwinTrainerSTUNetBaseAffinityV308DeployedVal` |
| Stage B fold/checkpoint | fold 0 / `checkpoint_best.pth` |
| Stage B initialization | official TotalSegmentator `base_ep4k`, then refreshed-data fine-tuning |
| Stage B output | 13 channels: 4 ABBC + 9 affinity |
| Decoder | `decode_affinity_agglo`, `T=0.75` |

Only Stage B is TotalSegmentator-initialized. Stage A is the refreshed-data
scratch model. The Stage-B training process reached epoch 180 and then ended at
the start of epoch 181 without producing `checkpoint_final.pth`; the valid
`checkpoint_best.pth` selected during training is therefore the packaged
artifact.

# 4. Model archive layout

Grand Challenge extracts the uploaded archive directly under `/opt/ml/model`.
The tarball is created with `tar -C model_payload -czf model.tar.gz .`, producing:

```text
/opt/ml/model/
├── nnunet/results/
│   ├── Dataset539_PelvicFemurAnatomyV3/
│   │   └── PengwinTrainerSTUNetBaseAnatomyV301__nnUNetResEncUNetLPlans__3d_fullres/
│   └── Dataset538_PelvicFemurBICMFragmentV5/
│       └── PengwinTrainerSTUNetBaseAffinityV308DeployedVal__nnUNetResEncUNetLPlans__3d_fullres/
└── stage1_router/stage1_target_router_fold0.joblib
```

The router artifact is serialized natively with scikit-learn 1.6.1, matching the
container dependency. Its predictions were checked against the original
scikit-learn 1.7.2 serialization on 4,096 deterministic inputs.

# 5. Local evaluation

All rows below use the same 68-case official-aligned evaluation set.

| Experiment | Dice | HD95 (mm) | ASSD (mm) | Instance F1 | Split fragments |
|---|---:|---:|---:|---:|---:|
| Existing release | 0.884121 | 3.857464 | 1.039815 | **0.933582** | 457 |
| Refreshed-data scratch | 0.865772 | 4.009136 | 0.927063 | 0.846223 | 537 |
| v3.4 candidate, fixed `T=0.45` | **0.885355** | 3.422187 | 0.800181 | 0.921553 | 456 |
| v3.4 candidate, selected `T=0.75` | 0.882088 | **3.260951** | **0.774668** | 0.930066 | **426** |

The selected threshold trades 0.00327 Dice relative to `T=0.45` for better
HD95, ASSD, fragment over-splitting and instance F1. Relative to the existing
release, it improves HD95 by 0.5965 mm, ASSD by 0.2651 mm and removes 31 split
fragments, while Dice changes by -0.0020 and instance F1 by -0.0035. These are
local proxy results, not hidden-test leaderboard results.

# 6. Preprocessing and inference

Images are canonicalized to LPS and processed through the nnU-Net plans used in
training. Stage A predicts mutually exclusive anatomy masks. The selected
anatomy ROI is converted to bone-window CT plus routing-derived context channels
for Stage B. Inference runs the two models serially on one GPU, restores the
prediction to the input geometry and writes one
`/output/images/pelvic-fracture-segmentation/*.mha` label map.

# 7. Reproducibility and limitations

- Fold 0 is used for both stages; no ensemble is deployed.
- `T=0.75` was selected on the reported validation data, so an independent
  Grand Challenge container/hidden-test check is required before activation.
- Stage B has no `checkpoint_final.pth` because the run ended after epoch 180;
  the packaged checkpoint is the valid `checkpoint_best.pth`.
- The official challenge evaluator and hidden test distribution are not local,
  so the reported metrics are comparative proxies.
- Exact artifact hashes and source paths are recorded in the release
  `MODEL_MANIFEST.json`.

# 8. Runtime and licensing

The Docker image targets an NVIDIA T4 16 GiB GPU. Stage A and Stage B are loaded
serially to limit peak memory. STU-Net-B and nnU-Net are used under their
respective open-source licenses; the Stage-B initialization comes from the
official TotalSegmentator `base_ep4k` checkpoint.

# 9. References

- Isensee, F. et al. nnU-Net, *Nature Methods* 18, 203-211 (2021).
- STU-Net / TotalSegmentator bone pretraining.
- PENGWIN 2026 Task 1 baseline:
  `github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline`.

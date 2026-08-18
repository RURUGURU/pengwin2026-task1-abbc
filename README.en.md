# PENGWIN 2026 Task 1 — final preserved source

This is a two-stage nnU-Net/STU-Net pipeline for anatomy segmentation followed by fracture-fragment
instance segmentation. The worktree is a post-competition archive cleanup based on the immutable
release `v3.12@eac5e1f`; no remote push was performed.

The runtime uses V301 anatomy prediction, an RF target-family router, V308 Sacrum/Hip/Femur experts
with 13 outputs (4 ABBC + 9 affinity), and average-linkage decoding at `T=0.75`. Femur is decoded a
second time at `T=0.15` only when the primary result is empty or one very large instance, and the
alternative is accepted only if it increases the number of instances. Task 1 click injection is off.

The preserved model archive is `../model_bundles/v3_5_final_payload/model.tar.gz`, SHA-256
`049c38ea4abf1629a4d5f79a68a27918fd4103941fbf4f500b76211e93192919`. The Grand Challenge API
records a `harp3133t v3.5` Final submission row at 16/43 and MP 17.1, but it does not provide an
exact source commit or model-byte attestation. Therefore this source and archive are not claimed to
be byte-identical to that execution, and the displayed rank is not a deduplicated official team rank.

- `inference/`: Grand Challenge entrypoint and runtime decoder
- `code_task1/`: trainer discovery and segmentation implementation
- `Dockerfile`: non-root container and final local runtime configuration
- `requirements.txt`: container dependency pins

Historical release notes, visualization-only files, and portal material were removed from this
runtime repository. The parent submission folder preserves current portal text, evaluations, and
provenance. Container rebuild and GPU end-to-end inference were not rerun during archive cleanup.

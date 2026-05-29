#!/bin/bash
# Package the V0 model payload into model.tar.gz for Grand Challenge upload.
#
# Trailing-dot convention (per GC docs): the tarball contents must land
# directly under /opt/ml/model/ at runtime, with NO `model_payload/` prefix.
#
# Tarball layout (relative paths inside the archive, after extract):
#   nnunet/results/Dataset537_.../<trainer>/fold_0/checkpoint_best.pth
#   nnunet/results/Dataset537_.../<trainer>/fold_0/debug.json
#   nnunet/results/Dataset537_.../<trainer>/plans.json
#   nnunet/results/Dataset537_.../<trainer>/dataset.json
#   nnunet/results/Dataset537_.../<trainer>/dataset_fingerprint.json
#   nnunet/preprocessed/Dataset537_.../nnUNetResEncUNetLPlans.json
#   nnunet/preprocessed/Dataset537_.../dataset.json
#   nnunet/preprocessed/Dataset537_.../dataset_fingerprint.json
set -euo pipefail
cd /workspace/submission/v0
tar -C model_payload -czf model.tar.gz .
ls -lh model.tar.gz
# quick integrity check: top-level entries must be nnunet/ etc., NOT model_payload/.
echo
echo "Top entries inside model.tar.gz:"
tar -tzf model.tar.gz | head -20

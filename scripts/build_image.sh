#!/bin/bash
# Build the preserved PENGWIN 2026 Task 1 archive image.
#
# Build context is the repository root (one level up from scripts/) so the
# Dockerfile can COPY both inference/ and code_task1/.
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-pengwin-task1-v3.12-archive:latest}"

cd "$(dirname "$0")/.."
docker build \
    -t "$IMAGE_TAG" \
    .

echo
echo "Built image: $IMAGE_TAG"
docker images "$IMAGE_TAG" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedAt}}"

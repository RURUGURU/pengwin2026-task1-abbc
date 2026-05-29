# PENGWIN 2026 Task 1 -- V0 pipeline-test container.
#
# Layout:
#   /opt/app/inference/inference.py   -> entrypoint
#   /opt/app/code_task1/              -> our internal eval / decoder helpers
#   /opt/ml/model/                    -> model.tar.gz contents at runtime
#                                        (extracted by Grand Challenge)
#
# nnUNet_results / nnUNet_preprocessed / nnUNet_raw are pointed at the
# extracted tarball tree (see ENV below).

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs that SimpleITK / scikit-image need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

# --- Python deps -----------------------------------------------------------
# Build context must be /workspace so both `submission/v0/...` and
# `code_task1/` are reachable. See scripts/build_image.sh.
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /opt/app/requirements.txt

# --- App code --------------------------------------------------------------
# Entrypoint is self-contained: the ABBC decoder + sliding-window export are
# inlined into inference.py (lifted from code_task1/eval.py for V0). We still
# ship code_task1/ for offline debugging / regression alignment.
COPY inference /opt/app/inference
COPY code_task1 /opt/app/code_task1

# --- Runtime environment ---------------------------------------------------
# The Grand Challenge platform extracts model.tar.gz to /opt/ml/model/ at
# runtime. Our tarball is packed with trailing-dot convention (per GC docs:
#   tar -czvf model.tar.gz -C model_payload .)
# so contents land directly under /opt/ml/model/ (no prefix subdir).
ENV PENGWIN_ROOT=/opt/ml/model \
    nnUNet_results=/opt/ml/model/nnunet/results \
    nnUNet_preprocessed=/opt/ml/model/nnunet/preprocessed \
    nnUNet_raw=/opt/ml/model/nnunet/raw \
    PYTHONPATH=/opt/app:/opt/app/code_task1

# Grand Challenge runs the container with --network none, no extra args.
ENTRYPOINT ["python", "/opt/app/inference/inference.py"]

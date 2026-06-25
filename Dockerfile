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
# ship code_task1/ for offline debugging / regression alignment AND for the
# trainer-discovery shim to import from.
COPY inference /opt/app/inference
COPY code_task1 /opt/app/code_task1

# --- nnUNet trainer-discovery shim -----------------------------------------
# nnUNet v2 discovers trainer classes by walking
# `nnunetv2/training/nnUNetTrainer/`. Our PengwinTrainer*ABBCV291 lives in
# /opt/app/code_task1/core.py which is OUTSIDE that walk -> we copy a tiny
# shim into the nnUNet dir that re-exports our trainer classes by name. The
# site-packages path is base-image dependent, so we discover it at build
# time. This MUST run before USER drop because site-packages is root-only.
RUN NN_TR_DIR="$(python -c 'import nnunetv2.training.nnUNetTrainer as m; print(m.__path__[0])')" \
    && cp /opt/app/inference/pengwin_trainers_shim.py "$NN_TR_DIR/pengwin_trainers.py" \
    && echo "[pengwin_v0] trainer shim installed at $NN_TR_DIR/pengwin_trainers.py" \
    && python -c "import nnunetv2.training.nnUNetTrainer.pengwin_trainers as m; print('[pengwin_v0] shim re-exports', m.__pengwin_trainer_count__, 'PengwinTrainer classes')"

# --- Runtime environment ---------------------------------------------------
# The Grand Challenge platform extracts model.tar.gz to /opt/ml/model/ at
# runtime. Our tarball is packed with trailing-dot convention (per GC docs:
#   tar -czvf model.tar.gz -C model_payload .)
# so contents land directly under /opt/ml/model/ (no prefix subdir).
#
# MPLCONFIGDIR / HOME fixes: the non-root `user` has no writable /home/user
# dir, so matplotlib's default cache path raises PermissionError on import.
# Point HOME and matplotlib cache at /tmp (the platform allows /tmp writes).
ENV PENGWIN_ROOT=/opt/ml/model \
    nnUNet_results=/opt/ml/model/nnunet/results \
    nnUNet_preprocessed=/opt/ml/model/nnunet/preprocessed \
    nnUNet_raw=/opt/ml/model/nnunet/raw \
    PYTHONPATH=/opt/app:/opt/app/code_task1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

# [v1.6 = V308 affinity DECODE-FUSION] Stage-B = V308 13ch head, but decoded by FUSION: take the
# V302 core-seed-watershed partition as a precise base, then sub-split a base instance ONLY where the
# learned affinity shows a COHERENT high-separation surface (real-fracture gate, ridge>=3000 vox) ->
# recovers V302's touching-fragment merges (recall 0.707->0.736, hd95 3.51->2.91, topology +0.02)
# while keeping V308-solo's over-split/phantom precision damage to the minimum of any variant
# (precision 0.928->0.868 vs V308's 0.837). dev proxy-F1 ~ V302 (-0.008) but improves exactly V302's
# weak GC ranks (recall/hd95/merge/topology/assd) -> the GC Mean-Position candidate. Same V308 weight
# tar as v1.5. Roll back to V308-solo: set AFFINITY_DECODE=1 + AGGLO_T=0.45 (remove FUSION_*). Roll
# back to V302: rebuild from tag v1.4.3 (or comment these lines).
ENV PENGWIN_DS538_TRAINER=PengwinTrainerSTUNetBaseAffinityV308 \
    PENGWIN_DS538_OUT_CH=13 \
    PENGWIN_FUSION_DECODE=1 \
    PENGWIN_FUSION_T=0.45 \
    PENGWIN_FUSION_RIDGE_VOX=3000

# [v1.7 = + Stage-A recall fix] The GC recall killer (Instance Recall 0.574 = pos 56) is whole-anatomy
# MISS: Ds539 gives a PRESENT hip ~0 voxels (or swaps L<->R) -> the 0.20x routing gate drops it ->
# the anatomy emits ZERO -> all its fragments become FN (and the swapped side a FP). Fix: reconcile the
# Ds539-size routing with the GEOMETRIC bone-skeleton decomposition -- (A) add any pelvic anatomy bone-
# skeleton found that routing dropped (uses the bone mask via the existing sanity fallback); (B) drop a
# routed Ds539 hip that bone-skeleton did NOT find but whose mask sits on the OTHER hip's bone (= L<->R
# swap). Verified on the 5 GC prelim-case logs: recovers the dropped/swapped anatomies, inert on healthy
# cases + on all of dev (no drops there). Default ON; PENGWIN_STAGEA_BONE_RECONCILE=0 to disable.
ENV PENGWIN_STAGEA_BONE_RECONCILE=1

# Grand Challenge security policy: container must not run as root.
# Create a service user with no shell, no password, no home write permissions
# other than the default. /opt/app is owned by root but world-readable from
# the COPY layers above, so the user only needs execute access to read code.
RUN groupadd -r user && useradd --no-log-init -r -g user user

USER user:user

# Grand Challenge runs the container with --network none, no extra args.
ENTRYPOINT ["python", "/opt/app/inference/inference.py"]

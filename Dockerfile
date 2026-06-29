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

# [v1.8 = ROLLBACK to v1.5 behaviour] v1.6 fusion + v1.7 Stage-A reconcile BOTH REGRESSED on GC
# (5 prelim cases: recall 0.574->0.522, f1 0.572->0.533, dice 0.919->0.872, hd95 5.31->10.79, merge
# 0.15->0.35). Root cause: (1) fusion only sub-splits a V302 core-seed base and CANNOT merge an
# over-split base, so on GC's OOD/noisy ROIs the base over-splits and fusion makes it WORSE (RightHip
# 5->8 in the prelim logs); V308-solo average-linkage CAN merge weak ridges, so it is more robust.
# (2) the bone-skeleton reconcile added tiny sliver anatomies (bone mask ~0.1% of volume) that decoded
# to NOTHING or to FP. The dev proxy could see neither (in-distribution cores -> fusion base did not
# over-split; zero whole-anatomy drops on dev -> reconcile untestable). So revert to v1.5 = V308
# affinity-solo decode = the best GC result so far. The fusion/reconcile code stays in inference.py but
# is ENV-DISABLED here. NOTE PENGWIN_STAGEA_BONE_RECONCILE defaults to 1 in code, so it MUST be set to 0
# explicitly. Same V308 weight tar. Equivalent to rebuilding from tag v1.5.
#
# [v1.9 = FULL-DATA Stage-B] Fracture weights upgraded V308 fold_0 (272 cases) -> V308 fold_all (all 340,
# SAME 200ep transfer/base_ep4k recipe, ~18h). Uses model_v1_6.tar.gz (Stage-A V301 fold_0 routing
# UNCHANGED + V308 fold_all + V302 fold_0 rollback). The ONLY change vs v1.5/v1.8 is +68 training cases
# for the fracture net. MUST set PENGWIN_DS538_FOLD=all so inference loads the fold_all checkpoint.
# Decode unchanged (AGGLO_T=0.45; full 68-case sweep is T-insensitive). Rollback: V302 -> DS538_TRAINER
# =...V302 + DS538_FOLD=0; exact v1.5 (V308 fold_0) -> re-deploy model_v1_5.tar.gz. Dev-68 ins_f1 leaky
# 0.7626 / honest-fold_0 0.7354; real GC is the judge.
ENV PENGWIN_DS538_TRAINER=PengwinTrainerSTUNetBaseAffinityV308 \
    PENGWIN_DS538_FOLD=all \
    PENGWIN_DS538_OUT_CH=13 \
    PENGWIN_AFFINITY_DECODE=1 \
    PENGWIN_AGGLO_T=0.45 \
    PENGWIN_FUSION_DECODE=0 \
    PENGWIN_STAGEA_BONE_RECONCILE=0

# Grand Challenge security policy: container must not run as root.
# Create a service user with no shell, no password, no home write permissions
# other than the default. /opt/app is owned by root but world-readable from
# the COPY layers above, so the user only needs execute access to read code.
RUN groupadd -r user && useradd --no-log-init -r -g user user

USER user:user

# Grand Challenge runs the container with --network none, no extra args.
ENTRYPOINT ["python", "/opt/app/inference/inference.py"]

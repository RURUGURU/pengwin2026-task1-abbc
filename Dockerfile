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

# 빌드 컨텍스트가 호스트의 빡빡한 권한(예: 0600)을 그대로 보존할 수 있다. GC 이미지는 비root
# 서비스 사용자로 돌아가므로, 체크아웃 umask 에 의존하지 말고 복사된 소스 권한을 정규화한다.
RUN chmod -R a+rX /opt/app/inference /opt/app/code_task1

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
    XDG_CACHE_HOME=/tmp/.cache \
    MALLOC_MMAP_THRESHOLD_=131072

# MALLOC_MMAP_THRESHOLD_ (above) is a MEMORY fix, and the only one this investigation found that is
# free. By default glibc raises its mmap threshold dynamically to whatever mmap'd block was freed
# last, up to 32 MB: after the first big numpy array is released, every subsequent allocation under
# 32 MB comes out of the brk heap and is NOT returned to the OS when freed. Pinning the threshold at
# 128 KB keeps those mid-sized blocks (torch parameter tensors, nnUNet's duplicate checkpoint
# state_dict, SimpleITK buffers) on mmap, so free() really is munmap() and the heap never accumulates
# them. Measured on case 285 (131.85 Mvox, the largest local volume), /usr/bin/time -v Maximum RSS,
# one inference at a time:
#     baseline                        8,343,784 KB / 8,314,352 KB   (noise floor 29,432 KB)
#     MALLOC_MMAP_THRESHOLD_=131072   8,114,364 KB                  = -214,704 KB = -0.205 GB
# 7x the noise floor, and it still pays on top of the predictor-lifetime fix (8,095,832 ->
# 7,905,452 KB = -0.182 GB), so the two are close to additive.
# Isolation: MALLOC_ARENA_MAX=2 alone measured 8,314,168 KB = EXACTLY baseline (0 effect), and
# MALLOC_TRIM_THRESHOLD_/MALLOC_TOP_PAD_ added only 25 MB on top, so the mmap threshold owns the
# whole win and is the only variable set here. In-process malloc_trim(0) at the same free points
# measured 0.010 GB (see the previous commit) -- this env var is not the same lever, it prevents the
# retention rather than trying to undo it.
# Result-identical: case 285 -> labels [151, 152], 2 fragments, volumes 2,706,885 / 429,957 (both).
# No throughput cost observed: that run was the FASTEST of the whole series (1:31 vs 1:40 / 1:53
# baseline), though wall clock here is noisy because other jobs shared the host.

# [v2.0 = v1.5 weights + target-family router]
# Stage-A V301 fold_0 and Stage-B V308 fold_0 stay unchanged. A lightweight
# random-forest router packaged in model.tar.gz chooses pelvic vs femur and
# prevents non-target anatomies from reaching Stage-B. The router artifact must
# be present at /opt/ml/model/stage1_router/stage1_target_router_fold0.joblib.
# PENGWIN_TARGET_ROUTER=1 fails fast if the artifact is missing, which avoids a
# silent fallback to the older Ds539 volume-ratio route.
# [v3.11 = v3.9 메모리 경로 + 팀원 v3.5 의 always-expert 런타임 설정]
#
# 이 ENV 는 GC Final Test 에서 **MP 13.6 (팀 3위)** 를 실제로 낸 구성이다
# (`submission_task1/teammate_v35_package/MODEL_MANIFEST.json` 의 `runtime` 블록과 동일).
# 우리 v3.6 배포(PENGWIN_AGGLO_T=0.45 / femur 0.15, unified Stage-B)는 같은 보드에서 16.4 였다.
#
# 바뀌는 것은 **가중치 선택과 T 뿐**이고 코드 경로는 v3.9 그대로다:
#   · Stage-B 를 부위별 전문가로 (Sacrum / 좌우Hip 공용 / Femur, 전부 v3.4 TotalSegmentator-base
#     V308 에서 encoder 를 얼리고 decoder+헤드만 1~3에폭 튜닝한 것)
#   · AGGLO_T 0.45 → 0.75, femur override 제거 (팀원은 부위 구분 없이 0.75 하나)
#   · RF family 라우터 유지 + confidence margin 0.15
#
# ⚠️ model.tar.gz 는 반드시 팀원 번들이어야 한다 —
#    sha256 049c38ea4abf1629a4d5f79a68a27918fd4103941fbf4f500b76211e93192919 (1,409,476,486 B).
#    우리 v3_0 번들에는 expert 체크포인트가 없어서 로드에 실패한다.
# ⚠️ 아래 fallback(`...V308DeployedVal`)은 4개 부위가 전부 전문가로 매핑되므로 도달하지 않는다.
#    도달하면 그 이름의 체크포인트가 번들에 없어 실패하므로, 로그에서 부위별 트레이너 이름을 확인할 것.
ENV PENGWIN_DS539_TRAINER=PengwinTrainerSTUNetBaseAnatomyV301 \
    PENGWIN_DS539_FOLD=0 \
    PENGWIN_DS538_TRAINER=PengwinTrainerSTUNetBaseAffinityV308DeployedVal \
    PENGWIN_DS538_TRAINER_SACRUM=PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal \
    PENGWIN_DS538_TRAINER_HIP=PengwinTrainerSTUNetBaseAffinityV308HipExpertDeployedVal \
    PENGWIN_DS538_TRAINER_FEMUR=PengwinTrainerSTUNetBaseAffinityV308FemurExpertDeployedVal \
    PENGWIN_DS538_FOLD=0 \
    PENGWIN_DS538_OUT_CH=13 \
    PENGWIN_AFFINITY_DECODE=1 \
    PENGWIN_AGGLO_T=0.75 \
    PENGWIN_FEMUR_ADAPTIVE_T=0.15 \
    PENGWIN_FEMUR_ADAPTIVE_MINVOX=150000 \
    PENGWIN_FUSION_DECODE=0 \
    PENGWIN_STAGEA_BONE_RECONCILE=0 \
    PENGWIN_ROUTE_CC_MODE=largest \
    PENGWIN_CLICK_INJECT=0 \
    PENGWIN_TARGET_ROUTER=1 \
    PENGWIN_RF_CONF_MARGIN=0.15 \
    PENGWIN_TARGET_ROUTER_PATH=/opt/ml/model/stage1_router/stage1_target_router_fold0.joblib

# Grand Challenge security policy: container must not run as root.
# Create a service user with no shell, no password, no home write permissions
# other than the default. /opt/app is owned by root but world-readable from
# the COPY layers above, so the user only needs execute access to read code.
RUN groupadd -r user && useradd --no-log-init -r -g user user

USER user:user

# Grand Challenge runs the container with --network none, no extra args.
ENTRYPOINT ["python", "/opt/app/inference/inference.py"]

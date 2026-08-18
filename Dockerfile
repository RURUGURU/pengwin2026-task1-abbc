# Task 1 최종 로컬 재현 기준. 원본 v3.12 runtime 설정을 유지하고 설명만 정리했다.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip && pip install -r /opt/app/requirements.txt

COPY inference /opt/app/inference
COPY code_task1 /opt/app/code_task1
RUN chmod -R a+rX /opt/app/inference /opt/app/code_task1

# nnU-Net은 site-packages의 trainer tree만 탐색하므로 final trainer re-export shim을 설치한다.
RUN NN_TR_DIR="$(python -c 'import nnunetv2.training.nnUNetTrainer as m; print(m.__path__[0])')" \
    && cp /opt/app/inference/pengwin_trainers_shim.py "$NN_TR_DIR/pengwin_trainers.py" \
    && python -c "import nnunetv2.training.nnUNetTrainer.pengwin_trainers as m; print(m.__pengwin_trainer_count__)"

# Grand Challenge가 model.tar.gz를 /opt/ml/model에 해제하는 계약이다. 비root cache와
# 중간 tensor의 glibc heap 잔류를 피하기 위한 mmap threshold도 원본 release 값을 유지한다.
ENV PENGWIN_ROOT=/opt/ml/model \
    nnUNet_results=/opt/ml/model/nnunet/results \
    nnUNet_preprocessed=/opt/ml/model/nnunet/preprocessed \
    nnUNet_raw=/opt/ml/model/nnunet/raw \
    PYTHONPATH=/opt/app:/opt/app/code_task1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    MALLOC_MMAP_THRESHOLD_=131072

# V301 anatomy + V308 부위별 expert + 13채널 affinity decoder.
# 기본 T=0.75이며 Femur 결과가 비거나 큰 단일 조각일 때만 T=0.15 대안을 비교한다.
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

RUN groupadd -r user && useradd --no-log-init -r -g user user
USER user:user
ENTRYPOINT ["python", "/opt/app/inference/inference.py"]

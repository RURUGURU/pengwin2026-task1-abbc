#!/bin/bash
# v3.14 model.tar.gz 를 만든다 — Stage-B 전문가 3종만 우리 V5 모델로 갈아끼운다.
#
# 무엇이 들어가는가
#   · Ds539 해부 라우터 · stage1 RF 라우터 · plans/dataset json  → **팀원 번들 것 그대로**
#   · Ds538 조각 전문가 3종                                      → **우리 학습본**
#       Sacrum : PengwinTrainerSTUNetBaseV5SacrumW   (V5 손실 + sacrum 오버샘플 2.0, 160ep)
#       Hip    : PengwinTrainerSTUNetBaseV5HipW      (V5 손실 + hip 오버샘플 2.0, 160ep)
#       Femur  : PengwinTrainerSTUNetBaseV5FemurFT   (위 160ep 뒤 femur 추출 91%·LR 1e-4 로 18ep 파인튜닝)
#
# 가중치 선택 기준 (중요)
#   각 전문가는 **담당 부위 instance-F1** 로 고른 `checkpoint_best_anat.pth` 를 쓴다.
#   nnU-Net 의 `checkpoint_best` 는 혼합 검증 EMA(pelvic 72.7%)로 고르는데, 부위 전문가를
#   그 기준으로 뽑으면 엉뚱한 에폭이 나온다 — 실측으로 Femur 29에폭·Sacrum 71에폭 어긋났다.
#   번들에는 그렇게 고른 가중치를 컨테이너가 찾는 이름(`checkpoint_best.pth`)으로 넣는다.
#
# 크기
#   옵티마이저/스케줄러 상태를 벗겨 전문가당 444MB → 222MB. 번들 합계 약 1.1GB
#   (팀원 번들 1.5GB). 추론에 쓰는 키만 남긴다.
set -euo pipefail
ROOT=${PENGWIN_ROOT:-/home/ljg/Project/PENGWIN2026}
SRC_TEAM=$ROOT/submission_task1/teammate_v35_package/model_payload
SRC_OURS=$ROOT/code_task1/result/experiments/v4/nnunet/results/Dataset538_PelvicFemurBICMFragmentV5
OUT=${1:?사용: build_v314_payload.sh <출력디렉터리> [tar 경로]}
TAR=${2:-}
PY=${PENGWIN_PY:-/opt/miniconda3/envs/pengwin_v2/bin/python}
PLANS=nnUNetResEncUNetLPlans__3d_fullres

[ -d "$SRC_TEAM" ] || { echo "🔴 팀원 번들 없음: $SRC_TEAM"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"

# 1) 팀원 번들에서 **옛 Ds538 전문가만 빼고** 전부 복사 (Ds539 · 라우터 · plans)
rsync -a --exclude 'Dataset538_PelvicFemurBICMFragmentV5/PengwinTrainerSTUNetBaseAffinityV308*ExpertDeployedVal*' \
      "$SRC_TEAM/" "$OUT/"

# 2) 우리 전문가 3종을 넣는다 (옵티마이저 상태 제거)
PENGWIN_SRC="$SRC_OURS" PENGWIN_OUT="$OUT" PYTHONPATH=$ROOT/code_task1 "$PY" - <<'PY'
import os, shutil, sys, json, torch
sys.path.insert(0, os.path.join(os.environ["PYTHONPATH"]))
from core.base import _register_numpy_safe_globals
SRC, OUT = os.environ["PENGWIN_SRC"], os.environ["PENGWIN_OUT"]
DST = f"{OUT}/nnunet/results/Dataset538_PelvicFemurBICMFragmentV5"
P = "nnUNetResEncUNetLPlans__3d_fullres"
KEEP = ("network_weights", "init_args", "trainer_name",
        "inference_allowed_mirroring_axes", "current_epoch")
for t in ("PengwinTrainerSTUNetBaseV5SacrumW",
          "PengwinTrainerSTUNetBaseV5HipW",
          "PengwinTrainerSTUNetBaseV5FemurFT"):
    s, d = f"{SRC}/{t}__{P}", f"{DST}/{t}__{P}"
    ck = f"{s}/fold_0/checkpoint_best_anat.pth"
    if not os.path.exists(ck):
        raise SystemExit(f"🔴 부위별 best 가 없다: {ck}\n"
                         f"   혼합 EMA 기준 checkpoint_best 로 대체하면 안 된다 — §7-50 참조.")
    os.makedirs(f"{d}/fold_0", exist_ok=True)
    for j in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        shutil.copy2(f"{s}/{j}", f"{d}/{j}")
    with _register_numpy_safe_globals():
        c = torch.load(ck, map_location="cpu", weights_only=False)
    torch.save({k: c[k] for k in KEEP if k in c}, f"{d}/fold_0/checkpoint_best.pth")
    bj = json.load(open(f"{s}/fold_0/anat_best.json"))
    print(f"  {t:34s} ep{c.get('current_epoch')} · 부위 F1 {bj['best_f1']:.4f} ({bj['anatomy']})")
PY

# 3) 로드 검증 — 키를 벗겨서 못 읽게 되면 컨테이너에서 all-zero 가 나온다(§7-38)
PENGWIN_OUT="$OUT" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$PY" - <<'PY' 2>/dev/null
import os, torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
base = f"{os.environ['PENGWIN_OUT']}/nnunet/results/Dataset538_PelvicFemurBICMFragmentV5"
P = "nnUNetResEncUNetLPlans__3d_fullres"
for t in ("PengwinTrainerSTUNetBaseV5SacrumW","PengwinTrainerSTUNetBaseV5HipW","PengwinTrainerSTUNetBaseV5FemurFT"):
    p = nnUNetPredictor(device=torch.device("cpu"), verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(f"{base}/{t}__{P}", use_folds=(0,), checkpoint_name="checkpoint_best.pth")
    with torch.no_grad():
        y = p.network(torch.zeros(1, 1, 64, 64, 64))
    y0 = y[0] if isinstance(y, (list, tuple)) else y
    assert y0.shape[1] == 14, f"{t}: 출력채널 {y0.shape[1]} != 14 (Dockerfile 의 OUT_CH/AFF_START 와 불일치)"
    print(f"  ✔ {t:34s} 로드·순전파 OK · 14채널")
PY


# 3.5) 🔴 자족성 검사 — 심링크가 있으면 GC 가 tar 추출을 거부한다
#      (2026-08-14: 게이트가 번들 안에 /home/ljg/... 절대 심링크를 만들어 제출이 실패했다)
_ln=$(find "$OUT" -type l | wc -l)
if [ "$_ln" -ne 0 ]; then
  echo "🔴 번들에 심링크 $_ln 개가 있다 — GC 는 이걸 못 푼다:"
  find "$OUT" -type l -printf "   %p -> %l\n"
  exit 1
fi
echo "  ✔ 심링크 0개 (자족적)"

echo "번들 크기: $(du -sh "$OUT" | cut -f1)"
if [ -n "$TAR" ]; then
  tar -czf "$TAR" -C "$OUT" .
  echo "tar: $TAR ($(stat -c%s "$TAR") B)"
  echo "sha256: $(sha256sum "$TAR" | cut -d' ' -f1)"
fi

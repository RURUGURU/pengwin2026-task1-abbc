# v3.11 — always-expert Stage-B + T=0.75 (팀원 v3.5 런타임을 우리 코드 위에)

## 왜 이 릴리스인가

GC Task 1 Final Test 보드 실측(2026-08-08):

| 계정 | 구성 | Mean Position | 팀 순위 |
|---|---|---:|---|
| `harp3133t` | always-expert + `AGGLO_T=0.75` | **13.6** | 3위 |
| `ruruguru` (v3.6) | unified Stage-B + `0.45/femur 0.15` | 16.4 | 4위 |

같은 팀인데 **decode 설정과 Stage-B 가중치 선택만으로 2.8 MP** 차이가 난다.
차이의 정체는 split 과 topology 다:

| | ruruguru v3.6 | harp3133t |
|---|---|---|
| Split Errors | 0.150 (28위) | **0.063 (4위)** |
| Topology | 0.746 (14위) | **0.819 (보드 1위)** |
| Merge Errors | 0.308 | 0.304 (거의 동일) |

⇒ **merge 비용 없이** split 을 낮춘다. 이 릴리스는 그 구성을 `ruruguru` 계정으로 올릴 수 있게
우리 코드 베이스 위에 재현한 것이다.

## 무엇이 v3.9 와 다른가

1. **`code_task1/core.py`** — Stage-B 전문가 로딩용 **별칭 클래스 4개** 추가.
   정본(`code_task1/core/stage2_fracture.py:417-433`)에는 있었으나 **배포 미러에는 빠져 있었다.**
   없으면 `pengwin_trainers_shim` 이 이름을 re-export 하지 못해 nnU-Net trainer discovery 가
   실패하고 컨테이너가 로드 단계에서 죽는다.
2. **`inference/inference.py`** — `DS538_EXPERT_TRAINERS` 배선(부위별 Stage-B 지연 로딩,
   한 번에 하나만 상주). env 가 비면 v3.9 와 바이트 동일하다.
3. **`Dockerfile`** — 런타임 ENV 를 팀원 v3.5 검증 구성으로. `AGGLO_T` 0.45→**0.75**,
   femur override 제거, 부위별 expert 트레이너 지정, `RF_CONF_MARGIN=0.15`.
   빌드 컨텍스트 권한 정규화(`chmod -R a+rX`)도 포함 — GC 는 비root 로 돌린다.

**v3.6~v3.9 의 메모리 작업은 전부 유지된다** (peak RSS 10.16 → 6.41 GB).
v3.5 가 Final Test 에서 OOM 으로 죽은 전례가 있어 이 부분은 양보하지 않는다.

## 🔴 업로드 시 반드시 지킬 것

### 1. model.tar.gz 는 팀원 번들이어야 한다

```
파일    submission_task1/teammate_v35_package/model.tar.gz
크기    1,409,476,486 B
sha256  049c38ea4abf1629a4d5f79a68a27918fd4103941fbf4f500b76211e93192919
```

우리 `model_bundles/v3_0/model.tar.gz` 에는 **expert 체크포인트가 없어서 로드에 실패한다.**
번들 안에 있어야 하는 것:

```
nnunet/results/Dataset539_PelvicFemurAnatomyV3/PengwinTrainerSTUNetBaseAnatomyV301__.../fold_0/checkpoint_best.pth
nnunet/results/Dataset538_PelvicFemurBICMFragmentV5/PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal__.../fold_0/checkpoint_best.pth
                                                    ...HipExpertDeployedVal__.../fold_0/checkpoint_best.pth
                                                    ...FemurExpertDeployedVal__.../fold_0/checkpoint_best.pth
stage1_router/stage1_target_router_fold0.joblib
```

### 2. 빌드 후 로그에서 확인할 것

- Stage-A 체크섬이 랜덤이 아닐 것 — `w0sum=1.0718e+02`
- 라우터 로드 — `n_features=37`
- **부위별 Stage-B 트레이너 이름이 로그에 찍힐 것** (`[Sacrum] Stage-B trainer -> ...SacrumExpert...`)
  이게 안 보이면 expert 가 안 걸린 것이고, fallback(`...V308DeployedVal`)은 번들에 체크포인트가
  없어서 실패한다.

### 3. "마지막 run 만 채점된다"

Final Test 는 마지막 실행이 성적이 된다. **이것보다 나쁜 것을 나중에 올리면 그게 최종 성적이다.**

## 검증 상태

- `pengwin_trainers_shim` 이 4개 이름 전부 re-export 확인 (실제 import, `PengwinTrainer*` 8개)
- `ANATOMY_RANGES` 의 부위 이름과 `DS538_EXPERT_TRAINERS` 키가 정확히 일치
- `code_task1/core.py` 문법 검증 통과
- ⚠️ **로컬 컨테이너 빌드/스모크는 미실행** — GC 빌드 후 위 로그 3종으로 확인할 것

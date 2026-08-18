# PENGWIN 2026 Task 1 — 최종 보존 소스

골반·대퇴 CT에서 Sacrum, Left/Right Hip, Femur를 찾고 각 골절 조각을 instance ID로 분리하는
2-stage nnU-Net/STU-Net 파이프라인이다. 이 작업트리는 원본 release `v3.12@eac5e1f`를 기반으로
대회 종료 뒤 문서와 미사용 파일만 정리한 로컬 archive branch다. 외부 저장소에는 push하지 않았다.

## 현재 실행 계약

1. Stage A `PengwinTrainerSTUNetBaseAnatomyV301`이 5-class anatomy를 예측한다.
2. RF target-family router가 pelvic/femur 계열을 선택한다.
3. Stage B가 V308 Sacrum/Hip/Femur expert 중 하나를 13채널(ABBC 4 + affinity 9)로 실행한다.
4. average-linkage affinity decoder는 기본 `T=0.75`를 사용한다. Femur 결과가 0개이거나 큰 단일
   조각으로 남을 때만 `T=0.15` 재디코드를 계산하고 instance 수가 실제로 늘어난 경우에만 채택한다.
5. Task 1에서는 click 입력을 사용하지 않는다.

최종 payload 정본은 상위 폴더의
`../model_bundles/v3_5_final_payload/model.tar.gz`이며 SHA-256은
`049c38ea4abf1629a4d5f79a68a27918fd4103941fbf4f500b76211e93192919`이다.
Grand Challenge snapshot의 `harp3133t v3.5` Final 행은 16/43, MP 17.1이지만, API가 정확한 source
commit이나 model bytes를 제공하지 않으므로 이 소스·tar와 GC 실행을 byte-identical이라고 주장하지
않는다. 순위는 account/submission 행 순위이며 공식 중복 제거 team rank가 아니다.

## 구조

- `inference/`: Grand Challenge entrypoint, resampling, router, affinity decoder
- `code_task1/`: checkpoint class discovery와 공통 segmentation 구현
- `Dockerfile`: 최종 로컬 runtime 변수와 non-root 실행 계약
- `requirements.txt`: 컨테이너 dependency pin
- `scripts/build_image.sh`: 로컬 image build helper

과거 release note, 시각화 전용 파일, 포털 문서는 이 runtime 저장소에서 제거했다. 포털 문구와
평가·provenance는 상위 `submission_task1/portal/` 및 `submission_task1/reports/`에서 관리한다.

```bash
bash scripts/build_image.sh
```

이번 archive 정리에서는 container build와 GPU end-to-end inference를 다시 실행하지 않았다.

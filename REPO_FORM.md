# GitHub "Create a new repository" form -- values to paste

> Use this when filling the form at https://github.com/new for the V0
> submission. Bilingual KR / EN to match the original draft language.

## Form values (KR / EN)

| Field (KR) | Field (EN) | Value | Reasoning (KR / EN) |
|---|---|---|---|
| 소유자 | Owner | `<user fills>` (개인 계정 또는 org) | 본인 GitHub 계정 또는 팀 org. Grand Challenge "Link to GitHub" 시 이 owner/repo 조합으로 연결됨. / Your personal handle or team org. Grand Challenge "Link to GitHub" wires up to `OWNER/REPO`. |
| 저장소 이름 | Repository name | **`pengwin2026-task1-abbc`** | 챌린지 + 태스크 + 메서드를 한 줄에. 모두 소문자 + 하이픈 (GitHub 권장). / Challenge + task + method in one slug, lowercase + hyphen (GitHub-recommended). |
| 저장소 이름 (대안 1) | Repository name (alt 1) | `pengwin-task1-v0-abbc` | V0 버전 명시; 향후 V1/V2 분리 리포 운영 시 적합. / Calls out V0 explicitly; useful if you plan separate V1/V2 repos. |
| 저장소 이름 (대안 2) | Repository name (alt 2) | `pengwin2026-task1` | 메서드 표기 생략한 short slug; V0 -> V1... 진화를 한 리포로 유지할 때. / Method-agnostic short slug; keep one repo across V0 -> V1 evolution. |
| 설명 | Description | (320자 이하, 아래 항목 참고) | GitHub 검색/소셜 카드에 노출. AlgorithmRegistration.txt 의 한 줄 요약을 trimming해 사용. / Surfaces in GitHub search and social cards; trimmed one-liner from AlgorithmRegistration.txt. |
| 공개 범위 | Visibility | **Public** (권장 / recommended) | Grand Challenge "Link to GitHub" 는 public repo 가 가장 간단. private 도 PAT 설정하면 가능하지만 추가 절차. / Public is the simplest path for "Link to GitHub"; private works but needs a PAT. |
| README 추가 | Add a README file | **OFF** | 우리가 직접 작성한 `README.md` 를 푸시함. / We ship our own `README.md`. |
| .gitignore 추가 | Add .gitignore | **OFF** | 모델 가중치 / 데이터 차단을 위한 우리만의 `.gitignore` 를 푸시함. / We ship our own `.gitignore` that blocks weights and data. |
| 라이선스 추가 | Choose a license | **OFF** (None) | 우리가 MIT `LICENSE` 파일을 푸시함. / We ship an MIT `LICENSE`. |

## Description field (320 chars)

```
Per-Anatomy ABBC nnU-Net with Core-Seed Watershed for pelvic & femoral fracture-fragment instance segmentation in CT (PENGWIN 2026 Task 1). Anatomy-routed nnU-Net v2 ResEnc-L, 4-class boundary-weighted softmax, core-seed watershed decoder, ~1cm^3 CC prune. V0 pipeline-test; femur is a zero fallback.
```

(약 320자 / ~320 chars; GitHub Description 한도 350자 내에서 안전.)

## After clicking "Create repository"

GitHub 가 만들어내는 빈 리포로 이 트리를 push 하세요. 자세한 명령은
[`PUSH_COMMANDS.md`](PUSH_COMMANDS.md) 참고.

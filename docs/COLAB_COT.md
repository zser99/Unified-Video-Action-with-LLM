# Colab 실행 & CoT/LLM 오케스트레이션

이 문서는 원본 UVA 코드 위에 추가된 두 가지 변경을 정리합니다.

1. **Colab에서 LIBERO-10 평가가 돌아가도록 만든 호환성 작업** — Colab의 Python 3.12 / 최신 패키지 환경에서 `eval_sim.py` 평가 로직을 그대로 재현.
2. **추론 시점 CoT(Chain-of-Thought) / LLM 오케스트레이션** — 동결된(frozen) UVA 체크포인트 위에 플래너를 씌워 `language_goal`을 단계별로 재작성하는 래퍼.

UVA 가중치는 **전혀 학습/수정하지 않습니다.** 모든 추가 기능은 추론 시점에만 동작합니다.

---

## 1. 새로 추가된 파일

### 노트북 (`notebooks/`)
| 파일 | 용도 | GPU |
|------|------|-----|
| `colab_libero10_eval.ipynb` | 원본 UVA baseline을 Colab에서 LIBERO-10으로 평가 (smoke / light / full 3단계 예산) | 필요 |
| `colab_libero10_cot_eval.ipynb` | 동일 평가에 CoT/LLM 래퍼를 얹어 `baseline` vs `rule` vs `llm` 비교 | 필요 |
| `colab_option_a_cot.ipynb` | 시뮬레이터/GPU 없이 CoT 플래너 출력만 확인 | 불필요 |

### CoT 모듈 (`unified_video_action/cot/`)
| 파일 | 역할 |
|------|------|
| `planner.py` | `CoTPlan` 데이터클래스, 추상 `CoTPlanner`, `RuleBasedCoTPlanner`(API 불필요), `_format_language_goal`, `pick_candidate_index` 리랭커 |
| `llm_planner.py` | `LLMCoTPlanner` — OpenAI Chat Completions(이미지 있으면 vision) 호출. 키 없거나 에러 시 rule 플래너로 fallback |
| `obs_encoding.py` | 시뮬레이터 obs에서 RGB 프레임을 찾아 JPEG data URL로 인코딩, proprio 요약 |
| `factory.py` | `create_planner("rule"\|"llm", ...)` |
| `__init__.py` | 공개 심볼 export |

### 정책 래퍼
- `unified_video_action/policy/cot_orchestrated_policy.py` — `CoTOrchestratedPolicy`. 내부 UVA 정책을 감싸고 `replan_every` 스텝마다 플래너를 호출해 `language_goal`을 갱신한 뒤 내부 정책에 위임.

### 루트 스크립트
- `eval_sim_cot.py` — `eval_sim.py`와 동일하지만 CoT 래퍼 옵션 추가 (`--planner`, `--no_cot`, `--quick_test` 등).
- `plan_cot_only.py` — UVA/시뮬레이터 없이 CoT 플래너 출력만 JSON으로 덤프 (CLI).

---

## 2. Colab 호환성을 위한 소스 변경

Colab은 Python 3.12 + 최신 pip/패키지라 원본 환경(`conda_environment.yml`, Python 3.10)과 충돌합니다. 다음 변경으로 코드 수정 없이도 노트북에서 동작하도록 했습니다.

- **`unified_video_action/model/common/lr_scheduler.py`**
  - 신버전 `diffusers`에서 사라진 `from diffusers.optimization import Union, ...` 를 표준 `typing` import로 교체하고, `diffusers` import 실패 시 `transformers.optimization` + `torch.optim.Optimizer`로 fallback 하는 `try/except` 추가.
- **`unified_video_action/env_runner/libero_image_runner.py`**
  - 파일 최상단에 **`mujoco_py` shim** 추가 — `mujoco_py`가 없으면 `mujoco`(3.x)를 `sys.modules["mujoco_py"]`로 등록해 robomimic/LIBERO import가 통과되도록 함. (mujoco-py 빌드 불필요 → Python 3.12에서 직접 실행 가능.)
- **노트북 초기화 셀에서 처리하는 런타임 shim (소스 미수정, 세션 한정 패치)**
  - `pytorch3d.transforms` shim: `scipy`/`torch` 기반으로 `axis_angle_to_matrix`, `matrix_to_axis_angle`, `matrix_to_rotation_6d`, `rotation_6d_to_matrix` 구현.
  - `jax.random.KeyArray` 호환 shim (accelerate→flax→jax 연쇄 대응).
  - Python 3.12용 `hydra` mutable-default 패치 (`OverrideDirname`).
  - **`AsyncVectorEnv` → in-process `SyncVectorEnv` 치환**: Colab에서 fork 기반 멀티프로세스 MuJoCo가 EGL GPU 렌더링과 충돌하므로, env runner의 `AsyncVectorEnv`를 in-process `SyncVectorEnv`로 바꿔 안정화.
  - headless 렌더링을 위해 robosuite/mujoco import 이전에 `MUJOCO_GL=egl`(hang 시 `osmesa`) 설정.

---

## 3. CoT/LLM 오케스트레이션 동작 방식

```
[env obs] ─▶ CoTOrchestratedPolicy.predict_action
                │  replan_every 스텝마다:
                ├─▶ planner.plan(base_goal, step_index, obs_image) ─▶ CoTPlan(subgoal, language_goal, cot_trace)
                │       rule: 일반 매니퓰레이션 4단계(approach→grasp→move→release) 순환
                │       llm : OpenAI vision/text 호출로 현재 시점 subgoal 생성 (실패 시 rule fallback)
                └─▶ inner UVA.predict_action(obs, language_goal=재작성된 목표) ─▶ action
```

- `language_goal`은 `"{base_goal} | {subgoal}"` 형식으로 합쳐 CLIP 텍스트 인코더에 전달됩니다(길이 가드 포함).
- `num_candidates > 1`이면 여러 subgoal로 action을 뽑은 뒤 `pick_candidate_index`(`first` 또는 `smallest_delta`)로 하나 선택.
- 내부 정책 인터페이스 연동: `unified_video_action/policy/unified_video_action_policy.py`의 `predict_action(obs_dict, language_goal=None)` 시그니처와 `libero_image_runner.py`가 `language_goal=[...]`를 넘기는 경로를 그대로 사용합니다.

### 플래너 모드
| 모드 | API 키 | 설명 |
|------|--------|------|
| baseline (`--no_cot`) | 불필요 | 래퍼 없이 내부 UVA만 — `eval_sim.py`와 동일 |
| `rule` | 불필요 | 규칙 기반 4단계 subgoal 순환 |
| `llm` | `OPENAI_API_KEY` 필요 | OpenAI vision/text CoT. 키/에러 시 rule로 자동 fallback |

---

## 4. 사용법

### Colab
1. 해당 노트북을 Colab에서 연다 (`Runtime → GPU`).
2. 노트북 셀 안의 `UVA_REPO_URL` / `UVA_REPO_BRANCH`를 본인 fork로 수정.
3. 셀을 위에서부터 실행. `gym` 설치 단계에서 한 번 런타임 재시작이 필요할 수 있음(노트북 안내 참고).
4. smoke(1 task×1 ep) → light(10×3) → full(10×50, Pro+ 권장) 순으로 예산 조절.

### 로컬 / 서버 (NVIDIA GPU)
```bash
# baseline (eval_sim.py와 동일)
python eval_sim_cot.py -c checkpoints/libero10.ckpt -o outputs/base --no_cot

# rule 기반 CoT
python eval_sim_cot.py -c checkpoints/libero10.ckpt -o outputs/rule --planner rule --quick_test

# OpenAI vision CoT
export OPENAI_API_KEY=sk-...
python eval_sim_cot.py -c checkpoints/libero10.ckpt -o outputs/llm \
    --planner llm --llm_model gpt-4o-mini --quick_test --verbose_cot
```

### 플래너 출력만 확인 (GPU/시뮬레이터 불필요)
```bash
# rule (API 불필요)
python plan_cot_only.py --base_goal "put the bowl on the stove" --num_replans 4 -o outputs/plans_rule.json

# llm
export OPENAI_API_KEY=sk-...
python plan_cot_only.py --planner llm --base_goal "pick up the mug" --image frame.jpg -o outputs/plans_llm.json
```

---

## 5. 참고 수치 (LIBERO-10)
- 논문 Table I: **UVA = 0.90** (10 task × 50 rollout)
- Supplementary Table VIII: **UVA = 0.93** (30-test, task당 3 rollout)

CoT 노트북은 `baseline` / `rule` / `llm` 의 `test_mean_score`를 같은 조건에서 비교하기 위한 것입니다.

## 6. 트러블슈팅 (요약)
- `Union from diffusers.optimization` → `diffusers==0.18.2` 고정 후 재실행 (또는 패치된 `lr_scheduler.py` 사용).
- `mujoco gladLoadGL error` → import 전에 `MUJOCO_GL=egl` 미설정. 런타임 재시작 후 첫 셀부터.
- `AsyncVectorEnv reset()에서 hang` → `n_envs=1` + in-process `SyncVectorEnv` 패치 사용.
- LLM이 계속 rule로 fallback → `OPENAI_API_KEY` 미설정 (trace에 `[LLM skipped]`).

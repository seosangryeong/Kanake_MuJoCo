# Kanake MuJoCo

KANAKE(KAERI-SNAKE IV) 16관절 뱀로봇을 MuJoCo에서 강화학습하는 패키지입니다. 머리 시선을 고정한 채 S자 serpentine으로 속도 명령을 따라가는 정책을 RSL-RL PPO로 학습합니다. 평지에서 시작해 Teraffe와 같은 난이도의 지형 커리큘럼(물결, 계단, 턱, 경사, 요철, 둔덕)까지 다룹니다. URDF와 STL은 패키지에 들어 있고, 학습에는 ROS 2 빌드가 필요 없습니다.

## 설치

Python 3.10 이상에서 실행합니다. 이미 `rl/.venv`가 있으면 건너뛰어도 됩니다.

```bash
cd ~/ros2_ws/src/kanake_mujoco
bash rl/setup_env.sh
```

## 빠른 실행

```bash
# 제어판 UI: 정책 불러오기, 지형·레벨·마찰 변경, 키보드/자동 주행
rl/.venv/bin/python rl/play_ui.py

# 키보드 조종 (뷰어 창에서 누르고 있는 동안: ↑ 가속, ↓ 브레이크, ←/→ 회전, Shift+←/→ 시선)
rl/.venv/bin/python rl/teleop.py --checkpoint rl/runs/<run>/model_XXXX.pt

# 고정 명령 시나리오로 정량 평가 (속도·머리 시선·미끄러짐)
rl/.venv/bin/python rl/evaluate.py --checkpoint rl/runs/<run>/model_XXXX.pt --head

# TensorBoard
rl/.venv/bin/python rl/logs.py
```

`play.py`와 `teleop.py`에서 `--terrain random --terrain-level 0~4`로 지형을, `--collision only|both`로 충돌 캡슐 표시를 고를 수 있습니다. 뷰어에서 Backspace는 누르지 마세요. 뷰어가 데이터를 리셋해 정책 상태와 어긋납니다.

## 학습 명령 (버전별)

학습 출력은 `rl/runs/<run>/`에 저장됩니다(커밋하지 않음). 한 번에 하나만 돌리는 것을 기준으로 `--workers 16`(CPU 프로세스), `--num-envs 256`이 기본입니다.

```bash
# v4: 평지, j2 목 관절, 머리 yaw 명령 ±40°
rl/.venv/bin/python rl/train.py --randomize --head-yaw-range 0.7 --iterations 3000 --log-dir rl/runs/serp_v4_neck

# v7: v4에서 이어서 지형 커리큘럼 (지형에서는 외력 방식 이방성 마찰 필요)
rl/.venv/bin/python rl/train.py --resume rl/runs/serp_v4_neck/model_2999.pt \
  --terrain random --lateral force --randomize --iterations 3000 --log-dir rl/runs/serp_v7_terrain

# v8: 몸통 pitch 확장, 고유감각 이력, 지형 critic, 뒤집힘·밀기·노이즈 랜덤화
rl/.venv/bin/python rl/train.py --init-from rl/runs/serp_v7_terrain/model_5999.pt \
  --climb --climb-pitch 1.2 --friction-lateral 1.5 --randomize \
  --flip-probability 0.25 --push-interval 8 --obs-noise 0.01 --iterations 4000 --log-dir rl/runs/serp_v8_climb

# v9: v8 + 파형 관절별 진폭 액션, 접촉 기반 지지 보상
rl/.venv/bin/python rl/train.py --init-from rl/runs/serp_v8_climb/model_3999.pt \
  --amplitude-per-joint --support-reward --iterations 4000 --log-dir rl/runs/serp_v9_amp
```

- `--resume`은 같은 관측·액션 구조에서 정책, 옵티마이저, 정규화를 이어받습니다.
- `--init-from`은 관측이 뒤에 추가되거나 액션 범위·구조가 바뀐 경우의 warm start입니다. 새 입력 가중치는 0으로, 바뀐 액션은 같은 관절 명령이 나오도록 다시 매핑해서 시작 시점에 이전 정책과 똑같이 움직입니다.

## 물리 모델

| 항목 | 설정 |
|---|---|
| 기저 | tail이 자유 기저, `j1`(머리 쪽) ~ `j16` 구동. 짝수 관절은 yaw(수평), 홀수 관절은 pitch(수직) |
| 충돌 형상 | 링크마다 캡슐 1개 (반지름 19 mm, 커버 외경). CAD 메쉬는 보이기만 함. 등방성 조건에서 메쉬와 결과가 같음 |
| 마찰 | 링크 축 방향 0.2, 옆 방향 1.0(v8부터 1.5). serpentine은 옆 > 축이어야 전진함. 등방성 0.7(원래 설정)에서는 뒤로 감 |
| 이방성 구현 | `pair`: 캡슐–평면 접촉 쌍 (평면에서만 접선이 링크 축에 정렬됨). `force`: 접촉은 축 마찰로 두고 옆 초과분을 접촉점에 외력으로 가함 (heightfield·박스·메쉬에서도 동작, [friction.py](kanake_mujoco/friction.py)) |
| 관절 | 위치 제어 kp 3, kv 1, ±85°, 토크 10 N·m. 400 Hz 물리, 40 Hz 정책 |

## 액션

정책은 관절 각도를 직접 내지 않고, yaw 관절에 흐르는 serpenoid 진행파와 관절별 보정값을 냅니다 ([env.py](kanake_mujoco/env.py)).

```
yaw 관절 k (머리→꼬리):  θ_k = A_k · sin(φ − k·π/4) + γ + r_k     φ += 2π f dt
pitch 관절:              θ   = r
```

| 항목 | v2–v8 | v9 (`--amplitude-per-joint`) |
|---|---|---|
| 진폭 A | 공통 1개, 0–60° | 파형 관절(j4–j16)마다 1개 |
| 주파수 f | 0.3–1.2 Hz (끌 수 없음) | 동일 |
| 조향 γ | ±29° | 동일 |
| 보정 r | 16개. 머리 j1 ±1.0, 목 j2 ±1.2, 몸통 ±0.15 rad (climb: pitch ±0.8–1.2) | 동일 |
| 액션 수 | 19 | 25 |

파형 파라미터는 0.2 s, 몸통 보정은 80 ms 저역통과 필터를 거칩니다. 필터는 보정값만으로 고주파 진동을 만들어 이방성 마찰 위를 미끄러지는 편법(v1)을 막습니다.

## 명령, 관측, 보상

- **명령:** 월드 속도 벡터(0–0.15 m/s, 방향은 몸 방향 ±1 rad)와 진행 방향 기준 머리 yaw 오프셋(±40°). 모든 방향 관측은 명령 진행 방향 기준입니다.
- **관측:** 명령, 머리 yaw 오차·각속도·자세, 몸 방향 오차, COM 속도(필터/순간), 꼬리 자세, 관절 위치·속도·토크, CPG 위상·파라미터, 필터된 보정값, 이전 액션. climb에서는 0.5 s 고유감각 이력(위치 오차, 토크, 머리·꼬리 자세)이 붙습니다.
- **critic (climb):** 정책 관측 + 몸 주변 지형 높이 55점, 마찰, 지형 단계, 몸 바로 섬 정도. 학습에만 쓰는 특권 정보입니다.
- **보상:** COM 속도 추종, 머리 yaw 추종(좁은·넓은 커널), 머리 3축 각속도, 머리 흔들림, 몸 방향, 머리 수평(climb은 지면 기울기 기준), 측면 미끄러짐, pitch·보정 크기, 액션 변화, 전력. 랜덤화를 켜면 바로 섬, climb은 뜬 링크(또는 접촉 없는 링크) 감점이 더해집니다.

## 지형과 도메인 랜덤화

- **지형 (`--terrain random`):** Teraffe `procedural_terrain.py`와 같은 7종 heightfield(8 m × 8 m, 1.6 cm 격자). 최종 난이도도 Teraffe와 같습니다(높이 12–30 cm, 계단 6–12 cm, 턱 10–20 cm, 파장 1.2–2.8 m). 5단계 높이 배율 0.15–1.0, 평지 비율 80–5%. 환경마다 에피소드 속도 오차로 승급·강등합니다 ([procedural.py](kanake_mujoco/procedural.py)).
- **랜덤화 (`--randomize`):** 링크 질량 ±10%, 마찰 축 ×0.7–1.4 / 옆 ×0.8–1.2. 추가로 `--flip-probability`(굴린 상태 시작), `--push-interval`(옆·롤 충격), `--obs-noise`, kp/kv ±10%.

## 정책 버전

| 실험 | 내용 |
|---|---|
| serp_v1 | 실패: 정책이 파형을 끄고 보정값 진동으로 미끄러짐 |
| serp_v2 | 평지 serpentine 기본, 시선 고정 |
| serp_v3_head | 머리 yaw 명령 ±40° (j2가 파형 포함) |
| serp_v4_neck | 평지 최종: j2 목 관절 분리, 머리 흔들림 감소 |
| serp_v5/v6_lift | 실패: 머리 5–10 cm 들기. 목 자세 사전 + kp 10에서도 제자리 버팀 |
| serp_v7_terrain | v4 + 지형 커리큘럼 (외력 이방성 마찰) |
| serp_v8_climb | v7 + 옆 마찰 1.5, 몸통 pitch ±1.2 rad, 이력, 지형 critic, 뒤집힘·밀기 |
| serp_v9_amp | v8 + 관절별 진폭, 접촉 기반 지지 보상 |

UI 정책 목록의 설명은 [rl/config/policy_catalog.json](rl/config/policy_catalog.json)에 있습니다.

## 폴더 구성

```text
kanake_mujoco/         URDF/STL, 모델 조립, 마찰, 지형, 단일 환경
  model.py             캡슐 충돌, 이방성 마찰(pair), heightfield 지형
  friction.py          외력 방식 이방성 마찰
  procedural.py        지형 은행과 커리큘럼 단계
  env.py               액션·관측·보상·랜덤화
rl/train.py            PPO 학습, 이어 학습(--resume), warm start(--init-from)
rl/play.py             정책 재생
rl/teleop.py           키보드/조이스틱 조종
rl/play_ui.py          Teraffe식 제어판 UI
rl/evaluate.py         시나리오 평가
rl/kanake_rl/          병렬 환경(멀티프로세스), PPO 설정, 제어판, 정책 목록
rl/third_party/        RSL-RL 2.3.1 소스
```

ROS 2 Humble에서 빌드하면 `ros2 run kanake_mujoco view --terrain park`로 모델을 볼 수 있습니다. URDF와 STL의 출처는 `IsaacLab_snake/kanake6_sim_523/kanake6_sim/kanake6_sim`, 경사로·공원 지형은 `teraffe_mujoco/teraffe_mujoco/terrain.py`입니다.

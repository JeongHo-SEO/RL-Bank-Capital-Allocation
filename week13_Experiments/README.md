```text
[공통 인프라 - 건드리지 않음]
Lending Club → Segment Calibration → Simulator 구축
                                            ↓
                              여기서부터 PPO vs ML 갈림

[PPO]                              [ML]
Simulator 안에서                   Simulator 안에서
Actor-Critic으로                   ML 모델이
매달 배분 결정                     매달 배분 결정
        ↓                                  ↓
        └──────── 동일 Simulator ──────────┘
                         ↓
                   성능 비교

[PPO] - Simulator 안에서 매달:

state_t 관찰
    ↓
PPO Actor → action_t 결정 (31차원 배분벡터)
    ↓
Simulator가 transition 계산
    ↓
reward_t 받아서 다음 학습에 반영


[ML]

state_t 관찰
    ↓
BC ML 모델 → action_t 결정
    ↓
Simulator가 transition 계산 (동일)
    ↓
reward는 기록만 (학습 없음)
```

***

- 김재환님 베이스라인 수정 반영하였음.
- baseline들이 "내가 쓸 수 있는 돈"을 계산할 때 최소 유지해야 할 현금을 빼지 않고 전체 예산을 다 써버려서 파산 => 수정하셨다고 함.

## Experiments

### 메인 비교

- PPO (Attention) vs ML (Behavioral Cloning == BC)
- "RL vs 지도학습" 가장 대조적
- Simulator 안에서 매달 state를 받아 action을 결정하는 주체만 다름
  - PPO:  state_t → Actor-Critic → action_t (학습된 policy)
  - BC:   state_t → ML 모델     → action_t (지도학습된 모델)

1. train_segment_ppo.py와 동일하게  데이터 로딩 → segment calibration 구축
2. budget_aware_heuristic으로 simulator 돌리며 (state, action) 쌍 수집
3. MLP 또는 XGBoost로 지도학습
4. 학습된 모델을 simulator에 넣고 PPO와 동일한 방식으로 평가
5. test_policy_results.csv 저장

### 서브 비교 1

PPO (Attention) vs PPO (GRU 시계열패턴)
→ Actor 구조의 영향

### 서브 비교 2

ML Risk Scoring: LR vs XGBoost <- 성능 up
→ upstream 품질의 영향

***

## Running COde

`Baselines` directory 터미널에서 실행.

```powershell
# 기존 방식: LR + PPO(Attention)
python train_segment_ppo.py --policy-architecture attention --scorer lr --episodes 800 --output-dir outputs/PPO_LR_attention_800

# Main 비교: - LR + BC(MLP), LP + BC(XGB)
python train_bc.py --bc-model mlp --scorer lr --collect-episodes 800 --bc-epochs 50 --output-dir outputs/BC_MLP_LR_ep800_epoch50 # MLP: collect 800, 50 epochs 학습
python train_bc.py --bc-model xgb --scorer lr --collect-episodes 800 --output-dir outputs/BC_XGB_LR_ep800 # XGB: collect 800, n_estimators=200 (tree)

# Sub1: Actor 구조 비교 - LR + PPO(GRU)
python train_segment_ppo.py --policy-architecture gru --scorer lr --episodes 800 --output-dir outputs/PPO_LR_gru_800

# Sub2: Scorer 비교 - XGB + PPO(Attention)
python train_segment_ppo.py --policy-architecture attention --scorer xgb --episodes 800 --output-dir outputs/PPO_XGB_attention_800
```

### Performance

`test_policy_results.csv`

- cumulative_profit_mean   ← 누적수익 평균
- liquidated_episode_count ← 파산 횟수 (100 에피소드 중)

#### AUC => 큰 차이 없음.

1. LR
2. XGB

```json
"training_rows": 32803,
"validation_auc": 0.7003193654297885,
"test_auc": 0.6746368238155477,
"model": "logistic_regression_temporal",
```

```json
"training_rows": 32803,
"validation_auc": 0.7011935662444555,
"test_auc": 0.6744042206936538,
"model": "xgboost_temporal",
```
# RL 모델 전체 요약

## 데이터 역할 분리

| | Lending Club 데이터 | 시뮬레이터 |
|---|---|---|
| 역할 | 시뮬레이터 보정 재료 | RL 학습 환경 |
| 쓰이는 시점 | 학습 전 (준비 단계) | 학습 중 (실시간) |

왜 Lending Club에서 바로 학습 안 했나:
1. 약 130개 월별 timestep → RL에 필요한 수천 episode에 턱없이 부족
2. Counterfactual 문제: 역사 데이터는 과거 은행이 실제로 한 결정의 결과만 기록됨.
   RL이 "다른 선택을 했을 때의 결과"를 탐색하려면 시뮬레이터가 필요.

---

## 전체 파이프라인

```text
Lending Club 데이터 (2007~2018)
        ↓
   데이터 전처리
        ↓
   ML Risk Scoring      ← Logistic Regression
        ↓
  Segment Calibration   ← 30개 그룹 분류 + 그룹별 통계 추정
        ↓
  Bank Simulator 구축   ← 시뮬레이터 = RL의 환경
        ↓
   PPO 정책 학습        ← 시뮬레이터 안에서 학습
        ↓
  Monte Carlo 평가      ← 100번 반복 rollout
```

---

## 1단계: 데이터 전처리

- 사후 결과 변수 제거 (Leakage 방지): `loan_status`, `total_pymnt`, `int_rate`, `grade` 등
- 시간순 분할 (랜덤 분할 X)

```text
Train 70%  →  Validation 20%  →  Test 10%
과거                                미래
```

---

## 2단계: ML Risk Scoring

목적: 차주 특성 23개 → risk_score 1개로 압축 후 5개 bucket 분류

- 모델: Logistic Regression
- Validation AUC: 0.700 / Test AUC: 0.675
- 정확한 분류보다 위험 순서 ranking이 목적

---

## 3단계: Segment Calibration

목적: 시뮬레이터 파라미터 추정을 위해 차주를 30개 그룹으로 묶기

```text
개별 차주 72,755명
        ↓  "비슷한 것끼리 묶기"
    30개 segment = risk bucket (5) × 대출금액 (3) × 대출기간 (2)
        ↓  "그룹별 통계 추정"
  PD, 이자율, 회수율, 수익률, 월별수요
        ↓  "시뮬레이터 입력값으로 사용"
    Simulator
```

각 segment마다 historical data에서 추정:

| 추정값 | 의미 |
|---|---|
| PD | 부도 확률 |
| interest_rate | 평균 이자율 |
| recovery_rate | 부도 시 회수율 (전체 고정: 10%) |
| expected_profit_rate | 기대 수익률 |
| monthly_demand | 월별 대출 신청 금액 |

---

## 4단계: Simulator (MDP 환경)

### State S_t (331차원)

| 변수 | 차원 | 의미 |
|---|---|---|
| B_t | 1 | 현재 현금 / $50M (정규화) |
| G_t | 30 | 이번 달 segment별 대출 수요 |
| E_t | 180 = 30×6 | segment × 만기구간별 대출 잔액 |
| θ | 120 = 30×4 | 각 segment의 PD, 이자율, 회수율, 기대수익률 |

### Action A_t (30차원)

```text
Actor 출력 (31차원 raw) → 2단계 변환

raw[0]    → sigmoid → deploy_fraction   "예산의 몇 % 쓸지"
raw[1:31] → softmax → weights           "어느 segment에 얼마 비율로"

최종 배분 x_j = deploy_fraction × 배분가능금액 × weights_j
               단, x_j ≤ G_j (수요 초과 불가)
```

### Transition

```text
B_{t+1} = B_t
        + 원금상환 + 이자수입 + 부도회수
        - 자금조달비용 - 운영비용 - 인플레이션손실
        - 신규대출 실행액

부도는 매달 binomial sampling으로 확률적 발생:
  hazard_j = 1 - (1 - PD_j)^(1/term_j)
  default_count ~ Binomial(n, hazard_j)
```

### Reward

```text
R_t = period_profit_t / B_0  -  λ × (shortfall_t / B_0)²

period_profit = 이자수입 - 부도손실 - 자금조달비용 - 운영비용 - 인플레이션손실

shortfall = max(0,  B_min + κ × outstanding  -  B_t)
                    └─────── 요구 유동성 ───────┘
```

| 파라미터 | 값 |
| --- | --- |
| B_0 (초기 현금) | $50M |
| B_min (최소 유보) | $10M (= 20% × $50M) |
| κ (노출액 유보 비율) | 0.05 |
| λ (페널티 가중치) | 1.0 (grid search: 0, 0.1, 1, 10, 100) |

### Episode 종료 조건

```text
청산 조건 1: B_t < 요구 유동성         → 유동성 부족
청산 조건 2: 자본 / outstanding < 8%  → BIS 자기자본비율 위반

청산 시: 자산의 60% 손실, 에피소드 즉시 종료
정상 종료: 36개월 후 → 60개월 terminal runoff 수익 추가 합산
```

### 학습 시나리오 (6가지 랜덤 교체)

| 시나리오 | 변경 내용 |
|---|---|
| base | 기본 |
| default_1.5x | 부도율 1.5배 |
| high_risk_demand_1.5x | 고위험 수요 1.5배 |
| recovery_0.5x | 회수율 0.5배 |
| liquidity_0.7x | 초기 현금 70% |
| combined_stress | 위 3개 동시 |

---

## 5단계: PPO 학습

- 알고리즘: PPO (Proximal Policy Optimization)
- 구조: Attention Actor-Critic

```text
S_t (331차원)
    ↓
[Token embedding]      ← 각 segment를 token으로 변환
    ↓
[Transformer Encoder]  ← segment 간 상대 비교 (2 layers, 4 heads)
    ↓
Actor  → (u_t, w_t) → 배분 결정
Critic → V(S_t) 추정
```

Attention을 쓴 이유: 30개 segment 중 어디가 지금 더 매력적인지 상대 비교가 필요한 구조.

| PPO 파라미터 | 값 |
|---|---|
| episodes | 800 |
| γ (할인율) | 0.99 |
| learning_rate | 3e-4 |
| clip_coef | 0.20 |
| gae_lambda | 0.95 |
| entropy_coef | 0.01 |
| value_coef | 0.50 |
| ppo_epochs | 6 |
| minibatch_size | 1024 |
| hidden_size | 256 |
| attention_heads | 4 |
| attention_layers | 2 |

---

## 6단계: 평가

100개 seed로 모든 정책 동일 환경에서 비교 (Common Random Numbers).

### Baseline 6종

| Baseline | 전략 |
|---|---|
| reject_all | 모두 거절 |
| fixed_conservative | 저위험 segment만, 55% 배분 |
| fixed_aggressive | 전 segment, 90% 배분 |
| greedy_expected_profit | 기대수익률 높은 순서대로 채움 |
| budget_aware_heuristic | 현금 상태에 따라 공격성 조절 |
| one_step_lp_closed_form | 단기 LP 최적해 (= greedy와 동일) |

### 최종 결과 (Test 기준, 단위: USD)

| 정책 | 누적 수익 | 청산 |
|---|---|---|
| PPO | +$0.40M | 0/100 |
| reject_all | -$3.58M | 0/100 |
| greedy_expected_profit | -$30.0M | 100/100 |
| one_step_lp | -$30.0M | 100/100 |
| fixed_conservative | -$30.0M | 100/100 |
| fixed_aggressive | -$30.0M | 100/100 |
| budget_aware_heuristic | -$30.1M | 100/100 |

### Stress Test 결과 (PPO만, Test 기준)

| 시나리오 | PPO 수익 | 청산 |
|---|---|---|
| 고위험 수요 1.5x | +$0.40M | 0/100 |
| 유동성 0.7x | +$0.28M | 0/100 |
| 회수율 0.5x | -$0.05M | 0/100 |
| 부도율 2x | -$11.3M | 0/100 |
| 복합 스트레스 | -$4.0M | 0/100 |

---

## 한계점

1. Accepted loans만 사용 → 실제 전체 신청자 풀이 아님
2. Counterfactual 검증 불가 → 시뮬레이터 안에서 더 낫다는 것만 증명
3. Segment 평균화 → 같은 그룹 내 개인차 무시
4. Simulator assumption 의존 → 실제 은행 도입 전 더 정교한 모델 필요

---

## 한 줄 요약

> 과거 데이터로 시뮬레이터를 만들고, RL이 그 안에서 "얼마나 공격적으로 대출할지"를 상황에 맞게 학습.
> 고정 전략들은 전부 청산되는 환경에서 유일하게 생존 + 수익 달성.
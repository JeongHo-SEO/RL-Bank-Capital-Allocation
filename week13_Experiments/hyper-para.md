## 하이퍼파라미터 정리

### 시뮬레이터 파라미터 (금융 도메인)

| 파라미터 | 값 | 의미 |
|---|---|---|
| `initial_cash` | $50,000,000 | 초기 은행 현금 |
| `min_cash_ratio` | 0.20 | 최소 유보 현금 비율 → $10M |
| `reserve_exposure_ratio` (κ) | 0.05 | outstanding 잔액의 5%도 유보 |
| `recovery_rate` | 0.10 | 부도 시 10% 회수 |
| `funding_cost_annual` | 0.03 | 연 3% 자금조달비용 |
| `servicing_cost_rate` | 0.005 | 연 0.5% 운영비용 |
| `expected_inflation_annual` | 0.025 | 연 2.5% 인플레이션 |
| `horizon_months` | 36 | 의사결정 기간 |
| `terminal_runoff_months` | 60 | 종료 후 자연상환 기간 |

### Reward 파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| `liquidity_penalty_lambda` (λ) | 1.0 | 유동성 페널티 가중치 (grid search: 0, 0.1, 1, 10, 100) |
| `TERMINAL_VALUE_WEIGHT` | 0.0 | Terminal NPV 가중치 (비활성화) |
| `TERMINAL_RUNOFF_DISCOUNT` | 0.99 | Runoff 월별 할인율 |
| `MIN_CAPITAL_RATIO` | 0.08 | BIS 자기자본비율 최소 기준 |
| `LIQUIDITY_LIQUIDATION_HAIRCUT` | 0.60 | 청산 시 자산 손실 비율 |

### PPO 학습 파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| `episodes` | 800 (final) / 300 (default) | 총 학습 에피소드 수 |
| `rollout_episodes` | 16 (final) / 8 (default) | 업데이트당 rollout 수 |
| `learning_rate` | 3e-4 | 학습률 |
| `gamma` (γ) | 0.99 | 장기 보상 할인율 |
| `gae_lambda` | 0.95 | GAE Advantage 추정 파라미터 |
| `clip_coef` | 0.20 | PPO clipping 범위 |
| `entropy_coef` | 0.01 | 탐색 엔트로피 계수 |
| `value_coef` | 0.50 | Critic loss 가중치 |
| `ppo_epochs` | 6 | rollout당 업데이트 반복 횟수 |
| `minibatch_size` | 1024 | 미니배치 크기 |

### 네트워크 구조

| 파라미터 | 값 | 의미 |
|---|---|---|
| `hidden_size` | 256 | Hidden layer 크기 |
| `policy_architecture` | attention | Attention Actor-Critic |
| `attention_heads` | 4 | Multi-head attention 헤드 수 |
| `attention_layers` | 2 | Attention 레이어 수 |
| `attention_dropout` | 0.0 | Dropout 비율 |

### Segment 정의

| 파라미터 | 값 | 의미 |
|---|---|---|
| `risk_buckets` | 5 | 위험도 구간 수 |
| `amount_tiers` | 3 | 대출금액 구간 수 |
| `term_buckets` | 2 | 대출기간 구간 수 (36 / 60개월) |
| `maturity_buckets` | 6 | 만기 잔여 구간 수 |
| **total segments** | **30** | 5 × 3 × 2 |
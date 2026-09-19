# Simulation-Based RL for Liquidity-Constrained Lending Allocation

Final topic:

```text
Simulation-Based Reinforcement Learning for Bank-Level
Liquidity-Constrained Monthly Lending Allocation
```

The project no longer treats Lending Club history as logged trajectories for
direct offline-RL evaluation. Historical data is used to calibrate a simulator:

```text
Lending Club Data
-> Default Probability Model
-> Risk x Amount x Term Segments
-> Calibrated Bank Lending Simulator
-> PPO Capital Allocation Policy
-> Monte Carlo Evaluation and Stress Tests
```

The defensible claim is:

```text
The RL policy is evaluated in a simulated lending environment calibrated from
Lending Club data, not proven optimal in the real Lending Club market.
```

## Role Split

```text
ML predicts risk.
RL allocates bank capital.
```

ML2 loan-condition recommendation was removed. The RL policy now makes the
bank-level monthly allocation decision.

## Main MDP

State:

```text
S_t = (R_t, G_t, E_t, theta)
```

- `R_t`: global risk state, including current liquidity, time remaining,
  capital, required liquidity, liquidity coverage, risk-weighted assets,
  capital adequacy, observed demand/default regime multipliers, and
  good/recovery opportunity, stress, risk-weight, repayment-delay, and
  fatigue multipliers
- `G_t`: current monthly applicant demand by borrower segment
- `E_t`: outstanding exposure by segment and remaining maturity bucket
- `theta`: segment calibration parameters observed by the actor

Default configuration:

```text
J = 5 risk buckets x 3 amount tiers x 2 term buckets = 30 segments
H = 6 maturity buckets
R_t = 16 global features
theta_j = (PD_j, interest_j, recovery_j, expected_profit_rate_j,
           risk_adjusted_expected_profit_rate_j, fatigue_j)
state dimension = 16 + J + JH + 6J = 406
```

The global risk features expose finite-horizon timing and liquidation-relevant
ratios directly to the actor and critic. The observed regime features make
future opportunity cost explicit: favorable regimes concentrate demand in
high-EPR segments, and post-stress recovery regimes concentrate demand in
lower-risk positive-EPR segments. Stress regimes can now create liquidity
drawdowns, funding shocks, capital hits, default clustering, repayment delay,
and risk-weight inflation. Repeatedly filling the same segment accumulates
fatigue, raising future default risk and reducing margin/demand quality.
Actor masks, myopic greedy/LP scores, and optional EPR priors use
`risk_adjusted_expected_profit_rate`, not raw EPR. It subtracts expected
default-cluster, liquidity/funding/capital shock, capital-charge,
repayment-delay, and fatigue costs from base segment economics, which allows
the LP residual policy to choose zero lending when all risk-adjusted scores are
negative. The final `6J` observation entries expose segment-level simulator
parameters, dynamic risk-adjusted EPR, and current segment fatigue to the PPO policy, reducing the
oracle-information gap with calibration-aware myopic baselines during
validation, test, and stress evaluation.

Action:

```text
A_t = x_t in R_+^J
0 <= x_t,j <= G_t,j
```

The default experiment now uses reserve-aware deployable capacity for PPO,
matching the safe-capacity rule used by the greedy and one-step LP baselines:

```text
deployable_capacity =
  max(0, (cash - required_liquidity) / (1 + reserve_exposure_ratio))
```

The old soft-reserve behavior remains available with
`--no-enforce-min-cash-constraint`, but it is not the default comparison mode.

The PPO actor emits raw values that are converted inside the Gymnasium
environment:

```text
actor(S_t) -> (gate_t, u_t, w_t) -> feasible allocation x_t
```

By default, this is a residual policy over `one_step_lp_closed_form`.
`gate_t` is a sigmoid no-lending gate that can shut LP deployment down to zero,
`u_t` adjusts total deployment around the baseline, and `w_t` adjusts segment
log-shares around the baseline allocation. Use `--policy-action-mode direct`
to recover the direct softmax allocation parameterization, whose raw action
dimension remains 31.

Transition:

```text
B_{t+1} = B_t + repayments + recoveries
          - funding/servicing costs - new disbursements
E_{t+1} = UpdateLoanBook(E_t, x_t, defaults, repayments)
```

Training Objective:

```text
RequiredLiquidity_t = B_min + kappa * OutstandingExposure_t
period_profit = interest income - default loss - funding cost - servicing cost
maximize E[period_profit / initial_cash]
subject to liquidation_or_shortfall_metric <= threshold
```

The default trainer uses primal-dual PPO with a tighter 3% liquidation target.
The fixed liquidity penalty is set to zero inside the simulator when
`--primal-dual` is active; PPO subtracts a learned dual penalty from rollout
rewards and updates that dual variable from episode constraint violations.
Step-level soft risk shaping also subtracts LCR/CAR buffer violations
(`--soft-risk-lcr-target`, `--soft-risk-car-target`) so the actor gets a smooth
warning signal before rare liquidation events. Disable primal-dual with
`--no-primal-dual` to return to a fixed `--liquidity-penalty-lambda` scalarized
reward.

Monthly interest income, funding cost, and servicing cost are all computed on
beginning-of-month outstanding exposure. This is equivalent to treating default
events as settling at month end. The simulator reports only `funding_cost` and
`servicing_cost`; the older `operating_cost` alias is not emitted.

At the terminal decision month, the simulator runs the remaining loan book
forward with no new lending and adds discounted runoff profit/loss. By default
`--terminal-value-weight` is `0.0`, so no additional residual terminal expected
NPV is credited unless explicitly enabled. Outputs report `decision_period_profit`,
`terminal_runoff_profit`, and `terminal_value` separately; `cumulative_profit`
is their sum.

Training and evaluation both use liquidation-on-breach. The simulator checks a
simplified capital adequacy ratio:

```text
capital adequacy ratio = capital / risk_weighted_assets
risk_weighted_assets = outstanding loan-book exposure
minimum capital adequacy ratio = 8%
```

Liquidity shortfall and BIS-style capital adequacy are not the same metric. The
rollout rule treats both as failure modes, but reports their triggers
separately. If capital adequacy falls below 8%, the episode is terminated by
`capital_adequacy` liquidation. If cash falls below required liquidity first,
the episode is terminated by `liquidity_shortfall` liquidation, modeling a
profitable-but-illiquid failure. Liquidity liquidation applies a 60% haircut to
current assets:

```text
current_assets = cash_budget + outstanding loan-book exposure
liquidity_liquidation_loss = 0.60 * current_assets
liquidation_proceeds = 0.40 * current_assets
```

After liquidation, no terminal runoff or terminal value is credited. The
liquidation loss is subtracted from evaluation cumulative profit.

For clarity, outputs report both `min_liquidity_coverage_ratio`
(`budget / required_liquidity`) and `min_capital_adequacy_ratio`
(`capital / risk_weighted_assets`). A positive liquidity shortfall can coexist
with a healthy capital adequacy ratio. Policy result tables also include
`episode_count`, `liquidated_episode_count`,
`capital_liquidation_episode_count`, and
`liquidity_liquidation_episode_count`, so the trigger and number of liquidated
episodes are explicit. Result tables also report `excess_profit_over_reject_mean`,
`liquidation_rate`, `liquidation_constrained_profit_mean`,
`cumulative_profit_cvar_5_mean`, `cumulative_profit_worst_decile_mean`, and
`stress_regime_profit_mean` so absolute P&L, excess-over-reject, and tail-risk
frontiers can be read separately.

Borrower risk scores are used to define risk buckets. Segment default
probabilities are then calibrated from empirical segment-level bad-loan rates,
not from `mean(risk_score) * scale`.

Because the full state-action space is continuous and high-dimensional, exact
DP is not the main method. PPO is used as an approximate actor-critic method.

## Key Files

- `train_segment_ppo.py`: main PPO training, evaluation, and stress-test script
- `lending_rl_agent/segment_simulator.py`: segment calibration, Gymnasium environment, baselines, rollout evaluation
- `lending_rl_agent/ppo.py`: lightweight PyTorch PPO implementation
- `lending_rl_agent/data.py`: Lending Club loading, leakage-aware cleaning, temporal risk scoring

Older threshold/DQN experiments have been moved under `legacy/` so the root
workspace only exposes the final segment-level PPO methodology.

## Install

```bash
python3 -m pip install -r requirements-neural.txt
```

`gymnasium` is required for the segment allocation environment.

## Quick Smoke Test

```bash
python3 train_segment_ppo.py \
  --max-loans 20000 \
  --episodes 4 \
  --rollout-episodes 2 \
  --eval-episodes 2 \
  --horizon-months 6 \
  --hidden-size 64 \
  --minibatch-size 64 \
  --ppo-epochs 2 \
  --output-dir outputs/segment_ppo_final_smoke
```

Expected structural check:

```text
state dimension = 406
raw action dimension = 32 for residual mode, 31 for direct mode
allocation dimension = 30
```

## Main Training Run

```bash
python3 train_segment_ppo.py \
  --max-loans 120000 \
  --episodes 2000 \
  --rollout-episodes 128 \
  --eval-episodes 1000 \
  --horizon-months 36 \
  --hidden-size 256 \
  --minibatch-size 8192 \
  --ppo-epochs 4 \
  --liquidity-penalty-lambda 10 \
  --reserve-exposure-ratio 0.05 \
  --gpu-default-mode normal \
  --min-expected-profit-rate 0.0 \
  --policy-action-mode residual \
  --residual-baseline-policy one_step_lp_closed_form \
  --residual-use-lend-gate \
  --residual-deploy-scale 3.0 \
  --residual-scale 0.75 \
  --bc-deploy-target-fraction 0.35 \
  --allocation-temperature 0.7 \
  --allocation-top-k 10 \
  --demand-regime-mode ar1 \
  --stress-shock \
  --primal-dual \
  --primal-dual-constraint-metric liquidated \
  --primal-dual-constraint-threshold 0.03 \
  --soft-risk-penalty-lambda 0.25 \
  --soft-risk-lcr-target 1.05 \
  --soft-risk-car-target 0.10 \
  --terminal-runoff-months 60 \
  --output-dir outputs/segment_ppo_final
```

Repeated runs reuse scored-loan cache files under `outputs/cache`. Use
`--refresh-cache` after changing data preparation or segment calibration logic.
The final training pipeline uses reservoir sampling, scaled monthly demand,
stochastic default sampling by default, reserve-aware PPO capacity,
negative-EV segment masking, liquidation-on-breach, terminal runoff, and
stress-augmented train simulators. Demand uses a persistent AR(1) macro regime
that multiplies current demand and default hazard, with stronger demand movement
in higher-risk segments. Stress regimes add liquidity shocks, funding shocks,
capital hits, repayment delay, risk-weight inflation, and default clustering.
Segment fatigue makes repeated concentration raise future default risk and
reduce margin/demand quality. The actor therefore sees base and dynamic segment
parameters, risk-adjusted EPR, and observed regime multipliers during PPO training without
requiring separate methodology switches. Allocation concentration can be swept with
`--allocation-temperature`, `--allocation-top-k`, `--segment-epr-prior-scale`,
and `--expected-profit-shaping-weight`.

Optional BC warm start is available for stronger PPO variants:

```bash
python3 train_segment_ppo.py \
  --bc-warmstart-policy one_step_lp_closed_form \
  --bc-warmstart-episodes 128 \
  --bc-warmstart-epochs 3 \
  --output-dir outputs/ppo_bc_one_step_lp
```

This imitates the myopic baseline's segment ranking first, but by default
targets only 35% of the baseline deployment amount by training the residual
lend gate, not by forcing the deploy head to mimic full LP size. PPO
fine-tuning then learns when to open the LP overlay and when to reject lending
in the same stochastic, liquidation-aware simulator.

## Lambda Sweep

Run PPO trainings for different initial dual/fixed-liquidity penalties and
generate the frontier figure:

```bash
python3 scripts/run_lambda_sweep.py \
  --episodes 400 \
  --output-root outputs/lambda_sweep_400
```

For a longer final sweep, use `--episodes 800`. Extra training arguments can be
passed after `--`, for example:

```bash
python3 scripts/run_lambda_sweep.py \
  --episodes 400 \
  --output-root outputs/lambda_sweep_400 \
  -- --max-loans 120000 --eval-episodes 100
```

The wrapper trains λ values `{0, 0.1, 1, 10, 100}`. With primal-dual PPO enabled
these values initialize the dual variable and the dual then adapts during
training; with `--no-primal-dual` they are fixed scalarized liquidity penalties.
The wrapper writes one run directory per λ and saves
`lambda_frontier_validation.png` and `lambda_frontier_test.png`
under the sweep `figures/` directory. It also saves shortfall-based frontier
figures using expected shortfall as the liquidity-risk severity axis. Compare
`cumulative_profit_mean`, `liquidated_episode_count`,
`expected_shortfall_mean`, `min_liquidity_coverage_ratio_mean`, and
`min_capital_adequacy_ratio_mean` across runs to build the profit-liquidation
frontier. The reserve threshold is exposure-adjusted:
`B_min + kappa * outstanding exposure`.

To re-score an existing sweep without retraining, use:

```bash
python3 scripts/evaluate_existing_lambda_sweep.py \
  --input-root outputs/lambda_sweep_400 \
  --output-root outputs/lambda_sweep_400_haircut_liquidation_eval
```

This reloads each saved `segment_ppo.pt` checkpoint and reruns only the
validation/test/stress evaluation under the current liquidation-on-breach rule.

## Evaluation

Each policy is evaluated by Monte Carlo rollouts in validation and test
simulators. Baselines:

- `reject_all`
- `fixed_conservative`
- `fixed_aggressive`
- `greedy_expected_profit`
- `budget_aware_heuristic`
- `one_step_lp_closed_form`
- `ppo_rl`

`one_step_lp_closed_form` is the closed-form solution of the one-period linear
allocation problem: with linear per-dollar profit coefficients and one budget
constraint, the LP reduces to sorting segments by expected profit rate.

Metrics:

- cumulative profit
- terminal budget
- terminal runoff profit
- terminal expected NPV
- liquidity breach probability
- minimum liquidity coverage ratio
- expected shortfall
- expected shortfall ratio
- max shortfall
- capital adequacy ratio
- capital breach count
- max drawdown
- default loss
- funding cost
- servicing cost
- liquidation rate and liquidation loss
- gross and net cash inflow
- disbursed capital
- final-decision disbursement
- final-to-preterminal disbursement ratio
- final disbursement share
- capital utilization
- mean allocation by risk bucket
- primal-dual lambda and constraint violation during training
- mean demand/default/opportunity/stress/fatigue regime multipliers

Evaluation uses common random numbers: every policy is rolled out over the same
episode seed list, reducing comparison noise between PPO and baselines.

Sampling note: reservoir samples carry a demand scale factor so monthly demand
is expanded back toward full-file Lending Club demand. The risk scorer excludes
historical policy/loan-condition variables `int_rate`, `installment`, `grade`,
and `sub_grade`; those columns are still used for simulator calibration.

Stress scenarios:

- default hazard `2x`
- high-risk demand `1.5x`
- starting liquidity `0.7x`
- recovery `0.5x`
- combined stress

## Outputs

The selected output directory contains:

- `metrics.json`: full methodology, calibration, PPO config, validation/test/stress results
- `ppo_training_history.csv`: episode-level PPO training metrics
- `validation_policy_results.csv`: validation policy comparison, including decision/runoff/terminal profit and funding/servicing cost summaries
- `test_policy_results.csv`: holdout scenario policy comparison, including decision/runoff/terminal profit and funding/servicing cost summaries
- `validation_policy_episodes.csv`: rollout-level validation outcomes
- `test_policy_episodes.csv`: rollout-level test outcomes
- `stress_*_policy_results.csv`: stress-test comparisons
- `segment_ppo.pt`: trained PPO checkpoint

## Presentation Claim

Use this wording:

```text
Historical Lending Club data is used to calibrate the simulator, not to
directly evaluate the learned policy's counterfactual value.
```

And:

```text
This is an MDP, so Bellman equations exist. However, exact dynamic programming
is infeasible because the state includes continuous liquidity and
segment-by-maturity loan-book exposure plus segment calibration parameters,
while the action is a 30-dimensional continuous constrained allocation vector.
We therefore use simulator rollouts and train an approximate actor-critic
policy with PPO.
```

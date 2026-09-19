from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lending_rl_agent import load_accepted_loans, prepare_scored_loans_temporal
from lending_rl_agent.ppo import PPOAgent, PPOConfig, train_ppo, train_ppo_torch_batch
from lending_rl_agent.segment_simulator import (
    GLOBAL_STATE_NAMES,
    SEGMENT_PARAMETER_COUNT,
    SegmentCalibration,
    SegmentLendingEnv,
    budget_aware_segment_policy,
    evaluate_segment_policy,
    fixed_aggressive_policy,
    fixed_conservative_policy,
    greedy_expected_profit_policy,
    one_step_lp_policy,
    reject_all_policy,
)
from lending_rl_agent.torch_segment_simulator import (
    TorchSegmentBatchEnv,
    build_torch_batch_config,
    evaluate_torch_policy_batch,
    torch_policy_allocation,
)

SAMPLE_MODE = "reservoir"
TERMINAL_RUNOFF_DISCOUNT = 0.99
TERMINAL_VALUE_WEIGHT = 0.0
TRAIN_CALIBRATION_AUGMENTATION = "stress"
MIN_CAPITAL_RATIO = 0.08
LIQUIDITY_LIQUIDATION_HAIRCUT = 0.60
SIMULATOR_BACKEND = "torch"
ALLOCATION_TEMPERATURE = 1.0
SEGMENT_EPR_PRIOR_SCALE = 0.0
EXPECTED_PROFIT_SHAPING_WEIGHT = 0.0
ALLOCATION_TOP_K = 0
MIN_EXPECTED_PROFIT_RATE = 0.0
ENFORCE_MIN_CASH_CONSTRAINT = True
POLICY_ACTION_MODE = "residual"
RESIDUAL_BASELINE_POLICY = "one_step_lp_closed_form"
RESIDUAL_SCALE = 0.50
RESIDUAL_DEPLOY_SCALE = 2.50
RESIDUAL_USE_LEND_GATE = True
RESIDUAL_LEND_GATE_THRESHOLD = 0.02
BC_DEPLOY_TARGET_FRACTION = 0.35
DEMAND_REGIME_MODE = "ar1"
DEMAND_REGIME_RHO = 0.85
DEMAND_REGIME_VOLATILITY = 0.35
HIGH_RISK_DEMAND_SENSITIVITY = 1.00
DEFAULT_REGIME_SENSITIVITY = 0.90
OBSERVE_REGIME_STATE = True
OPPORTUNITY_REGIME = True
GOOD_OPPORTUNITY_SENSITIVITY = 2.20
RECOVERY_OPPORTUNITY_SENSITIVITY = 1.80
RECOVERY_OPPORTUNITY_DECAY = 0.70
RECOVERY_STRESS_THRESHOLD = 0.65
STRESS_SHOCK = True
STRESS_THRESHOLD = 0.60
STRESS_BASE_PROBABILITY = 0.01
STRESS_EVENT_PROBABILITY = 0.15
STRESS_EVENT_BONUS = 0.35
LIQUIDITY_SHOCK_EXPOSURE_FRAC = 0.012
FUNDING_SHOCK_EXPOSURE_FRAC = 0.002
CAPITAL_SHOCK_EXPOSURE_FRAC = 0.003
BASE_RISK_WEIGHT_MULTIPLIER = 5.5
STRESS_RISK_WEIGHT_INFLATION = 0.30
REPAYMENT_DELAY_STRENGTH = 0.25
DURATION_MISMATCH_STRENGTH = 0.12
STRESS_DEFAULT_CLUSTER_SENSITIVITY = 0.55
FATIGUE_DECAY = 0.90
FATIGUE_GROWTH = 0.18
FATIGUE_CAP = 3.0
FATIGUE_DEFAULT_SENSITIVITY = 0.25
FATIGUE_MARGIN_SENSITIVITY = 0.04
FATIGUE_DEMAND_SENSITIVITY = 0.08
USE_RISK_ADJUSTED_EPR = True
RISK_ADJUSTED_DEFAULT_COST_WEIGHT = 0.65
RISK_ADJUSTED_SHOCK_COST_WEIGHT = 1.00
RISK_ADJUSTED_CAPITAL_CHARGE_WEIGHT = 0.03
RISK_ADJUSTED_REPAYMENT_DELAY_WEIGHT = 0.75
RISK_ADJUSTED_FATIGUE_COST_WEIGHT = 1.00
SOFT_RISK_PENALTY_LAMBDA = 0.25
SOFT_RISK_LCR_TARGET = 1.05
SOFT_RISK_CAR_TARGET = 0.10
PRIMAL_DUAL = True
PRIMAL_DUAL_CONSTRAINT_METRIC = "liquidated"
PRIMAL_DUAL_CONSTRAINT_THRESHOLD = 0.03
PRIMAL_DUAL_LR = 0.10
PRIMAL_DUAL_MAX_LAMBDA = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO for segment-level liquidity-constrained lending allocation."
    )
    parser.add_argument("--accepted-path", default="archive 2/accepted_2007_to_2018Q4.csv.gz")
    parser.add_argument("--max-loans", type=int, default=120_000)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--risk-buckets", type=int, default=5)
    parser.add_argument("--amount-tiers", type=int, default=3)
    parser.add_argument("--term-buckets", type=int, default=2)
    parser.add_argument("--maturity-buckets", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--horizon-months", type=int, default=36)
    parser.add_argument("--initial-cash", type=float, default=50_000_000.0)
    parser.add_argument("--min-cash-ratio", type=float, default=0.20)
    parser.add_argument(
        "--reserve-exposure-ratio",
        type=float,
        default=0.05,
        help="Additional liquidity reserve required per dollar of outstanding exposure.",
    )
    parser.add_argument("--liquidity-penalty-lambda", type=float, default=1.0)
    parser.add_argument("--terminal-runoff-months", type=int, default=60)
    parser.add_argument(
        "--expected-inflation-annual",
        type=float,
        default=0.025,
        help="Annual expected inflation drag on idle cash, applied monthly in real-value terms.",
    )
    parser.add_argument("--recovery-rate", type=float, default=0.15)
    parser.add_argument("--funding-cost-annual", type=float, default=0.02)
    parser.add_argument("--servicing-cost-rate", type=float, default=0.003)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--policy-architecture", choices=["attention", "mlp"], default="attention")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gpu-default-mode",
        choices=["normal", "expected"],
        default="normal",
        help="GPU simulator default model. normal samples stochastic default losses; expected is deterministic.",
    )
    parser.add_argument(
        "--min-expected-profit-rate",
        type=float,
        default=MIN_EXPECTED_PROFIT_RATE,
        help="Hard-mask actor lending to segments at or below this expected-profit-rate threshold.",
    )
    parser.add_argument(
        "--negative-epr-penalty",
        type=float,
        default=0.0,
        help="Reward penalty for actor probability mass assigned to masked unprofitable segments.",
    )
    parser.add_argument(
        "--enforce-min-cash-constraint",
        action=argparse.BooleanOptionalAction,
        default=ENFORCE_MIN_CASH_CONSTRAINT,
        help="Use reserve-aware deployable capacity for PPO, matching the baseline safe-capacity rule.",
    )
    parser.add_argument(
        "--allocation-temperature",
        type=float,
        default=ALLOCATION_TEMPERATURE,
        help="Softmax temperature for segment allocation logits; lower values make allocations sharper.",
    )
    parser.add_argument(
        "--segment-epr-prior-scale",
        type=float,
        default=SEGMENT_EPR_PRIOR_SCALE,
        help="Add this multiple of segment expected-profit rate to actor segment logits.",
    )
    parser.add_argument(
        "--expected-profit-shaping-weight",
        type=float,
        default=EXPECTED_PROFIT_SHAPING_WEIGHT,
        help="Optional reward shaping weight for expected deployment profit.",
    )
    parser.add_argument(
        "--allocation-top-k",
        type=int,
        default=ALLOCATION_TOP_K,
        help="If positive, restrict actor allocation to the top-k eligible segment logits each step.",
    )
    parser.add_argument(
        "--terminal-value-weight",
        type=float,
        default=TERMINAL_VALUE_WEIGHT,
        help="Weight on residual terminal expected-NPV after terminal runoff.",
    )
    parser.add_argument(
        "--policy-action-mode",
        choices=["direct", "residual"],
        default=POLICY_ACTION_MODE,
        help="Use direct softmax allocation or residual corrections over a baseline allocation.",
    )
    parser.add_argument(
        "--residual-baseline-policy",
        choices=[
            "budget_aware_heuristic",
            "fixed_aggressive",
            "fixed_conservative",
            "one_step_lp_closed_form",
            "greedy_expected_profit",
            "reject_all",
        ],
        default=RESIDUAL_BASELINE_POLICY,
        help="Baseline allocation used when --policy-action-mode residual is enabled.",
    )
    parser.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE)
    parser.add_argument("--residual-deploy-scale", type=float, default=RESIDUAL_DEPLOY_SCALE)
    parser.add_argument(
        "--residual-use-lend-gate",
        action=argparse.BooleanOptionalAction,
        default=RESIDUAL_USE_LEND_GATE,
        help="In residual mode, add a separate sigmoid gate that can turn LP lending nearly to zero.",
    )
    parser.add_argument("--residual-lend-gate-threshold", type=float, default=RESIDUAL_LEND_GATE_THRESHOLD)
    parser.add_argument(
        "--bc-warmstart-policy",
        choices=[
            "none",
            "budget_aware_heuristic",
            "fixed_aggressive",
            "fixed_conservative",
            "one_step_lp_closed_form",
            "greedy_expected_profit",
        ],
        default="none",
        help="Optionally imitation-pretrain the actor on a myopic baseline before PPO fine-tuning.",
    )
    parser.add_argument(
        "--bc-warmstart-episodes",
        type=int,
        default=0,
        help="Number of baseline rollout episodes to use per BC epoch. Set with --bc-warmstart-policy.",
    )
    parser.add_argument("--bc-warmstart-epochs", type=int, default=3)
    parser.add_argument(
        "--bc-learning-rate",
        type=float,
        default=None,
        help="Learning rate for BC warm start. Defaults to --learning-rate.",
    )
    parser.add_argument(
        "--bc-deploy-target-fraction",
        type=float,
        default=BC_DEPLOY_TARGET_FRACTION,
        help="During BC warm start, imitate baseline ranking but target only this fraction of baseline deploy.",
    )
    parser.add_argument(
        "--demand-regime-mode",
        choices=["iid", "ar1"],
        default=DEMAND_REGIME_MODE,
        help="Demand process. ar1 applies persistent macro multipliers to demand and default hazard.",
    )
    parser.add_argument("--demand-regime-rho", type=float, default=DEMAND_REGIME_RHO)
    parser.add_argument("--demand-regime-volatility", type=float, default=DEMAND_REGIME_VOLATILITY)
    parser.add_argument("--high-risk-demand-sensitivity", type=float, default=HIGH_RISK_DEMAND_SENSITIVITY)
    parser.add_argument("--default-regime-sensitivity", type=float, default=DEFAULT_REGIME_SENSITIVITY)
    parser.add_argument(
        "--observe-regime-state",
        action=argparse.BooleanOptionalAction,
        default=OBSERVE_REGIME_STATE,
        help="Expose current demand/default regime multipliers in the global observation state.",
    )
    parser.add_argument(
        "--opportunity-regime",
        action=argparse.BooleanOptionalAction,
        default=OPPORTUNITY_REGIME,
        help="Make good/recovery regimes concentrate higher-quality demand, creating cash opportunity cost.",
    )
    parser.add_argument(
        "--good-opportunity-sensitivity",
        type=float,
        default=GOOD_OPPORTUNITY_SENSITIVITY,
        help="Demand boost for high-EPR segments when the observed regime turns favorable.",
    )
    parser.add_argument(
        "--recovery-opportunity-sensitivity",
        type=float,
        default=RECOVERY_OPPORTUNITY_SENSITIVITY,
        help="Demand boost for low-risk positive-EPR segments after stress eases.",
    )
    parser.add_argument(
        "--recovery-opportunity-decay",
        type=float,
        default=RECOVERY_OPPORTUNITY_DECAY,
        help="Persistence of post-stress recovery opportunity demand.",
    )
    parser.add_argument(
        "--recovery-stress-threshold",
        type=float,
        default=RECOVERY_STRESS_THRESHOLD,
        help="Regime level above which a downward transition creates recovery opportunity.",
    )
    parser.add_argument(
        "--stress-shock",
        action=argparse.BooleanOptionalAction,
        default=STRESS_SHOCK,
        help="Enable regime-dependent liquidity, capital, funding, default-cluster, and repayment-delay shocks.",
    )
    parser.add_argument("--stress-threshold", type=float, default=STRESS_THRESHOLD)
    parser.add_argument("--stress-base-probability", type=float, default=STRESS_BASE_PROBABILITY)
    parser.add_argument("--stress-event-probability", type=float, default=STRESS_EVENT_PROBABILITY)
    parser.add_argument("--stress-event-bonus", type=float, default=STRESS_EVENT_BONUS)
    parser.add_argument("--liquidity-shock-exposure-frac", type=float, default=LIQUIDITY_SHOCK_EXPOSURE_FRAC)
    parser.add_argument("--funding-shock-exposure-frac", type=float, default=FUNDING_SHOCK_EXPOSURE_FRAC)
    parser.add_argument("--capital-shock-exposure-frac", type=float, default=CAPITAL_SHOCK_EXPOSURE_FRAC)
    parser.add_argument("--base-risk-weight-multiplier", type=float, default=BASE_RISK_WEIGHT_MULTIPLIER)
    parser.add_argument("--stress-risk-weight-inflation", type=float, default=STRESS_RISK_WEIGHT_INFLATION)
    parser.add_argument("--repayment-delay-strength", type=float, default=REPAYMENT_DELAY_STRENGTH)
    parser.add_argument("--duration-mismatch-strength", type=float, default=DURATION_MISMATCH_STRENGTH)
    parser.add_argument("--stress-default-cluster-sensitivity", type=float, default=STRESS_DEFAULT_CLUSTER_SENSITIVITY)
    parser.add_argument("--fatigue-decay", type=float, default=FATIGUE_DECAY)
    parser.add_argument("--fatigue-growth", type=float, default=FATIGUE_GROWTH)
    parser.add_argument("--fatigue-cap", type=float, default=FATIGUE_CAP)
    parser.add_argument("--fatigue-default-sensitivity", type=float, default=FATIGUE_DEFAULT_SENSITIVITY)
    parser.add_argument("--fatigue-margin-sensitivity", type=float, default=FATIGUE_MARGIN_SENSITIVITY)
    parser.add_argument("--fatigue-demand-sensitivity", type=float, default=FATIGUE_DEMAND_SENSITIVITY)
    parser.add_argument(
        "--use-risk-adjusted-epr",
        action=argparse.BooleanOptionalAction,
        default=USE_RISK_ADJUSTED_EPR,
        help="Use risk-adjusted EPR for actor masks, priors, and myopic LP/greedy scores.",
    )
    parser.add_argument("--risk-adjusted-default-cost-weight", type=float, default=RISK_ADJUSTED_DEFAULT_COST_WEIGHT)
    parser.add_argument("--risk-adjusted-shock-cost-weight", type=float, default=RISK_ADJUSTED_SHOCK_COST_WEIGHT)
    parser.add_argument("--risk-adjusted-capital-charge-weight", type=float, default=RISK_ADJUSTED_CAPITAL_CHARGE_WEIGHT)
    parser.add_argument("--risk-adjusted-repayment-delay-weight", type=float, default=RISK_ADJUSTED_REPAYMENT_DELAY_WEIGHT)
    parser.add_argument("--risk-adjusted-fatigue-cost-weight", type=float, default=RISK_ADJUSTED_FATIGUE_COST_WEIGHT)
    parser.add_argument("--soft-risk-penalty-lambda", type=float, default=SOFT_RISK_PENALTY_LAMBDA)
    parser.add_argument("--soft-risk-lcr-target", type=float, default=SOFT_RISK_LCR_TARGET)
    parser.add_argument("--soft-risk-car-target", type=float, default=SOFT_RISK_CAR_TARGET)
    parser.add_argument(
        "--primal-dual",
        action=argparse.BooleanOptionalAction,
        default=PRIMAL_DUAL,
        help="Train PPO with a learned dual penalty on liquidation/shortfall constraint violations.",
    )
    parser.add_argument(
        "--primal-dual-constraint-metric",
        choices=[
            "liquidated",
            "liquidity_breach",
            "expected_shortfall",
            "lcr_buffer_violation",
            "car_buffer_violation",
            "soft_risk_penalty",
        ],
        default=PRIMAL_DUAL_CONSTRAINT_METRIC,
    )
    parser.add_argument("--primal-dual-constraint-threshold", type=float, default=PRIMAL_DUAL_CONSTRAINT_THRESHOLD)
    parser.add_argument("--primal-dual-lr", type=float, default=PRIMAL_DUAL_LR)
    parser.add_argument("--primal-dual-initial-lambda", type=float, default=None)
    parser.add_argument("--primal-dual-max-lambda", type=float, default=PRIMAL_DUAL_MAX_LAMBDA)
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/segment_ppo_final")
    args = parser.parse_args()
    args.sample_mode = SAMPLE_MODE
    args.soft_reserve = not bool(args.enforce_min_cash_constraint)
    args.terminal_runoff_discount = TERMINAL_RUNOFF_DISCOUNT
    args.terminal_value_weight = float(args.terminal_value_weight)
    args.train_calibration_augmentation = TRAIN_CALIBRATION_AUGMENTATION
    args.min_capital_ratio = MIN_CAPITAL_RATIO
    args.liquidity_liquidation_haircut = LIQUIDITY_LIQUIDATION_HAIRCUT
    args.simulator_backend = SIMULATOR_BACKEND
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored, scorer_summary, cache_info = load_or_build_scored(args)
    calibrations = build_calibrations(args, scored)
    train_scenarios = build_train_scenarios(args, calibrations["train"])
    train_env = build_env(
        args,
        train_scenarios[0]["calibration"],
        seed=args.seed,
        starting_cash=train_scenarios[0]["starting_cash"],
        scenario_name=train_scenarios[0]["name"],
        liquidate_on_breach=True,
    )
    torch_device = resolve_torch_device(args.device)
    torch_config = build_torch_batch_config(args)
    torch_dim_env = TorchSegmentBatchEnv(
        [train_scenarios[0]["calibration"]],
        torch_config,
        device=torch_device,
        starting_cash=[train_scenarios[0]["starting_cash"]],
        scenario_names=[train_scenarios[0]["name"]],
        seed=args.seed,
    )

    config = PPOConfig(
        episodes=args.episodes,
        rollout_episodes=args.rollout_episodes,
        update_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        device=args.device,
        architecture=args.policy_architecture,
        segment_count=calibrations["train"].segment_count,
        maturity_bucket_count=calibrations["train"].maturity_bucket_count,
        segment_parameter_count=SEGMENT_PARAMETER_COUNT,
        global_feature_count=int(getattr(train_env, "global_state_count", len(GLOBAL_STATE_NAMES))),
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        attention_dropout=args.attention_dropout,
        primal_dual=bool(args.primal_dual),
        dual_lambda=float(
            args.primal_dual_initial_lambda
            if args.primal_dual_initial_lambda is not None
            else args.liquidity_penalty_lambda
        ),
        dual_learning_rate=args.primal_dual_lr,
        dual_max_lambda=args.primal_dual_max_lambda,
        constraint_metric=args.primal_dual_constraint_metric,
        constraint_threshold=args.primal_dual_constraint_threshold,
    )
    initial_dual_lambda = float(config.dual_lambda)

    def train_env_factory(local_seed: int) -> SegmentLendingEnv:
        scenario = train_scenarios[int(local_seed) % len(train_scenarios)]
        return build_env(
            args,
            scenario["calibration"],
            seed=local_seed,
            starting_cash=scenario["starting_cash"],
            scenario_name=scenario["name"],
            liquidate_on_breach=True,
        )

    def train_batch_factory(episode_offset: int, batch_size: int) -> TorchSegmentBatchEnv:
        scenarios = [
            train_scenarios[(args.seed + episode_offset + index) % len(train_scenarios)]
            for index in range(batch_size)
        ]
        return TorchSegmentBatchEnv(
            [scenario["calibration"] for scenario in scenarios],
            torch_config,
            device=torch_device,
            starting_cash=[scenario["starting_cash"] for scenario in scenarios],
            scenario_names=[scenario["name"] for scenario in scenarios],
            seed=args.seed + episode_offset,
        )

    warmstart_agent = maybe_bc_warmstart(args, config, train_batch_factory, torch_dim_env)

    if SIMULATOR_BACKEND == "torch":
        agent, history = train_ppo_torch_batch(
            train_batch_factory,
            observation_dim=torch_dim_env.observation_dim,
            action_dim=torch_dim_env.action_dim,
            config=config,
            progress_every=args.progress_every,
            agent=warmstart_agent,
        )
    else:
        agent, history = train_ppo(
            train_env_factory,
            observation_dim=train_env.observation_space.shape[0],
            action_dim=train_env.action_space.shape[0],
            config=config,
            progress_every=args.progress_every,
            agent=warmstart_agent,
        )
    history.to_csv(output_dir / "ppo_training_history.csv", index=False)
    agent.save(
        output_dir / "segment_ppo.pt",
        metadata={
            "segment_count": calibrations["train"].segment_count,
            "maturity_bucket_count": calibrations["train"].maturity_bucket_count,
            "methodology": "segment_level_continuous_capital_allocation",
            "liquidate_on_breach_training": True,
            "policy_architecture": args.policy_architecture,
            "global_state_names": list(GLOBAL_STATE_NAMES),
            "min_expected_profit_rate": args.min_expected_profit_rate,
            "gpu_default_mode": args.gpu_default_mode,
            "enforce_min_cash_constraint": args.enforce_min_cash_constraint,
            "policy_action_mode": args.policy_action_mode,
            "residual_baseline_policy": args.residual_baseline_policy,
            "residual_scale": args.residual_scale,
            "residual_deploy_scale": args.residual_deploy_scale,
            "residual_use_lend_gate": args.residual_use_lend_gate,
            "residual_lend_gate_threshold": args.residual_lend_gate_threshold,
            "bc_warmstart_policy": args.bc_warmstart_policy,
            "bc_warmstart_episodes": args.bc_warmstart_episodes,
            "bc_warmstart_epochs": args.bc_warmstart_epochs,
            "bc_deploy_target_fraction": args.bc_deploy_target_fraction,
            "demand_regime_mode": args.demand_regime_mode,
            "demand_regime_rho": args.demand_regime_rho,
            "demand_regime_volatility": args.demand_regime_volatility,
            "high_risk_demand_sensitivity": args.high_risk_demand_sensitivity,
            "default_regime_sensitivity": args.default_regime_sensitivity,
            "observe_regime_state": args.observe_regime_state,
            "opportunity_regime": args.opportunity_regime,
            "good_opportunity_sensitivity": args.good_opportunity_sensitivity,
            "recovery_opportunity_sensitivity": args.recovery_opportunity_sensitivity,
            "recovery_opportunity_decay": args.recovery_opportunity_decay,
            "recovery_stress_threshold": args.recovery_stress_threshold,
            "stress_shock": args.stress_shock,
            "stress_threshold": args.stress_threshold,
            "stress_base_probability": args.stress_base_probability,
            "stress_event_probability": args.stress_event_probability,
            "stress_event_bonus": args.stress_event_bonus,
            "liquidity_shock_exposure_frac": args.liquidity_shock_exposure_frac,
            "funding_shock_exposure_frac": args.funding_shock_exposure_frac,
            "capital_shock_exposure_frac": args.capital_shock_exposure_frac,
            "base_risk_weight_multiplier": args.base_risk_weight_multiplier,
            "stress_risk_weight_inflation": args.stress_risk_weight_inflation,
            "repayment_delay_strength": args.repayment_delay_strength,
            "duration_mismatch_strength": args.duration_mismatch_strength,
            "stress_default_cluster_sensitivity": args.stress_default_cluster_sensitivity,
            "fatigue_decay": args.fatigue_decay,
            "fatigue_growth": args.fatigue_growth,
            "fatigue_cap": args.fatigue_cap,
            "fatigue_default_sensitivity": args.fatigue_default_sensitivity,
            "fatigue_margin_sensitivity": args.fatigue_margin_sensitivity,
            "fatigue_demand_sensitivity": args.fatigue_demand_sensitivity,
            "use_risk_adjusted_epr": args.use_risk_adjusted_epr,
            "risk_adjusted_default_cost_weight": args.risk_adjusted_default_cost_weight,
            "risk_adjusted_shock_cost_weight": args.risk_adjusted_shock_cost_weight,
            "risk_adjusted_capital_charge_weight": args.risk_adjusted_capital_charge_weight,
            "risk_adjusted_repayment_delay_weight": args.risk_adjusted_repayment_delay_weight,
            "risk_adjusted_fatigue_cost_weight": args.risk_adjusted_fatigue_cost_weight,
            "soft_risk_penalty_lambda": args.soft_risk_penalty_lambda,
            "soft_risk_lcr_target": args.soft_risk_lcr_target,
            "soft_risk_car_target": args.soft_risk_car_target,
            "primal_dual": args.primal_dual,
            "constraint_metric": args.primal_dual_constraint_metric,
            "initial_dual_lambda": initial_dual_lambda,
            "final_dual_lambda": float(config.dual_lambda),
        },
    )

    validation_results, validation_episodes = evaluate_all_policies(args, calibrations["validation"], agent, args.seed + 10_000)
    test_results, test_episodes = evaluate_all_policies(args, calibrations["test"], agent, args.seed + 20_000)
    write_policy_results(validation_results, output_dir / "validation_policy_results.csv")
    write_policy_results(test_results, output_dir / "test_policy_results.csv")
    validation_episodes.to_csv(output_dir / "validation_policy_episodes.csv", index=False)
    test_episodes.to_csv(output_dir / "test_policy_episodes.csv", index=False)

    stress_results = evaluate_stress_suite(args, calibrations["test"], agent)
    for name, result in stress_results.items():
        write_policy_results(result, output_dir / f"stress_{name}_policy_results.csv")

    metrics = {
        "topic": "Simulation-Based Reinforcement Learning for Bank-Level Liquidity-Constrained Monthly Lending Allocation",
        "methodology": "historical_data_to_segment_calibrated_simulator_to_ppo_policy_rollouts",
        "claim_scope": "simulated_lending_environment_calibrated_from_lending_club_data",
        "state": {
            "form": "S_t=(global_risk_state,G_t,E_t,theta)",
            "dimension": int(train_env.observation_space.shape[0]),
            "global_state_count": int(getattr(train_env, "global_state_count", len(GLOBAL_STATE_NAMES))),
            "global_state_names": list(GLOBAL_STATE_NAMES),
            "segment_count": calibrations["train"].segment_count,
            "maturity_bucket_count": calibrations["train"].maturity_bucket_count,
            "segment_parameter_count": SEGMENT_PARAMETER_COUNT,
            "segment_parameter_names": list(train_env._reset_info()["segment_parameter_names"]),
            "policy_architecture": args.policy_architecture,
        },
        "action": {
            "form": "actor_raw -> deployment_fraction_and_segment_weights -> feasible_allocation",
            "raw_dimension": int(train_env.action_space.shape[0]),
            "allocation_dimension": calibrations["train"].segment_count,
            "policy_action_mode": args.policy_action_mode,
            "residual_baseline_policy": args.residual_baseline_policy,
        },
        "cache": cache_info,
        "scorer": scorer_summary,
        "calibration": {name: calibration.metadata for name, calibration in calibrations.items()},
        "training_scenarios": [
            {
                "name": scenario["name"],
                "calibration_name": scenario["calibration"].name,
                "starting_cash": scenario["starting_cash"],
            }
            for scenario in train_scenarios
        ],
        "training": vars(args),
        "training_rule": {
            "liquidate_on_breach": True,
            "liquidity_liquidation_haircut": LIQUIDITY_LIQUIDATION_HAIRCUT,
            "expected_inflation_annual": args.expected_inflation_annual,
            "allocation_temperature": args.allocation_temperature,
            "segment_epr_prior_scale": args.segment_epr_prior_scale,
            "expected_profit_shaping_weight": args.expected_profit_shaping_weight,
            "allocation_top_k": args.allocation_top_k,
            "policy_action_mode": args.policy_action_mode,
            "residual_baseline_policy": args.residual_baseline_policy,
            "residual_scale": args.residual_scale,
            "residual_deploy_scale": args.residual_deploy_scale,
            "residual_use_lend_gate": args.residual_use_lend_gate,
            "residual_lend_gate_threshold": args.residual_lend_gate_threshold,
            "bc_deploy_target_fraction": args.bc_deploy_target_fraction,
            "enforce_min_cash_constraint": args.enforce_min_cash_constraint,
            "gpu_default_mode": args.gpu_default_mode,
            "min_expected_profit_rate": args.min_expected_profit_rate,
            "fixed_liquidity_penalty_lambda": args.liquidity_penalty_lambda,
            "effective_liquidity_penalty_lambda": effective_liquidity_penalty_lambda(args),
            "primal_dual": args.primal_dual,
            "primal_dual_constraint_metric": args.primal_dual_constraint_metric,
            "primal_dual_constraint_threshold": args.primal_dual_constraint_threshold,
            "primal_dual_initial_lambda": initial_dual_lambda,
            "primal_dual_final_lambda": float(config.dual_lambda),
            "primal_dual_lr": args.primal_dual_lr,
            "primal_dual_max_lambda": args.primal_dual_max_lambda,
            "demand_regime_mode": args.demand_regime_mode,
            "demand_regime_rho": args.demand_regime_rho,
            "demand_regime_volatility": args.demand_regime_volatility,
            "high_risk_demand_sensitivity": args.high_risk_demand_sensitivity,
            "default_regime_sensitivity": args.default_regime_sensitivity,
            "observe_regime_state": args.observe_regime_state,
            "opportunity_regime": args.opportunity_regime,
            "good_opportunity_sensitivity": args.good_opportunity_sensitivity,
            "recovery_opportunity_sensitivity": args.recovery_opportunity_sensitivity,
            "recovery_opportunity_decay": args.recovery_opportunity_decay,
            "recovery_stress_threshold": args.recovery_stress_threshold,
            "stress_shock": args.stress_shock,
            "stress_threshold": args.stress_threshold,
            "stress_base_probability": args.stress_base_probability,
            "stress_event_probability": args.stress_event_probability,
            "stress_event_bonus": args.stress_event_bonus,
            "liquidity_shock_exposure_frac": args.liquidity_shock_exposure_frac,
            "funding_shock_exposure_frac": args.funding_shock_exposure_frac,
            "capital_shock_exposure_frac": args.capital_shock_exposure_frac,
            "base_risk_weight_multiplier": args.base_risk_weight_multiplier,
            "stress_risk_weight_inflation": args.stress_risk_weight_inflation,
            "repayment_delay_strength": args.repayment_delay_strength,
            "duration_mismatch_strength": args.duration_mismatch_strength,
            "stress_default_cluster_sensitivity": args.stress_default_cluster_sensitivity,
            "fatigue_decay": args.fatigue_decay,
            "fatigue_growth": args.fatigue_growth,
            "fatigue_cap": args.fatigue_cap,
            "fatigue_default_sensitivity": args.fatigue_default_sensitivity,
            "fatigue_margin_sensitivity": args.fatigue_margin_sensitivity,
            "fatigue_demand_sensitivity": args.fatigue_demand_sensitivity,
            "use_risk_adjusted_epr": args.use_risk_adjusted_epr,
            "risk_adjusted_default_cost_weight": args.risk_adjusted_default_cost_weight,
            "risk_adjusted_shock_cost_weight": args.risk_adjusted_shock_cost_weight,
            "risk_adjusted_capital_charge_weight": args.risk_adjusted_capital_charge_weight,
            "risk_adjusted_repayment_delay_weight": args.risk_adjusted_repayment_delay_weight,
            "risk_adjusted_fatigue_cost_weight": args.risk_adjusted_fatigue_cost_weight,
            "soft_risk_penalty_lambda": args.soft_risk_penalty_lambda,
            "soft_risk_lcr_target": args.soft_risk_lcr_target,
            "soft_risk_car_target": args.soft_risk_car_target,
            "description": "Training and evaluation both terminate an episode on capital-adequacy or liquidity breach. Liquidity liquidation subtracts a 60% haircut on current assets.",
            "terminal_value_weight": args.terminal_value_weight,
        },
        "validation": validation_results,
        "test": test_results,
        "stress": stress_results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episodes": args.episodes,
                "device": args.device,
                "state_dimension": int(train_env.observation_space.shape[0]),
                "action_dimension": int(train_env.action_space.shape[0]),
                "validation_ppo": compact_policy_summary(validation_results.get("ppo_rl", {})),
                "test_ppo": compact_policy_summary(test_results.get("ppo_rl", {})),
            },
            indent=2,
        )
    )


def maybe_bc_warmstart(
    args: argparse.Namespace,
    config: PPOConfig,
    train_batch_factory,
    dim_env: TorchSegmentBatchEnv,
) -> PPOAgent | None:
    if args.bc_warmstart_policy == "none" or int(args.bc_warmstart_episodes) <= 0:
        return None
    if getattr(args, "simulator_backend", SIMULATOR_BACKEND) != "torch":
        raise ValueError("BC warm start is currently implemented for the torch simulator backend only.")

    agent = PPOAgent(
        observation_dim=dim_env.observation_dim,
        action_dim=dim_env.action_dim,
        config=config,
    )
    bc_warmstart_torch(agent, train_batch_factory, args, config)
    return agent


def bc_warmstart_torch(agent: PPOAgent, train_batch_factory, args: argparse.Namespace, config: PPOConfig) -> None:
    episodes = max(1, int(args.bc_warmstart_episodes))
    epochs = max(1, int(args.bc_warmstart_epochs))
    batch_size = max(1, int(args.rollout_episodes))
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=float(args.bc_learning_rate or args.learning_rate))
    agent.model.train()

    for epoch in range(epochs):
        completed = 0
        epoch_losses: list[float] = []
        while completed < episodes:
            current_batch = min(batch_size, episodes - completed)
            env = train_batch_factory(1_000_000 + epoch * episodes + completed, current_batch)
            obs = env.reset()
            while not bool(env.done.all().item()):
                active = ~env.done
                if not bool(active.any().item()):
                    break
                allocation = torch_policy_allocation(env, args.bc_warmstart_policy).detach()
                target = actor_targets_from_allocation(
                    env,
                    allocation,
                    config,
                    deploy_target_fraction=float(args.bc_deploy_target_fraction),
                ).detach()
                pred = agent.model.distribution(obs[active]).mean
                loss = (pred - target[active]).square().mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), config.max_grad_norm)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

                with torch.no_grad():
                    obs, _, _, _ = env.step_allocation(allocation)
            completed += current_batch
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(
            f"bc_warmstart_epoch={epoch + 1}/{epochs} "
            f"policy={args.bc_warmstart_policy} loss={mean_loss:.6f}",
            flush=True,
        )


def actor_targets_from_allocation(
    env: TorchSegmentBatchEnv,
    allocation: torch.Tensor,
    config: PPOConfig,
    deploy_target_fraction: float | None = None,
) -> torch.Tensor:
    if env.config.policy_action_mode == "residual":
        return residual_actor_targets_from_allocation(env, allocation, config, deploy_target_fraction=deploy_target_fraction)
    eps = 1e-6
    capacity = env.deployable_capacity().clamp_min(eps)
    disbursement = allocation.sum(dim=1)
    target_disbursement = disbursement
    if deploy_target_fraction is not None:
        target_disbursement = disbursement * float(np.clip(deploy_target_fraction, 0.0, 1.0))
    deploy_fraction = (target_disbursement / capacity).clamp(1e-4, 1.0 - 1e-4)
    deploy_logit = torch.logit(deploy_fraction).unsqueeze(1)

    eligible = env.profitable_mask
    eligible_count = eligible.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    fallback_weights = torch.where(eligible, 1.0 / eligible_count, torch.zeros_like(allocation))
    learned_weights = allocation / disbursement.unsqueeze(1).clamp_min(eps)
    weights = torch.where(disbursement.unsqueeze(1) > eps, learned_weights, fallback_weights)

    logits = torch.log(weights.clamp_min(eps))
    eligible_mean = (logits * eligible.float()).sum(dim=1, keepdim=True) / eligible_count
    logits = torch.where(eligible, logits - eligible_mean, torch.full_like(logits, -10.0))
    raw_segment_logits = max(float(env.config.allocation_temperature), 1e-3) * (
        logits - float(env.config.segment_epr_prior_scale) * env.risk_adjusted_expected_profit_rate()
    )
    return torch.cat([deploy_logit, raw_segment_logits], dim=1).clamp(
        -float(config.action_clip),
        float(config.action_clip),
    )


def residual_actor_targets_from_allocation(
    env: TorchSegmentBatchEnv,
    allocation: torch.Tensor,
    config: PPOConfig,
    deploy_target_fraction: float | None = None,
) -> torch.Tensor:
    eps = 1e-6
    base = env.residual_baseline_allocation().detach()
    base_sum = base.sum(dim=1)
    target_sum = allocation.sum(dim=1)
    target_multiplier = None
    if deploy_target_fraction is not None:
        max_multiplier = float(np.exp(max(float(env.config.residual_deploy_scale), 0.0))) - 1e-4
        target_multiplier = float(np.clip(deploy_target_fraction, 0.0, max(1.0, max_multiplier)))
        target_sum = base_sum * target_multiplier
    base_total = base_sum.clamp_min(eps)
    target_total = target_sum.clamp_min(eps)
    has_base = base_sum > eps
    deploy_scale = max(float(env.config.residual_deploy_scale), 0.0)
    if bool(env.config.residual_use_lend_gate):
        if target_multiplier is None:
            gate_target = (target_sum / base_total).clamp(1e-4, 1.0 - 1e-4)
        else:
            gate_target = torch.full_like(base_sum, float(np.clip(min(target_multiplier, 1.0), 1e-4, 1.0 - 1e-4)))
        gate_target = torch.where(has_base, gate_target, torch.full_like(gate_target, 1e-4))
        gate_raw = torch.logit(gate_target.clamp(1e-4, 1.0 - 1e-4)).unsqueeze(1)
        deploy_ratio_target = 1.0 if target_multiplier is None else max(target_multiplier / max(min(target_multiplier, 1.0), 1e-4), 1e-4)
        deploy_ratio = torch.full_like(base_total, float(deploy_ratio_target))
        deploy_raw = torch.zeros((allocation.shape[0], 1), dtype=torch.float32, device=allocation.device)
        if deploy_scale > eps:
            deploy_ratio = deploy_ratio.clamp(
                float(np.exp(-deploy_scale)) + 1e-4,
                float(np.exp(deploy_scale)) - 1e-4,
            )
            deploy_raw = torch.atanh((torch.log(deploy_ratio) / deploy_scale).clamp(-0.999, 0.999)).unsqueeze(1)
    else:
        gate_raw = None
        deploy_raw = None
    if not bool(env.config.residual_use_lend_gate):
        if deploy_scale <= eps:
            deploy_raw = torch.zeros((allocation.shape[0], 1), dtype=torch.float32, device=allocation.device)
        else:
            deploy_ratio = (target_total / base_total).clamp(
                float(np.exp(-deploy_scale)) + 1e-4,
                float(np.exp(deploy_scale)) - 1e-4,
            )
            deploy_raw = torch.atanh((torch.log(deploy_ratio) / deploy_scale).clamp(-0.999, 0.999)).unsqueeze(1)
    elif deploy_scale > eps and deploy_target_fraction is None:
        deploy_ratio = (target_total / base_total).clamp(
            float(np.exp(-deploy_scale)) + 1e-4,
            float(np.exp(deploy_scale)) - 1e-4,
        )
        deploy_raw = torch.atanh((torch.log(deploy_ratio) / deploy_scale).clamp(-0.999, 0.999)).unsqueeze(1)

    eligible = env.profitable_mask
    eligible_count = eligible.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    base_weights = base / base_total.unsqueeze(1)
    learned_weights = allocation / allocation.sum(dim=1, keepdim=True).clamp_min(eps)
    fallback_weights = torch.where(eligible, 1.0 / eligible_count, torch.zeros_like(allocation))
    target_weights = torch.where(allocation.sum(dim=1, keepdim=True) > eps, learned_weights, fallback_weights)
    log_delta = torch.log(target_weights.clamp_min(eps)) - torch.log(base_weights.clamp_min(eps))
    log_delta = torch.where(eligible, log_delta, torch.zeros_like(log_delta))
    log_delta = log_delta - (log_delta * eligible.float()).sum(dim=1, keepdim=True) / eligible_count
    residual_raw = (
        max(float(env.config.allocation_temperature), 1e-3)
        * log_delta
        / max(float(env.config.residual_scale), eps)
    )
    parts = [deploy_raw, residual_raw] if gate_raw is None else [gate_raw, deploy_raw, residual_raw]
    return torch.cat(parts, dim=1).clamp(
        -float(config.action_clip),
        float(config.action_clip),
    )


def build_calibrations(args: argparse.Namespace, scored: pd.DataFrame) -> dict[str, SegmentCalibration]:
    train = SegmentCalibration.from_scored(
        scored,
        split="train",
        risk_bucket_count=args.risk_buckets,
        amount_tier_count=args.amount_tiers,
        term_bucket_count=args.term_buckets,
        maturity_bucket_count=args.maturity_buckets,
        recovery_rate=args.recovery_rate,
        funding_cost_annual=args.funding_cost_annual,
        servicing_cost_rate=args.servicing_cost_rate,
    )
    calibrations = {"train": train}
    for split in ["validation", "test"]:
        if len(scored[scored["split"] == split]):
            calibrations[split] = SegmentCalibration.from_scored(
                scored,
                split=split,
                risk_edges=train.risk_edges,
                amount_edges=train.amount_edges,
                risk_bucket_count=args.risk_buckets,
                amount_tier_count=args.amount_tiers,
                term_bucket_count=args.term_buckets,
                maturity_bucket_count=args.maturity_buckets,
                recovery_rate=args.recovery_rate,
                funding_cost_annual=args.funding_cost_annual,
                servicing_cost_rate=args.servicing_cost_rate,
            )
    missing = [split for split in ["validation", "test"] if split not in calibrations]
    if missing:
        raise ValueError(f"Temporal split did not produce required holdout splits: {missing}")
    return calibrations


def build_train_scenarios(args: argparse.Namespace, train: SegmentCalibration) -> list[dict]:
    base = {"name": "base", "calibration": train, "starting_cash": None}
    return [
        base,
        {
            "name": "default_1p5x",
            "calibration": train.stressed("train_default_1p5x", default_multiplier=1.5),
            "starting_cash": None,
        },
        {
            "name": "high_risk_demand_1p5x",
            "calibration": train.stressed("train_high_risk_demand_1p5x", high_risk_demand_multiplier=1.5),
            "starting_cash": None,
        },
        {
            "name": "recovery_0p5x",
            "calibration": train.stressed("train_recovery_0p5x", recovery_multiplier=0.5),
            "starting_cash": None,
        },
        {
            "name": "liquidity_0p7x",
            "calibration": train,
            "starting_cash": args.initial_cash * 0.70,
        },
        {
            "name": "combined_stress",
            "calibration": train.stressed(
                "train_combined_stress",
                default_multiplier=1.5,
                high_risk_demand_multiplier=1.5,
                recovery_multiplier=0.5,
            ),
            "starting_cash": args.initial_cash * 0.70,
        },
    ]


def build_env(
    args: argparse.Namespace,
    calibration: SegmentCalibration,
    seed: int,
    starting_cash: float | None = None,
    scenario_name: str | None = None,
    liquidate_on_breach: bool = False,
) -> SegmentLendingEnv:
    return SegmentLendingEnv(
        calibration=calibration,
        initial_cash=args.initial_cash,
        min_cash_ratio=args.min_cash_ratio,
        horizon_months=args.horizon_months,
        liquidity_penalty_lambda=effective_liquidity_penalty_lambda(args),
        seed=seed,
        starting_cash=starting_cash,
        enforce_min_cash_constraint=bool(args.enforce_min_cash_constraint),
        reserve_exposure_ratio=args.reserve_exposure_ratio,
        terminal_runoff_months=args.terminal_runoff_months,
        terminal_runoff_discount=TERMINAL_RUNOFF_DISCOUNT,
        terminal_value_weight=args.terminal_value_weight,
        scenario_name=scenario_name,
        liquidate_on_breach=liquidate_on_breach,
        min_capital_ratio=MIN_CAPITAL_RATIO,
        liquidity_liquidation_haircut=LIQUIDITY_LIQUIDATION_HAIRCUT,
        expected_inflation_annual=args.expected_inflation_annual,
        allocation_temperature=args.allocation_temperature,
        segment_epr_prior_scale=args.segment_epr_prior_scale,
        expected_profit_shaping_weight=args.expected_profit_shaping_weight,
        allocation_top_k=args.allocation_top_k,
        min_expected_profit_rate=args.min_expected_profit_rate,
        policy_action_mode=args.policy_action_mode,
        residual_baseline_policy=args.residual_baseline_policy,
        residual_scale=args.residual_scale,
        residual_deploy_scale=args.residual_deploy_scale,
        residual_use_lend_gate=args.residual_use_lend_gate,
        residual_lend_gate_threshold=args.residual_lend_gate_threshold,
        demand_regime_mode=args.demand_regime_mode,
        demand_regime_rho=args.demand_regime_rho,
        demand_regime_volatility=args.demand_regime_volatility,
        high_risk_demand_sensitivity=args.high_risk_demand_sensitivity,
        default_regime_sensitivity=args.default_regime_sensitivity,
        observe_regime_state=args.observe_regime_state,
        opportunity_regime=args.opportunity_regime,
        good_opportunity_sensitivity=args.good_opportunity_sensitivity,
        recovery_opportunity_sensitivity=args.recovery_opportunity_sensitivity,
        recovery_opportunity_decay=args.recovery_opportunity_decay,
        recovery_stress_threshold=args.recovery_stress_threshold,
        stress_shock=args.stress_shock,
        stress_threshold=args.stress_threshold,
        stress_base_probability=args.stress_base_probability,
        stress_event_probability=args.stress_event_probability,
        stress_event_bonus=args.stress_event_bonus,
        liquidity_shock_exposure_frac=args.liquidity_shock_exposure_frac,
        funding_shock_exposure_frac=args.funding_shock_exposure_frac,
        capital_shock_exposure_frac=args.capital_shock_exposure_frac,
        base_risk_weight_multiplier=args.base_risk_weight_multiplier,
        stress_risk_weight_inflation=args.stress_risk_weight_inflation,
        repayment_delay_strength=args.repayment_delay_strength,
        duration_mismatch_strength=args.duration_mismatch_strength,
        stress_default_cluster_sensitivity=args.stress_default_cluster_sensitivity,
        fatigue_decay=args.fatigue_decay,
        fatigue_growth=args.fatigue_growth,
        fatigue_cap=args.fatigue_cap,
        fatigue_default_sensitivity=args.fatigue_default_sensitivity,
        fatigue_margin_sensitivity=args.fatigue_margin_sensitivity,
        fatigue_demand_sensitivity=args.fatigue_demand_sensitivity,
        use_risk_adjusted_epr=args.use_risk_adjusted_epr,
        risk_adjusted_default_cost_weight=args.risk_adjusted_default_cost_weight,
        risk_adjusted_shock_cost_weight=args.risk_adjusted_shock_cost_weight,
        risk_adjusted_capital_charge_weight=args.risk_adjusted_capital_charge_weight,
        risk_adjusted_repayment_delay_weight=args.risk_adjusted_repayment_delay_weight,
        risk_adjusted_fatigue_cost_weight=args.risk_adjusted_fatigue_cost_weight,
        soft_risk_penalty_lambda=args.soft_risk_penalty_lambda,
        soft_risk_lcr_target=args.soft_risk_lcr_target,
        soft_risk_car_target=args.soft_risk_car_target,
    )


def effective_liquidity_penalty_lambda(args: argparse.Namespace) -> float:
    return 0.0 if bool(getattr(args, "primal_dual", False)) else float(args.liquidity_penalty_lambda)


def evaluate_all_policies(
    args: argparse.Namespace,
    calibration: SegmentCalibration,
    agent,
    seed: int,
    starting_cash: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    if getattr(args, "simulator_backend", SIMULATOR_BACKEND) == "torch":
        return evaluate_all_policies_torch(args, calibration, agent, seed, starting_cash=starting_cash)

    def env_factory(local_seed: int) -> SegmentLendingEnv:
        return build_env(args, calibration, seed=local_seed, starting_cash=starting_cash, liquidate_on_breach=True)

    policies = {
        "ppo_rl": ("actor", lambda obs: agent.act(obs, deterministic=True)),
        "reject_all": ("baseline", lambda env, obs: reject_all_policy(env)),
        "fixed_conservative": ("baseline", lambda env, obs: fixed_conservative_policy(env)),
        "fixed_aggressive": ("baseline", lambda env, obs: fixed_aggressive_policy(env)),
        "greedy_expected_profit": ("baseline", lambda env, obs: greedy_expected_profit_policy(env)),
        "budget_aware_heuristic": ("baseline", lambda env, obs: budget_aware_segment_policy(env)),
        "one_step_lp_closed_form": ("baseline", lambda env, obs: one_step_lp_policy(env)),
    }

    results = {}
    episode_frames = []
    for name, (kind, policy) in policies.items():
        if kind == "actor":
            summary, episodes = evaluate_segment_policy(
                env_factory,
                actor_fn=policy,
                episodes=args.eval_episodes,
                seed=seed,
            )
        else:
            summary, episodes = evaluate_segment_policy(
                env_factory,
                policy_fn=policy,
                episodes=args.eval_episodes,
                seed=seed,
            )
        results[name] = summary
        episodes.insert(0, "policy", name)
        episode_frames.append(episodes)
    annotate_policy_frontier_metrics(results, args)
    return results, pd.concat(episode_frames, ignore_index=True)


def evaluate_all_policies_torch(
    args: argparse.Namespace,
    calibration: SegmentCalibration,
    agent,
    seed: int,
    starting_cash: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    device = resolve_torch_device(args.device)
    config = build_torch_batch_config(args)
    policies = [
        "ppo_rl",
        "reject_all",
        "fixed_conservative",
        "fixed_aggressive",
        "greedy_expected_profit",
        "budget_aware_heuristic",
        "one_step_lp_closed_form",
    ]

    def make_env(local_seed: int) -> TorchSegmentBatchEnv:
        return TorchSegmentBatchEnv(
            [calibration for _ in range(args.eval_episodes)],
            config,
            device=device,
            starting_cash=[starting_cash for _ in range(args.eval_episodes)],
            scenario_names=[calibration.name for _ in range(args.eval_episodes)],
            seed=local_seed,
        )

    results = {}
    episode_frames = []
    for policy in policies:
        summary, episodes = evaluate_torch_policy_batch(
            make_env,
            policy=policy,
            agent=agent,
            episodes=args.eval_episodes,
            seed=seed,
        )
        results[policy] = summary
        episode_frames.append(episodes)
    annotate_policy_frontier_metrics(results, args)
    return results, pd.concat(episode_frames, ignore_index=True)


def annotate_policy_frontier_metrics(results: dict, args: argparse.Namespace) -> None:
    reject_profit = results.get("reject_all", {}).get("cumulative_profit_mean")
    target = float(getattr(args, "primal_dual_constraint_threshold", PRIMAL_DUAL_CONSTRAINT_THRESHOLD))
    for summary in results.values():
        episode_count = max(int(summary.get("episode_count", 0) or 0), 1)
        liquidated_count = int(summary.get("liquidated_episode_count", 0) or 0)
        liquidation_rate = liquidated_count / episode_count
        summary["liquidation_rate"] = float(liquidation_rate)
        if reject_profit is not None and np.isfinite(float(reject_profit)):
            summary["excess_profit_over_reject_mean"] = float(summary.get("cumulative_profit_mean", np.nan)) - float(
                reject_profit
            )
        else:
            summary["excess_profit_over_reject_mean"] = float("nan")
        summary["liquidation_constraint_target"] = target
        summary["liquidation_constraint_satisfied"] = float(liquidation_rate <= target)
        summary["liquidation_constrained_profit_mean"] = (
            float(summary.get("cumulative_profit_mean", np.nan)) if liquidation_rate <= target else float("nan")
        )


def evaluate_stress_suite(args: argparse.Namespace, test_calibration: SegmentCalibration, agent) -> dict[str, dict]:
    stress_calibrations = {
        "default_2x": test_calibration.stressed("default_2x", default_multiplier=2.0),
        "high_risk_demand_1p5x": test_calibration.stressed("high_risk_demand_1p5x", high_risk_demand_multiplier=1.5),
        "liquidity_0p7x": test_calibration.stressed("liquidity_0p7x"),
        "recovery_0p5x": test_calibration.stressed("recovery_0p5x", recovery_multiplier=0.5),
        "combined_stress": test_calibration.stressed(
            "combined_stress",
            default_multiplier=1.5,
            high_risk_demand_multiplier=1.5,
            recovery_multiplier=0.5,
        ),
    }
    results = {}
    for offset, (name, calibration) in enumerate(stress_calibrations.items()):
        starting_cash = args.initial_cash * (0.70 if name in {"liquidity_0p7x", "combined_stress"} else 1.0)
        summary, _ = evaluate_all_policies(
            args,
            calibration,
            agent,
            seed=args.seed + 50_000 + offset * 5_000,
            starting_cash=starting_cash,
        )
        results[name] = summary
    return results


def write_policy_results(results: dict, path: Path) -> None:
    rows = []
    for policy, summary in results.items():
        row = {"policy": policy}
        row.update(summary)
        rows.append(row)
    pd.DataFrame(rows).sort_values(
        ["liquidity_breach_mean", "cumulative_profit_mean"],
        ascending=[True, False],
    ).to_csv(path, index=False)


def compact_policy_summary(summary: dict) -> dict:
    keys = [
        "episode_count",
        "cumulative_profit_mean",
        "decision_period_profit_mean",
        "terminal_runoff_profit_mean",
        "terminal_value_mean",
        "liquidity_breach_mean",
        "liquidated_episode_count",
        "liquidity_liquidation_episode_count",
        "capital_liquidation_episode_count",
        "min_liquidity_coverage_ratio_mean",
        "min_capital_adequacy_ratio_mean",
        "expected_shortfall_mean",
        "liquidation_loss_mean",
        "cash_inflation_cost_mean",
        "expected_deployment_profit_mean",
        "risk_adjusted_deployment_profit_mean",
        "expected_profit_shaping_reward_mean",
        "excess_profit_over_reject_mean",
        "liquidation_rate",
        "liquidation_constrained_profit_mean",
        "cumulative_profit_cvar_5_mean",
        "cumulative_profit_worst_decile_mean",
        "stress_regime_profit_mean",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_or_build_scored(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, dict]:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"segment_scored_{cache_key(args)}.pkl"
    if cache_path.exists() and not args.refresh_cache:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        summary = dict(payload["summary"])
        summary["cache_hit"] = True
        return payload["scored"], summary, {"enabled": True, "hit": True, "path": str(cache_path)}

    raw = load_accepted_loans(
        args.accepted_path,
        max_rows=args.max_loans,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    scored, summary = prepare_scored_loans_temporal(
        raw,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    with cache_path.open("wb") as handle:
        pickle.dump({"scored": scored, "summary": summary}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = dict(summary)
    summary["cache_hit"] = False
    return scored, summary, {"enabled": True, "hit": False, "path": str(cache_path)}


def cache_key(args: argparse.Namespace) -> str:
    path = Path(args.accepted_path).expanduser().resolve()
    stat = path.stat()
    payload = {
        "version": "segment-ppo-v3_borrower_features_demand_scaling",
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "max_loans": args.max_loans,
        "sample_mode": args.sample_mode,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "risk_buckets": args.risk_buckets,
        "amount_tiers": args.amount_tiers,
        "term_buckets": args.term_buckets,
        "maturity_buckets": args.maturity_buckets,
        "seed": args.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()

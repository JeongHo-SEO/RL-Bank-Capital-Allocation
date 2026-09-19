from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from lending_rl_agent import load_accepted_loans, prepare_scored_loans_temporal
from lending_rl_agent.ppo import PPOConfig, train_ppo, train_ppo_torch_batch
from lending_rl_agent.segment_simulator import (
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
)

SAMPLE_MODE = "reservoir"
SOFT_RESERVE = True
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
MIN_EXPECTED_PROFIT_RATE = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO for segment-level liquidity-constrained lending allocation."
    )
    parser.add_argument("--accepted-path", default="../accepted_2007_to_2018Q4.csv")
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
    parser.add_argument("--scorer", choices=["lr", "xgb"], default="lr")
    parser.add_argument("--liquidity-penalty-lambda", type=float, default=1.0)
    parser.add_argument("--terminal-runoff-months", type=int, default=60)
    parser.add_argument(
        "--expected-inflation-annual",
        type=float,
        default=0.025,
        help="Annual expected inflation drag on idle cash, applied monthly in real-value terms.",
    )
    parser.add_argument("--recovery-rate", type=float, default=0.10)
    parser.add_argument("--funding-cost-annual", type=float, default=0.03)
    parser.add_argument("--servicing-cost-rate", type=float, default=0.005)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--policy-architecture", choices=["attention", "mlp", "gru"], default="attention")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--gru-dropout", type=float, default=0.0)
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
        default="expected",
        help="GPU simulator default model. expected is deterministic and matches segment expected-profit calibration.",
    )
    parser.add_argument(
        "--min-expected-profit-rate",
        type=float,
        default=MIN_EXPECTED_PROFIT_RATE,
        help="PPO-only default keeps every segment eligible; raise this to hard-mask low-EPR segments.",
    )
    parser.add_argument(
        "--negative-epr-penalty",
        type=float,
        default=0.0,
        help="Reward penalty for actor probability mass assigned to masked unprofitable segments.",
    )
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/segment_ppo_final")
    args = parser.parse_args()
    args.sample_mode = SAMPLE_MODE
    args.soft_reserve = SOFT_RESERVE
    args.terminal_runoff_discount = TERMINAL_RUNOFF_DISCOUNT
    args.terminal_value_weight = TERMINAL_VALUE_WEIGHT
    args.train_calibration_augmentation = TRAIN_CALIBRATION_AUGMENTATION
    args.min_capital_ratio = MIN_CAPITAL_RATIO
    args.liquidity_liquidation_haircut = LIQUIDITY_LIQUIDATION_HAIRCUT
    args.simulator_backend = SIMULATOR_BACKEND
    args.allocation_temperature = ALLOCATION_TEMPERATURE
    args.segment_epr_prior_scale = SEGMENT_EPR_PRIOR_SCALE
    args.expected_profit_shaping_weight = EXPECTED_PROFIT_SHAPING_WEIGHT
    args.allocation_top_k = ALLOCATION_TOP_K
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
        segment_parameter_count=4,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        attention_dropout=args.attention_dropout,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
    )

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

    if SIMULATOR_BACKEND == "torch":
        agent, history = train_ppo_torch_batch(
            train_batch_factory,
            observation_dim=torch_dim_env.observation_dim,
            action_dim=torch_dim_env.action_dim,
            config=config,
            progress_every=args.progress_every,
        )
    else:
        agent, history = train_ppo(
            train_env_factory,
            observation_dim=train_env.observation_space.shape[0],
            action_dim=train_env.action_space.shape[0],
            config=config,
            progress_every=args.progress_every,
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
            "min_expected_profit_rate": args.min_expected_profit_rate,
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
            "form": "S_t=(B_t,G_t,E_t,theta)",
            "dimension": int(train_env.observation_space.shape[0]),
            "segment_count": calibrations["train"].segment_count,
            "maturity_bucket_count": calibrations["train"].maturity_bucket_count,
            "segment_parameter_count": 4,
            "segment_parameter_names": list(train_env._reset_info()["segment_parameter_names"]),
            "policy_architecture": args.policy_architecture,
        },
        "action": {
            "form": "actor_raw -> deployment_fraction_and_segment_weights -> feasible_allocation",
            "raw_dimension": int(train_env.action_space.shape[0]),
            "allocation_dimension": calibrations["train"].segment_count,
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
            "allocation_temperature": ALLOCATION_TEMPERATURE,
            "segment_epr_prior_scale": SEGMENT_EPR_PRIOR_SCALE,
            "expected_profit_shaping_weight": EXPECTED_PROFIT_SHAPING_WEIGHT,
            "allocation_top_k": ALLOCATION_TOP_K,
            "description": "Training and evaluation both terminate an episode on capital-adequacy or liquidity breach. Liquidity liquidation subtracts a 60% haircut on current assets.",
            "terminal_value_weight": TERMINAL_VALUE_WEIGHT,
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
        liquidity_penalty_lambda=args.liquidity_penalty_lambda,
        seed=seed,
        starting_cash=starting_cash,
        enforce_min_cash_constraint=False,
        reserve_exposure_ratio=args.reserve_exposure_ratio,
        terminal_runoff_months=args.terminal_runoff_months,
        terminal_runoff_discount=TERMINAL_RUNOFF_DISCOUNT,
        terminal_value_weight=TERMINAL_VALUE_WEIGHT,
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
    )


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
    return results, pd.concat(episode_frames, ignore_index=True)


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
        "expected_profit_shaping_reward_mean",
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
        scorer_type=args.scorer,
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
        "scorer": args.scorer,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()

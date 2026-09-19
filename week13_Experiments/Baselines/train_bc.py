"""Behavioral Cloning training script for segment-level lending allocation.

Collects (state, action) pairs by running a heuristic policy in the simulator,
trains MLP or XGBoost via supervised learning, then evaluates the BC model
against baseline policies using the same simulator setup as train_segment_ppo.py.

Usage:
    python train_bc.py --bc-model mlp --scorer lr --output-dir outputs/BC_MLP_LR
    python train_bc.py --bc-model xgb --scorer lr --output-dir outputs/BC_XGB_LR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lending_rl_agent import load_accepted_loans, prepare_scored_loans_temporal
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
from lending_rl_agent.bc import MLPBehavioralCloning, XGBBehavioralCloning

# ---------------------------------------------------------------------------
# Constants (동일하게 유지 - train_segment_ppo.py 기준)
# ---------------------------------------------------------------------------
SAMPLE_MODE = "reservoir"
TERMINAL_RUNOFF_DISCOUNT = 0.99
TERMINAL_VALUE_WEIGHT = 0.0
MIN_CAPITAL_RATIO = 0.08
LIQUIDITY_LIQUIDATION_HAIRCUT = 0.60


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning for segment-level lending allocation."
    )
    # 데이터
    parser.add_argument("--accepted-path", default="../accepted_2007_to_2018Q4.csv")
    parser.add_argument("--max-loans", type=int, default=120_000)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--scorer", choices=["lr", "xgb"], default="lr")
    parser.add_argument("--seed", type=int, default=42)

    # Segment calibration
    parser.add_argument("--risk-buckets", type=int, default=5)
    parser.add_argument("--amount-tiers", type=int, default=3)
    parser.add_argument("--term-buckets", type=int, default=2)
    parser.add_argument("--maturity-buckets", type=int, default=6)
    parser.add_argument("--recovery-rate", type=float, default=0.10)
    parser.add_argument("--funding-cost-annual", type=float, default=0.03)
    parser.add_argument("--servicing-cost-rate", type=float, default=0.005)

    # Simulator
    parser.add_argument("--initial-cash", type=float, default=50_000_000.0)
    parser.add_argument("--min-cash-ratio", type=float, default=0.20)
    parser.add_argument("--horizon-months", type=int, default=36)
    parser.add_argument("--reserve-exposure-ratio", type=float, default=0.05)
    parser.add_argument("--liquidity-penalty-lambda", type=float, default=1.0)
    parser.add_argument("--terminal-runoff-months", type=int, default=60)
    parser.add_argument("--expected-inflation-annual", type=float, default=0.025)

    # BC 설정
    parser.add_argument("--bc-model", choices=["mlp", "xgb"], default="mlp")
    parser.add_argument("--collect-episodes", type=int, default=800,
                        help="Number of simulator episodes to collect (state, action) pairs from.")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--bc-batch-size", type=int, default=1024)
    parser.add_argument("--bc-lr", type=float, default=1e-3)

    # 평가
    parser.add_argument("--eval-episodes", type=int, default=100)

    # 출력
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output-dir", default="outputs/BC_MLP_LR")

    args = parser.parse_args()
    args.sample_mode = SAMPLE_MODE
    args.terminal_runoff_discount = TERMINAL_RUNOFF_DISCOUNT
    args.terminal_value_weight = TERMINAL_VALUE_WEIGHT
    args.min_capital_ratio = MIN_CAPITAL_RATIO
    args.liquidity_liquidation_haircut = LIQUIDITY_LIQUIDATION_HAIRCUT
    return args


# ---------------------------------------------------------------------------
# 데이터 수집: heuristic policy로 simulator 돌리며 (state, action) 쌍 수집
# ---------------------------------------------------------------------------

def collect_bc_data(
    env_factory,
    episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """budget_aware_heuristic으로 simulator를 돌려 (state, action) 쌍 수집."""
    states = []
    actions = []
    print(f"Collecting BC data: {episodes} episodes...")
    for ep in range(episodes):
        env = env_factory(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        while True:
            # heuristic이 결정한 allocation을 action label로 사용
            allocation = budget_aware_segment_policy(env)
            states.append(obs.copy())
            # allocation을 PPO raw action 형식으로 변환
            # deploy_fraction=1.0 → raw[0]=large positive, segment weights=allocation
            actions.append(allocation.astype(np.float32))

            obs, _, terminated, truncated, _ = env.step_allocation(allocation)
            if terminated or truncated:
                break

        if (ep + 1) % 100 == 0:
            print(f"  collected {ep+1}/{episodes} episodes, samples so far: {len(states):,}")

    print(f"Total collected: {len(states):,} (state, action) pairs")
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)


# ---------------------------------------------------------------------------
# Simulator 환경 구성 (train_segment_ppo.py와 동일)
# ---------------------------------------------------------------------------

def build_env(args: argparse.Namespace, calibration: SegmentCalibration, seed: int,
              starting_cash: float | None = None) -> SegmentLendingEnv:
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
        liquidate_on_breach=True,
        min_capital_ratio=MIN_CAPITAL_RATIO,
        liquidity_liquidation_haircut=LIQUIDITY_LIQUIDATION_HAIRCUT,
        expected_inflation_annual=args.expected_inflation_annual,
    )


def build_calibrations(args: argparse.Namespace, scored: pd.DataFrame) -> dict[str, SegmentCalibration]:
    train = SegmentCalibration.from_scored(
        scored, split="train",
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
                scored, split=split,
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
    return calibrations


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def evaluate_all_policies(
    args: argparse.Namespace,
    calibration: SegmentCalibration,
    bc_actor_fn,
    seed: int,
) -> tuple[dict, pd.DataFrame]:

    def env_factory(local_seed: int) -> SegmentLendingEnv:
        return build_env(args, calibration, seed=local_seed)

    policies = {
        "bc_model": ("baseline", lambda env, obs: bc_actor_fn(obs)),
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
        print(f"  Evaluating: {name}")
        if kind == "actor":
            summary, episodes = evaluate_segment_policy(
                env_factory, actor_fn=policy,
                episodes=args.eval_episodes, seed=seed,
            )
        else:
            summary, episodes = evaluate_segment_policy(
                env_factory, policy_fn=policy,
                episodes=args.eval_episodes, seed=seed,
            )
        results[name] = summary
        episodes.insert(0, "policy", name)
        episode_frames.append(episodes)

    return results, pd.concat(episode_frames, ignore_index=True)


def write_policy_results(results: dict, path: Path) -> None:
    rows = [{"policy": policy, **summary} for policy, summary in results.items()]
    pd.DataFrame(rows).sort_values(
        ["liquidated_episode_count", "cumulative_profit_mean"],
        ascending=[True, False],
    ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------

def load_or_build_scored(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.accepted_path).expanduser().resolve()
    stat = path.stat()
    payload = {
        "version": "segment-ppo-v3_borrower_features_demand_scaling",
        "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "max_loans": args.max_loans, "sample_mode": args.sample_mode,
        "train_ratio": args.train_ratio, "validation_ratio": args.validation_ratio,
        "risk_buckets": args.risk_buckets, "amount_tiers": args.amount_tiers,
        "term_buckets": args.term_buckets, "maturity_buckets": args.maturity_buckets,
        "seed": args.seed,
        "scorer": args.scorer,

    }
    cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"segment_scored_{cache_key}.pkl"

    if cache_path.exists() and not args.refresh_cache:
        print(f"Cache hit: {cache_path}")
        with cache_path.open("rb") as f:
            p = pickle.load(f)
        return p["scored"], dict(p["summary"])

    print("Building scored loans...")
    raw = load_accepted_loans(args.accepted_path, max_rows=args.max_loans,
                              sample_mode=args.sample_mode, seed=args.seed)
    scored, summary = prepare_scored_loans_temporal(
        raw, train_ratio=args.train_ratio, validation_ratio=args.validation_ratio,
        seed=args.seed, scorer_type=args.scorer,
    )
    with cache_path.open("wb") as f:
        pickle.dump({"scored": scored, "summary": summary}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return scored, dict(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로딩 및 calibration
    scored, scorer_summary = load_or_build_scored(args)
    calibrations = build_calibrations(args, scored)

    train_calibration = calibrations["train"]
    obs_dim = (
        1
        + train_calibration.segment_count
        + train_calibration.segment_count * train_calibration.maturity_bucket_count
        + train_calibration.segment_count * 4
    )
    action_dim = train_calibration.segment_count

    print(f"\n=== Behavioral Cloning: {args.bc_model.upper()} ===")
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")

    # 2. (state, action) 수집
    def train_env_factory(local_seed: int) -> SegmentLendingEnv:
        return build_env(args, train_calibration, seed=local_seed)

    states, actions = collect_bc_data(
        train_env_factory,
        episodes=args.collect_episodes,
        seed=args.seed,
    )

    # 3. BC 모델 학습
    print(f"\nTraining {args.bc_model.upper()} BC model...")
    if args.bc_model == "mlp":
        bc = MLPBehavioralCloning(
            observation_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=args.hidden_size,
            learning_rate=args.bc_lr,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            device="cpu",
        )
        losses = bc.fit(states, actions)
        # 학습 loss 저장
        pd.DataFrame({"epoch": range(1, len(losses)+1), "loss": losses}).to_csv(
            output_dir / "bc_training_loss.csv", index=False
        )
    else:
        bc = XGBBehavioralCloning(seed=args.seed)
        bc.fit(states, actions)

    # 4. 평가
    print("\n=== Validation Evaluation ===")
    val_results, val_episodes = evaluate_all_policies(
        args, calibrations["validation"], bc.act, seed=args.seed + 10_000
    )

    print("\n=== Test Evaluation ===")
    test_results, test_episodes = evaluate_all_policies(
        args, calibrations["test"], bc.act, seed=args.seed + 20_000
    )

    # 5. 결과 저장
    write_policy_results(val_results, output_dir / "validation_policy_results.csv")
    write_policy_results(test_results, output_dir / "test_policy_results.csv")
    val_episodes.to_csv(output_dir / "validation_policy_episodes.csv", index=False)
    test_episodes.to_csv(output_dir / "test_policy_episodes.csv", index=False)

    metrics = {
        "bc_model": args.bc_model,
        "scorer": args.scorer,
        "collect_episodes": args.collect_episodes,
        "collected_samples": len(states),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "teacher_policy": "budget_aware_heuristic",
        "scorer_summary": scorer_summary,
        "validation": val_results,
        "test": test_results,
        "args": vars(args),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 핵심 결과 출력
    bc_test = test_results.get("bc_model", {})
    print("\n=== Result Summary ===")
    print(json.dumps({
        "bc_model": args.bc_model,
        "scorer": args.scorer,
        "test_cumulative_profit_mean": bc_test.get("cumulative_profit_mean"),
        "test_liquidated_episode_count": bc_test.get("liquidated_episode_count"),
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()

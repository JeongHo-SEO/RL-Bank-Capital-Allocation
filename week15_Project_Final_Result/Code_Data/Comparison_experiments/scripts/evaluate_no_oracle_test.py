from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lending_rl_agent.ppo import PPOAgent, PPOConfig
from train_segment_ppo import build_calibrations, evaluate_all_policies, load_or_build_scored, parse_args as train_parse_args


def parse_local_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy on test demand with train-calibrated segment economics."
    )
    parser.add_argument("--checkpoint-dir", default="outputs/segment_ppo_800_lambda1_inflation25_eprfix_sharp_v2")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def main() -> None:
    local = parse_local_args()
    checkpoint_dir = Path(local.checkpoint_dir)
    output_dir = Path(local.output_dir) if local.output_dir else checkpoint_dir / "no_oracle_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        args = train_parse_args()
    finally:
        sys.argv = original_argv
    args.output_dir = str(output_dir)
    args.eval_episodes = int(local.eval_episodes)
    args.device = local.device

    ckpt = torch.load(checkpoint_dir / "segment_ppo.pt", map_location=args.device)
    config = PPOConfig(**ckpt["config"])
    config.device = args.device
    agent = PPOAgent(ckpt["observation_dim"], ckpt["action_dim"], config)
    agent.model.load_state_dict(ckpt["state_dict"])
    agent.model.eval()

    scored, _, _ = load_or_build_scored(args)
    calibrations = build_calibrations(args, scored)
    train = calibrations["train"]
    test = calibrations["test"]
    hybrid = replace(
        train,
        name="test_demand_train_theta",
        monthly_demand=test.monthly_demand,
        monthly_counts=test.monthly_counts,
        metadata={
            **train.metadata,
            "scenario": "test_demand_train_theta_no_oracle",
            "demand_source": "test",
            "segment_parameter_source": "train",
            "oracle_test_outcomes_used_for_segment_theta": False,
        },
    )

    results, episodes = evaluate_all_policies(args, hybrid, agent, seed=args.seed + 90_000)
    pd.DataFrame(
        [{"policy": policy, **summary} for policy, summary in results.items()]
    ).to_csv(output_dir / "no_oracle_test_policy_results.csv", index=False)
    episodes.to_csv(output_dir / "no_oracle_test_policy_episodes.csv", index=False)

    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "scenario": "test monthly demand with train-calibrated segment parameters",
        "results": results,
    }
    (output_dir / "no_oracle_test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ppo = results["ppo_rl"]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "ppo_cumulative_profit_mean": ppo.get("cumulative_profit_mean"),
                "ppo_liquidated_episode_count": ppo.get("liquidated_episode_count"),
                "ppo_liquidity_breach_mean": ppo.get("liquidity_breach_mean"),
                "ppo_min_lcr": ppo.get("min_liquidity_coverage_ratio_mean"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lending_rl_agent.ppo import PPOAgent, PPOConfig
from train_segment_ppo import (
    build_calibrations,
    evaluate_all_policies,
    evaluate_stress_suite,
    write_policy_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing lambda-sweep PPO checkpoints without retraining."
    )
    parser.add_argument("--input-root", default="outputs/lambda_sweep_400")
    parser.add_argument("--output-root", default="outputs/lambda_sweep_400_liquidation_eval")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    input_root = (repo_root / args.input_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path for path in input_root.glob("lam_*") if (path / "metrics.json").exists())
    if not run_dirs:
        raise SystemExit(f"No lambda run metrics found under {input_root}")

    for run_dir in run_dirs:
        evaluate_run(run_dir, output_root / run_dir.name, args)

    if not args.skip_plot:
        plot_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "plot_lambda_frontier.py"),
            "--metrics-glob",
            str(output_root / "lam_*" / "metrics.json"),
            "--output-dir",
            str(output_root / "figures"),
        ]
        subprocess.run(plot_cmd, cwd=repo_root, check=True)
        print(f"Re-evaluation figures written to {output_root / 'figures'}")


def evaluate_run(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_metrics = output_dir / "metrics.json"
    if output_metrics.exists() and not args.force:
        print(f"Skipping {input_dir.name}; existing evaluation at {output_metrics}")
        return

    source_metrics = json.loads((input_dir / "metrics.json").read_text(encoding="utf-8"))
    run_args = Namespace(**source_metrics["training"])
    if args.eval_episodes is not None:
        run_args.eval_episodes = int(args.eval_episodes)
    if args.seed is not None:
        run_args.seed = int(args.seed)

    scored = load_scored_from_cache(source_metrics)
    calibrations = build_calibrations(run_args, scored)
    agent = load_agent(input_dir / "segment_ppo.pt")

    print(
        f"Evaluating {input_dir.name}: lambda={run_args.liquidity_penalty_lambda:g}, "
        f"eval_episodes={run_args.eval_episodes}"
    )
    validation_results, validation_episodes = evaluate_all_policies(
        run_args, calibrations["validation"], agent, run_args.seed + 10_000
    )
    test_results, test_episodes = evaluate_all_policies(run_args, calibrations["test"], agent, run_args.seed + 20_000)
    stress_results = evaluate_stress_suite(run_args, calibrations["test"], agent)

    write_policy_results(validation_results, output_dir / "validation_policy_results.csv")
    write_policy_results(test_results, output_dir / "test_policy_results.csv")
    validation_episodes.to_csv(output_dir / "validation_policy_episodes.csv", index=False)
    test_episodes.to_csv(output_dir / "test_policy_episodes.csv", index=False)
    for name, result in stress_results.items():
        write_policy_results(result, output_dir / f"stress_{name}_policy_results.csv")

    if (input_dir / "segment_ppo.pt").exists():
        shutil.copy2(input_dir / "segment_ppo.pt", output_dir / "segment_ppo.pt")
    if (input_dir / "segment_ppo.json").exists():
        shutil.copy2(input_dir / "segment_ppo.json", output_dir / "segment_ppo.json")
    if (input_dir / "ppo_training_history.csv").exists():
        shutil.copy2(input_dir / "ppo_training_history.csv", output_dir / "ppo_training_history.csv")

    metrics = dict(source_metrics)
    metrics["evaluation_rule"] = {
        "liquidate_on_breach": True,
        "description": "Existing checkpoint re-evaluated only. Capital adequacy below 8% triggers capital liquidation. Liquidity shortfall triggers liquidity liquidation with a 60% haircut on current assets (cash budget plus loan-book exposure). Liquidation terminates the episode and subtracts liquidation loss from cumulative profit.",
        "source_run": str(input_dir),
    }
    metrics["training"] = vars(run_args)
    metrics["validation"] = validation_results
    metrics["test"] = test_results
    metrics["stress"] = stress_results
    output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_scored_from_cache(metrics: dict) -> pd.DataFrame:
    cache_path = Path(metrics["cache"]["path"])
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["scored"]


def load_agent(checkpoint_path: Path) -> PPOAgent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = PPOConfig(**checkpoint["config"])
    agent = PPOAgent(
        observation_dim=int(checkpoint["observation_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        config=config,
    )
    agent.model.load_state_dict(checkpoint["state_dict"])
    agent.model.eval()
    return agent


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_LAMBDAS = (0.0, 0.1, 1.0, 10.0, 100.0)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run PPO trainings across liquidity penalty lambdas and plot the frontier."
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--output-root", default="outputs/lambda_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Rerun a lambda even if metrics.json already exists.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training and only rebuild frontier figures from existing metrics.",
    )
    parser.add_argument(
        "--extra-metrics",
        nargs="*",
        default=[],
        help="Optional existing metrics.json files to include in the final plot.",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to train_segment_ppo.py after '--'.",
    )
    args = parser.parse_args()
    extra_training_args = list(args.training_args)
    if extra_training_args and extra_training_args[0] == "--":
        extra_training_args = extra_training_args[1:]
    return args, extra_training_args


def main() -> None:
    args, extra_training_args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        for lambda_value in args.lambdas:
            run_lambda(repo_root, output_root, lambda_value, args, extra_training_args)

    plot_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "plot_lambda_frontier.py"),
        "--metrics-glob",
        str(output_root / "lam_*" / "metrics.json"),
        "--output-dir",
        str(output_root / "figures"),
    ]
    if args.extra_metrics:
        plot_cmd.extend(["--metrics", *args.extra_metrics])
    print("Plotting lambda frontier...")
    subprocess.run(plot_cmd, cwd=repo_root, check=True)
    print(f"Done. Figures written to {output_root / 'figures'}")


def run_lambda(
    repo_root: Path,
    output_root: Path,
    lambda_value: float,
    args: argparse.Namespace,
    extra_training_args: list[str],
) -> None:
    output_dir = output_root / f"lam_{lambda_tag(lambda_value)}"
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"Skipping lambda={lambda_value:g}; metrics already exist at {metrics_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    cmd = [
        sys.executable,
        str(repo_root / "train_segment_ppo.py"),
        "--liquidity-penalty-lambda",
        str(lambda_value),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
        *extra_training_args,
    ]
    print(f"Running lambda={lambda_value:g}; log={log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, check=True)


def lambda_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()

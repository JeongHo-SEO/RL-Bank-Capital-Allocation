from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO lambda sweep profit-liquidity frontier.")
    parser.add_argument(
        "--metrics-glob",
        default="outputs/lambda_sweep/lam_*/metrics.json",
        help="Glob pattern for lambda-run metrics.json files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[],
        help="Optional explicit metrics.json paths to include in addition to the glob.",
    )
    parser.add_argument("--output-dir", default="outputs/lambda_sweep/figures")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted({Path(path) for path in glob.glob(args.metrics_glob)} | {Path(path) for path in args.metrics})
    if not paths:
        raise SystemExit(f"No metrics files found for pattern: {args.metrics_glob}")

    frames = []
    for path in paths:
        frames.append(load_metrics(path, args.splits))
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output_dir / "lambda_frontier_all_metrics.csv", index=False)

    for split in args.splits:
        split_data = data[data["split"] == split].copy()
        if split_data.empty:
            continue
        split_data.to_csv(output_dir / f"lambda_frontier_{split}.csv", index=False)
        plot_split(
            split_data,
            split,
            output_dir,
            x_column="liquidity_breach_mean",
            x_label="Liquidity breach probability",
            suffix="",
            x_scale=1.0,
            xlim=(-0.03, 1.03),
        )
        plot_split(
            split_data,
            split,
            output_dir,
            x_column="expected_shortfall_mean",
            x_label="Expected shortfall (USD millions)",
            suffix="_shortfall",
            x_scale=1_000_000.0,
            xlim=None,
        )


def load_metrics(path: Path, splits: list[str]) -> pd.DataFrame:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    training = metrics.get("training", {})
    lambda_value = float(training.get("liquidity_penalty_lambda", float("nan")))
    rows = []
    for split in splits:
        for policy, summary in metrics.get(split, {}).items():
            rows.append(
                {
                    "split": split,
                    "lambda": lambda_value,
                    "policy": policy,
                    "source": str(path.parent),
                    "episodes": training.get("episodes"),
                    "eval_episodes": training.get("eval_episodes"),
                    "episode_count": summary.get("episode_count"),
                    "cumulative_profit_mean": summary.get("cumulative_profit_mean"),
                    "cumulative_profit_std": summary.get("cumulative_profit_std"),
                    "liquidity_breach_mean": summary.get("liquidity_breach_mean"),
                    "liquidity_breach_std": summary.get("liquidity_breach_std"),
                    "liquidity_breach_episode_count": summary.get("liquidity_breach_episode_count"),
                    "min_liquidity_coverage_ratio_mean": summary.get("min_liquidity_coverage_ratio_mean"),
                    "expected_shortfall_mean": summary.get("expected_shortfall_mean"),
                    "decision_period_profit_mean": summary.get("decision_period_profit_mean"),
                    "terminal_runoff_profit_mean": summary.get("terminal_runoff_profit_mean"),
                    "terminal_value_mean": summary.get("terminal_value_mean"),
                    "default_loss_mean": summary.get("default_loss_mean"),
                    "funding_cost_mean": summary.get("funding_cost_mean"),
                    "servicing_cost_mean": summary.get("servicing_cost_mean"),
                    "liquidated_mean": summary.get("liquidated_mean"),
                    "liquidated_episode_count": summary.get("liquidated_episode_count"),
                    "liquidation_asset_base_mean": summary.get("liquidation_asset_base_mean"),
                    "liquidation_loss_mean": summary.get("liquidation_loss_mean"),
                    "capital_breach_mean": summary.get("capital_breach_mean"),
                    "capital_breach_episode_count": summary.get("capital_breach_episode_count"),
                    "capital_liquidation_episode_count": summary.get("capital_liquidation_episode_count"),
                    "liquidity_liquidation_episode_count": summary.get("liquidity_liquidation_episode_count"),
                    "capital_shortfall_mean": summary.get("capital_shortfall_mean"),
                    "min_capital_adequacy_ratio_mean": summary.get("min_capital_adequacy_ratio_mean"),
                    "terminal_capital_mean": summary.get("terminal_capital_mean"),
                    "liquidation_proceeds_mean": summary.get("liquidation_proceeds_mean"),
                    "disbursement_mean": summary.get("disbursement_mean"),
                }
            )
    return pd.DataFrame(rows)


def plot_split(
    data: pd.DataFrame,
    split: str,
    output_dir: Path,
    x_column: str,
    x_label: str,
    suffix: str,
    x_scale: float,
    xlim: tuple[float, float] | None,
) -> None:
    ppo = data[data["policy"] == "ppo_rl"].sort_values("lambda")
    baselines = data[data["policy"] != "ppo_rl"].sort_values("lambda").drop_duplicates("policy")

    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=160)
    if not ppo.empty:
        ax.plot(
            ppo[x_column] / x_scale,
            ppo["cumulative_profit_mean"] / 1_000_000.0,
            color="#2563eb",
            marker="o",
            linewidth=2.5,
            label="PPO lambda sweep",
        )
        for _, row in ppo.iterrows():
            offset = label_offset(int(row.name))
            ax.annotate(
                f"λ={row['lambda']:g}",
                (row[x_column] / x_scale, row["cumulative_profit_mean"] / 1_000_000.0),
                textcoords="offset points",
                xytext=offset,
                fontsize=8,
                color="#1d4ed8",
            )

    if not baselines.empty:
        ax.scatter(
            baselines[x_column] / x_scale,
            baselines["cumulative_profit_mean"] / 1_000_000.0,
            color="#525252",
            marker="x",
            s=58,
            linewidths=1.8,
            label="Baselines",
        )
        for _, row in baselines.iterrows():
            ax.annotate(
                short_policy_name(str(row["policy"])),
                (row[x_column] / x_scale, row["cumulative_profit_mean"] / 1_000_000.0),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=7,
                color="#404040",
            )

    ax.set_title(f"{split.title()} Profit-Liquidity Frontier")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative profit (USD millions)")
    ax.grid(True, alpha=0.28)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"lambda_frontier_{split}{suffix}.png")
    fig.savefig(output_dir / f"lambda_frontier_{split}{suffix}.pdf")
    plt.close(fig)


def label_offset(index: int) -> tuple[int, int]:
    offsets = [(6, 7), (6, -14), (6, 20), (6, -27), (6, 33)]
    return offsets[index % len(offsets)]


def short_policy_name(policy: str) -> str:
    replacements = {
        "reject_all": "reject",
        "fixed_conservative": "conservative",
        "fixed_aggressive": "aggressive",
        "greedy_expected_profit": "greedy",
        "budget_aware_heuristic": "budget-aware",
        "one_step_lp_closed_form": "one-step LP",
    }
    return replacements.get(policy, policy)


if __name__ == "__main__":
    main()

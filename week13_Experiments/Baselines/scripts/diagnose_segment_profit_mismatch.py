from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lending_rl_agent.segment_simulator import SegmentCalibration
from train_segment_ppo import SAMPLE_MODE, build_calibrations, load_or_build_scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare segment expected_profit_rate with simulator-implied runoff profit."
    )
    parser.add_argument("--accepted-path", default="archive 2/accepted_2007_to_2018Q4.csv.gz")
    parser.add_argument("--max-loans", type=int, default=120_000)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--risk-buckets", type=int, default=5)
    parser.add_argument("--amount-tiers", type=int, default=3)
    parser.add_argument("--term-buckets", type=int, default=2)
    parser.add_argument("--maturity-buckets", type=int, default=6)
    parser.add_argument("--recovery-rate", type=float, default=0.10)
    parser.add_argument("--funding-cost-annual", type=float, default=0.03)
    parser.add_argument("--servicing-cost-rate", type=float, default=0.005)
    parser.add_argument("--terminal-runoff-months", type=int, default=120)
    parser.add_argument("--terminal-runoff-discount", type=float, default=0.99)
    parser.add_argument("--exposure", type=float, default=1_000_000.0)
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/segment_profit_diagnostics")
    return parser.parse_args()


def runoff_profit_rates(
    calibration: SegmentCalibration,
    exposure: float,
    months: int,
    discount: float,
) -> pd.DataFrame:
    segment_count = calibration.segment_count
    maturity_bucket_count = calibration.maturity_bucket_count
    bucket_months = max(1.0, float(np.nanmax(calibration.avg_term_months)) / maturity_bucket_count)
    remaining_months = (np.arange(maturity_bucket_count, dtype=np.float64) + 0.5) * bucket_months
    principal_rate = np.clip(1.0 / np.maximum(remaining_months, 1.0), 0.0, 1.0)
    aging_rate = min(1.0, 1.0 / bucket_months)

    book = np.zeros((segment_count, maturity_bucket_count), dtype=np.float64)
    book[np.arange(segment_count), calibration.new_loan_bucket.astype(np.int64)] = float(exposure)

    nominal_profit = np.zeros(segment_count, dtype=np.float64)
    discounted_profit = np.zeros(segment_count, dtype=np.float64)
    interest_income = np.zeros(segment_count, dtype=np.float64)
    default_loss = np.zeros(segment_count, dtype=np.float64)
    funding_cost = np.zeros(segment_count, dtype=np.float64)
    servicing_cost = np.zeros(segment_count, dtype=np.float64)
    principal_paid_total = np.zeros(segment_count, dtype=np.float64)
    final_exposure = np.zeros(segment_count, dtype=np.float64)

    hazard = calibration.monthly_default_hazard[:, None]
    annual_rate = calibration.annual_interest_rate[:, None]
    recovery_rate = calibration.recovery_rate[:, None]

    discount_factor = 1.0
    months_to_runoff = np.zeros(segment_count, dtype=np.int64)
    for month in range(int(months)):
        active_before = book.sum(axis=1) > 1e-6
        if not np.any(active_before):
            break

        defaults = book * np.clip(hazard, 0.0, 0.95)
        surviving = np.maximum(book - defaults, 0.0)
        principal_paid = surviving * principal_rate[None, :]
        post_principal = np.maximum(surviving - principal_paid, 0.0)

        interest = book * (annual_rate / 12.0)
        recovery = defaults * recovery_rate
        loss = defaults - recovery
        funding = book * (calibration.funding_cost_annual / 12.0)
        servicing = book * (calibration.servicing_cost_rate / 12.0)

        period_interest = interest.sum(axis=1)
        period_loss = loss.sum(axis=1)
        period_funding = funding.sum(axis=1)
        period_servicing = servicing.sum(axis=1)
        period_profit = period_interest - period_loss - period_funding - period_servicing

        nominal_profit += period_profit
        discounted_profit += discount_factor * period_profit
        interest_income += period_interest
        default_loss += period_loss
        funding_cost += period_funding
        servicing_cost += period_servicing

        next_book = np.zeros_like(book)
        matured_principal = post_principal[:, 0] * aging_rate
        next_book[:, 0] += post_principal[:, 0] * (1.0 - aging_rate)
        for h in range(1, maturity_bucket_count):
            stay = post_principal[:, h] * (1.0 - aging_rate)
            advance = post_principal[:, h] * aging_rate
            next_book[:, h] += stay
            next_book[:, h - 1] += advance
        principal_paid_total += principal_paid.sum(axis=1) + matured_principal

        book = next_book
        active_after = book.sum(axis=1) > 1e-6
        months_to_runoff[(months_to_runoff == 0) & active_before & ~active_after] = month + 1
        discount_factor *= float(discount)

    final_exposure[:] = book.sum(axis=1)
    months_to_runoff[(months_to_runoff == 0) & (final_exposure <= 1e-6)] = int(months)

    rows = []
    for segment in range(segment_count):
        risk_bucket, amount_tier, term_bucket = calibration.decode_segment(segment)
        expected = float(calibration.expected_profit_rate[segment])
        actual = float(nominal_profit[segment] / exposure)
        discounted = float(discounted_profit[segment] / exposure)
        rows.append(
            {
                "split": calibration.name,
                "segment": segment,
                "label": calibration.segment_labels[segment],
                "risk_bucket": risk_bucket,
                "amount_tier": amount_tier,
                "term_bucket": term_bucket,
                "loan_pd": float(calibration.loan_pd[segment]),
                "monthly_default_hazard": float(calibration.monthly_default_hazard[segment]),
                "annual_interest_rate": float(calibration.annual_interest_rate[segment]),
                "avg_term_months": float(calibration.avg_term_months[segment]),
                "avg_loan_amount": float(calibration.avg_loan_amount[segment]),
                "new_loan_bucket": int(calibration.new_loan_bucket[segment]),
                "expected_profit_rate": expected,
                "actual_runoff_profit_rate": actual,
                "discounted_runoff_profit_rate": discounted,
                "mismatch": actual - expected,
                "sign_mismatch": bool(np.sign(expected) != np.sign(actual) and abs(expected) > 1e-9 and abs(actual) > 1e-9),
                "expected_positive_actual_negative": bool(expected > 0.0 and actual <= 0.0),
                "expected_negative_actual_positive": bool(expected <= 0.0 and actual > 0.0),
                "interest_rate": float(interest_income[segment] / exposure),
                "default_loss_rate": float(default_loss[segment] / exposure),
                "funding_cost_rate": float(funding_cost[segment] / exposure),
                "servicing_cost_rate_actual": float(servicing_cost[segment] / exposure),
                "principal_paid_rate": float(principal_paid_total[segment] / exposure),
                "final_exposure_rate": float(final_exposure[segment] / exposure),
                "months_to_runoff": int(months_to_runoff[segment]),
            }
        )
    return pd.DataFrame(rows)


def plot_split(frame: pd.DataFrame, output_dir: Path, split: str) -> None:
    split_frame = frame[frame["split"] == split].copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = np.where(split_frame["expected_positive_actual_negative"], "#dc2626", "#2563eb")
    ax.scatter(
        split_frame["expected_profit_rate"],
        split_frame["actual_runoff_profit_rate"],
        c=colors,
        s=70,
        alpha=0.85,
    )
    lower = float(min(split_frame["expected_profit_rate"].min(), split_frame["actual_runoff_profit_rate"].min()))
    upper = float(max(split_frame["expected_profit_rate"].max(), split_frame["actual_runoff_profit_rate"].max()))
    pad = max(0.01, (upper - lower) * 0.08)
    ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], color="black", linewidth=1.0, alpha=0.6)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Calibration expected_profit_rate")
    ax.set_ylabel("Simulator actual runoff profit rate")
    ax.set_title(f"{split}: expected vs actual segment profitability")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{split}_expected_vs_actual.png", dpi=180)
    plt.close(fig)

    by_risk = (
        split_frame.groupby("risk_bucket", as_index=False)
        .agg(
            expected_profit_rate=("expected_profit_rate", "mean"),
            actual_runoff_profit_rate=("actual_runoff_profit_rate", "mean"),
            segment_count=("segment", "count"),
        )
        .sort_values("risk_bucket")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.36
    x = np.arange(len(by_risk))
    ax.bar(x - width / 2, by_risk["expected_profit_rate"], width, label="expected")
    ax.bar(x + width / 2, by_risk["actual_runoff_profit_rate"], width, label="actual")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(by_risk["risk_bucket"].astype(str))
    ax.set_xlabel("Risk bucket")
    ax.set_ylabel("Profit rate")
    ax.set_title(f"{split}: average profitability by risk bucket")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{split}_profit_by_risk_bucket.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_args = SimpleNamespace(**vars(args))
    loader_args.sample_mode = SAMPLE_MODE
    scored, _, _ = load_or_build_scored(loader_args)
    calibrations = build_calibrations(loader_args, scored)

    frames = []
    for split in ["train", "validation", "test"]:
        frames.append(
            runoff_profit_rates(
                calibrations[split],
                exposure=args.exposure,
                months=args.terminal_runoff_months,
                discount=args.terminal_runoff_discount,
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(output_dir / "segment_expected_vs_actual.csv", index=False)

    summary = (
        frame.groupby("split", as_index=False)
        .agg(
            corr=("expected_profit_rate", lambda s: float(np.corrcoef(s, frame.loc[s.index, "actual_runoff_profit_rate"])[0, 1])),
            expected_positive_actual_negative=("expected_positive_actual_negative", "sum"),
            expected_negative_actual_positive=("expected_negative_actual_positive", "sum"),
            sign_mismatch=("sign_mismatch", "sum"),
            mean_expected=("expected_profit_rate", "mean"),
            mean_actual=("actual_runoff_profit_rate", "mean"),
            mean_mismatch=("mismatch", "mean"),
        )
        .sort_values("split")
    )
    summary.to_csv(output_dir / "segment_profit_mismatch_summary.csv", index=False)

    for split in ["train", "validation", "test"]:
        plot_split(frame, output_dir, split)

    for split in ["train", "validation", "test"]:
        split_frame = frame[frame["split"] == split]
        bad = split_frame[split_frame["expected_positive_actual_negative"]].sort_values("mismatch")
        print(f"\n[{split}]")
        print(summary[summary["split"] == split].to_string(index=False))
        if bad.empty:
            print("No expected-positive / actual-negative segments.")
        else:
            columns = [
                "segment",
                "label",
                "loan_pd",
                "annual_interest_rate",
                "avg_term_months",
                "expected_profit_rate",
                "actual_runoff_profit_rate",
                "mismatch",
            ]
            print("Expected-positive but actual-negative segments:")
            print(bad[columns].to_string(index=False))

    print(f"\nWrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lending_rl_agent.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


OUTCOME_LEAKAGE_COLUMNS = {
    "loan_status",
    "bad_loan",
    "total_pymnt",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "settlement_status",
    "debt_settlement_flag",
}

HISTORICAL_POLICY_PROXY_COLUMNS = {
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit obvious leakage risks in the segment PPO pipeline.")
    parser.add_argument(
        "--metrics",
        default="outputs/segment_ppo_800_lambda1_inflation25_eprfix_sharp_v2/metrics.json",
        help="metrics.json from a completed training run.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    code_features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    metrics_features = metrics.get("scorer", {}).get("risk_scorer_features", [])
    feature_set = set(metrics_features or code_features)

    outcome_overlap = sorted(feature_set & OUTCOME_LEAKAGE_COLUMNS)
    policy_proxy_overlap = sorted(feature_set & HISTORICAL_POLICY_PROXY_COLUMNS)

    calibration = metrics.get("calibration", {})
    train_calibration = calibration.get("train", {})
    scorer = metrics.get("scorer", {})

    report = {
        "metrics_path": str(metrics_path),
        "risk_scorer_feature_count": len(metrics_features or code_features),
        "risk_scorer_features": metrics_features or code_features,
        "outcome_leakage_feature_overlap": outcome_overlap,
        "historical_policy_proxy_feature_overlap": policy_proxy_overlap,
        "excluded_historical_policy_features": scorer.get("excluded_historical_policy_features"),
        "model": scorer.get("model"),
        "validation_auc": scorer.get("validation_auc"),
        "test_auc": scorer.get("test_auc"),
        "split_counts": scorer.get("split_counts"),
        "pd_calibration_method": train_calibration.get("pd_calibration_method"),
        "expected_profit_rate_model": train_calibration.get("expected_profit_rate_model"),
        "train_period": [train_calibration.get("first_month"), train_calibration.get("last_month")],
        "validation_period": [
            calibration.get("validation", {}).get("first_month"),
            calibration.get("validation", {}).get("last_month"),
        ],
        "test_period": [calibration.get("test", {}).get("first_month"), calibration.get("test", {}).get("last_month")],
        "verdict": "pass" if not outcome_overlap and not policy_proxy_overlap else "review",
        "important_caveats": [
            "Lending Club accepted loans are used as applicant-demand proxy; rejected applicants are unobserved.",
            "Only matured statuses are retained to define empirical default labels, so this is a calibrated simulator rather than direct real-world policy evaluation.",
            "Validation/test calibration creates out-of-time simulation scenarios; it is not logged-policy ground truth for the PPO policy.",
        ],
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
}

RAW_COLUMNS = [
    "issue_d",
    "loan_status",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "total_pymnt",
    "application_type",
    "acc_open_past_24mths",
    "mort_acc",
    "pub_rec_bankruptcies",
]

NUMERIC_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "fico_avg",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "term_months",
    "acc_open_past_24mths",
    "mort_acc",
    "pub_rec_bankruptcies",
]

CATEGORICAL_FEATURES = [
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
]


@dataclass(frozen=True)
class MarketData:
    """Monthly borrower pools consumed by the bank-level RL environment."""

    months: list[str]
    monthly_loans: list[pd.DataFrame]
    metadata: dict

    @property
    def month_count(self) -> int:
        return len(self.months)


def load_accepted_loans(
    accepted_path: str | Path,
    max_rows: int | None = 120_000,
    sample_mode: str = "reservoir",
    chunksize: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load a bounded Lending Club accepted-loan sample.

    ``reservoir`` scans the file and keeps a uniform sample across all months.
    ``head`` is much faster and useful only for smoke tests.
    """

    path = Path(accepted_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Accepted-loan file not found: {path}")

    usecols = lambda col: col in set(RAW_COLUMNS)

    if sample_mode not in {"reservoir", "head"}:
        raise ValueError("sample_mode must be either 'reservoir' or 'head'")

    if sample_mode == "head" or max_rows is None:
        frame = pd.read_csv(path, usecols=usecols, nrows=max_rows, low_memory=False)
        frame["_sample_weight"] = 1.0
        return frame

    rng = np.random.default_rng(seed)
    sample: pd.DataFrame | None = None
    source_rows = 0

    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        if chunk.empty:
            continue
        source_rows += len(chunk)
        chunk = chunk.copy()
        chunk["_sample_key"] = rng.random(len(chunk))
        if sample is None:
            sample = chunk
        else:
            sample = pd.concat([sample, chunk], ignore_index=True)
        if len(sample) > max_rows:
            sample = sample.nsmallest(max_rows, "_sample_key")
            sample = sample.reset_index(drop=True)

    if sample is None:
        return pd.DataFrame(columns=RAW_COLUMNS)

    sample = sample.drop(columns=["_sample_key"], errors="ignore").reset_index(drop=True)
    sample["_sample_weight"] = max(float(source_rows) / max(float(len(sample)), 1.0), 1.0)
    return sample


def prepare_scored_loans(raw_loans: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Clean loans, train the ML risk scorer, and attach ``risk_score``."""

    loans = _clean_loans(raw_loans)
    if loans.empty:
        raise ValueError("No mature accepted-loan records were available after cleaning.")

    scorer, model_summary = _fit_risk_scorer(loans, seed=seed)
    if scorer is None:
        loans["risk_score"] = _heuristic_risk_score(loans)
        model_summary["model"] = "heuristic"
    else:
        loans["risk_score"] = scorer.predict_proba(loans[_feature_columns()])[:, 1]
        model_summary["model"] = "logistic_regression"

    loans["risk_score"] = loans["risk_score"].clip(0.0, 1.0)
    loans = loans.sort_values(["issue_month", "risk_score"]).reset_index(drop=True)
    model_summary["loans_after_cleaning"] = int(len(loans))
    model_summary["bad_rate"] = float(loans["bad_loan"].mean())
    model_summary.update(_scorer_feature_summary(loans))
    return loans, model_summary


def prepare_scored_loans_temporal(
    raw_loans: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Train the ML scorer on earlier months and score every split.

    The final split is intentionally held out for one-time reporting.
    """

    loans = _clean_loans(raw_loans)
    if loans.empty:
        raise ValueError("No mature accepted-loan records were available after cleaning.")

    loans = _assign_temporal_splits(loans, train_ratio, validation_ratio)
    train_loans = loans[loans["split"] == "train"]
    y_train = train_loans["bad_loan"].astype(int)
    summary = {
        "training_rows": int(len(train_loans)),
        "validation_auc": None,
        "test_auc": None,
        "model": "logistic_regression_temporal",
        "split_counts": loans["split"].value_counts().to_dict(),
    }

    if len(train_loans) < 500 or y_train.nunique() < 2:
        loans["risk_score"] = _heuristic_risk_score(loans)
        summary["model"] = "heuristic_temporal"
    else:
        scorer = Pipeline(
            steps=[
                ("preprocess", _build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=350,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        scorer.fit(train_loans[_feature_columns()], y_train)
        loans["risk_score"] = scorer.predict_proba(loans[_feature_columns()])[:, 1]
        for split_name, key in [("validation", "validation_auc"), ("test", "test_auc")]:
            split_loans = loans[loans["split"] == split_name]
            if len(split_loans) and split_loans["bad_loan"].nunique() == 2:
                summary[key] = float(
                    roc_auc_score(
                        split_loans["bad_loan"].astype(int),
                        split_loans["risk_score"],
                    )
                )

    loans["risk_score"] = loans["risk_score"].clip(0.0, 1.0)
    loans = loans.sort_values(["issue_month", "risk_score"]).reset_index(drop=True)
    summary["loans_after_cleaning"] = int(len(loans))
    summary["bad_rate"] = float(loans["bad_loan"].mean())
    summary.update(_scorer_feature_summary(loans))
    return loans, summary


def build_market_data(scored_loans: pd.DataFrame) -> MarketData:
    """Group scored loans into monthly borrower pools."""

    required = {
        "issue_month",
        "funded_amnt",
        "total_pymnt",
        "term_months",
        "risk_score",
        "bad_loan",
    }
    missing = required.difference(scored_loans.columns)
    if missing:
        raise ValueError(f"Missing required scored-loan columns: {sorted(missing)}")

    months: list[str] = []
    monthly_loans: list[pd.DataFrame] = []
    grouped = scored_loans.groupby("issue_month", sort=True)
    keep_columns = [
        "funded_amnt",
        "installment",
        "int_rate",
        "total_pymnt",
        "term_months",
        "risk_score",
        "bad_loan",
        "loan_status",
    ]

    for month, group in grouped:
        month_loans = group[keep_columns].copy()
        month_loans = month_loans.sort_values("risk_score").reset_index(drop=True)
        months.append(str(pd.Period(month, freq="M")))
        monthly_loans.append(month_loans)

    metadata = {
        "month_count": len(months),
        "loan_count": int(len(scored_loans)),
        "first_month": months[0] if months else None,
        "last_month": months[-1] if months else None,
        "mean_monthly_demand": float(scored_loans.groupby("issue_month")["funded_amnt"].sum().mean()),
        "mean_risk_score": float(scored_loans["risk_score"].mean()),
    }
    return MarketData(months=months, monthly_loans=monthly_loans, metadata=metadata)


def _clean_loans(raw: pd.DataFrame) -> pd.DataFrame:
    loans = raw.copy()
    for column in RAW_COLUMNS:
        if column not in loans.columns:
            loans[column] = np.nan

    loans["loan_status"] = loans["loan_status"].astype(str)
    loans = loans[loans["loan_status"].isin(GOOD_STATUSES | BAD_STATUSES)].copy()

    loans["issue_month"] = pd.to_datetime(
        loans["issue_d"],
        format="%b-%Y",
        errors="coerce",
    ).dt.to_period("M").dt.to_timestamp()

    loans["term_months"] = (
        loans["term"].astype(str).str.extract(r"(\d+)")[0].pipe(pd.to_numeric, errors="coerce")
    )
    loans["int_rate"] = _parse_percent(loans["int_rate"])
    loans["revol_util"] = _parse_percent(loans["revol_util"])

    numeric_columns = (set(NUMERIC_FEATURES) - {"fico_avg"}) | {"total_pymnt"}
    for column in numeric_columns:
        loans[column] = pd.to_numeric(loans[column], errors="coerce")

    loans["term_months"] = loans["term_months"].fillna(36).clip(lower=1, upper=84)
    loans["fico_avg"] = loans[["fico_range_low", "fico_range_high"]].mean(axis=1)
    loans["bad_loan"] = loans["loan_status"].isin(BAD_STATUSES).astype(int)

    loans = loans.dropna(subset=["issue_month", "funded_amnt", "total_pymnt"])
    loans = loans[(loans["funded_amnt"] > 0) & (loans["total_pymnt"] >= 0)]
    return loans.reset_index(drop=True)


def _assign_temporal_splits(
    loans: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
) -> pd.DataFrame:
    loans = loans.copy()
    months = sorted(loans["issue_month"].dropna().unique())
    if len(months) < 3:
        loans["split"] = "train"
        return loans

    train_ratio = min(max(float(train_ratio), 0.1), 0.85)
    validation_ratio = min(max(float(validation_ratio), 0.05), 0.80)
    if train_ratio + validation_ratio >= 0.95:
        validation_ratio = 0.95 - train_ratio

    train_end = max(1, int(len(months) * train_ratio))
    validation_end = max(train_end + 1, int(len(months) * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, len(months) - 1)

    train_months = set(months[:train_end])
    validation_months = set(months[train_end:validation_end])
    loans["split"] = "test"
    loans.loc[loans["issue_month"].isin(train_months), "split"] = "train"
    loans.loc[loans["issue_month"].isin(validation_months), "split"] = "validation"
    return loans


def _fit_risk_scorer(loans: pd.DataFrame, seed: int) -> tuple[Pipeline | None, dict]:
    y = loans["bad_loan"].astype(int)
    summary = {
        "training_rows": int(len(loans)),
        "validation_auc": None,
    }
    if len(loans) < 500 or y.nunique() < 2:
        return None, summary

    x_train, x_valid, y_train, y_valid = train_test_split(
        loans[_feature_columns()],
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    model = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=350,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    valid_prob = model.predict_proba(x_valid)[:, 1]
    summary["validation_auc"] = float(roc_auc_score(y_valid, valid_prob))
    return model, summary


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def _heuristic_risk_score(loans: pd.DataFrame) -> pd.Series:
    fico = loans["fico_avg"].fillna(loans["fico_avg"].median())
    fico_risk = 1.0 - ((fico - 600.0) / 250.0).clip(0.0, 1.0)
    dti = loans["dti"].fillna(loans["dti"].median())
    dti_risk = (dti / 45.0).clip(0.0, 1.0)
    delinq = loans["delinq_2yrs"].fillna(loans["delinq_2yrs"].median())
    delinq_risk = (delinq / 5.0).clip(0.0, 1.0)
    inquiries = loans["inq_last_6mths"].fillna(loans["inq_last_6mths"].median())
    inquiry_risk = (inquiries / 6.0).clip(0.0, 1.0)
    return (0.45 * fico_risk + 0.25 * dti_risk + 0.15 * delinq_risk + 0.15 * inquiry_risk).clip(0.0, 1.0)


def _feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _scorer_feature_summary(loans: pd.DataFrame) -> dict:
    summary = {
        "risk_scorer_features": _feature_columns(),
        "excluded_historical_policy_features": ["int_rate", "installment", "grade", "sub_grade"],
    }
    if "_sample_weight" in loans.columns:
        weights = pd.to_numeric(loans["_sample_weight"], errors="coerce").fillna(1.0)
        summary["sample_weight_mean"] = float(weights.mean())
        summary["sample_weight_max"] = float(weights.max())
    return summary


def _parse_percent(values: Iterable) -> pd.Series:
    return pd.Series(values).astype(str).str.replace("%", "", regex=False).pipe(pd.to_numeric, errors="coerce")

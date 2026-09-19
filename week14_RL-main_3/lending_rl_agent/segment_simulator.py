from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without gymnasium installed
    raise ImportError("SegmentLendingEnv requires `gymnasium`. Install requirements-neural.txt.") from exc


BOOK_METRIC_KEYS = (
    "cash_inflow",
    "gross_cash_inflow",
    "cash_outflow",
    "interest_income",
    "principal_paid",
    "recovery",
    "default_loss",
    "defaulted_exposure",
    "funding_cost",
    "servicing_cost",
    "cash_inflation_cost",
    "liquidity_shock_loss",
    "funding_shock_cost",
    "capital_shock_loss",
    "opportunity_capture_value",
    "fatigue_franchise_cost",
    "period_profit",
)

SEGMENT_PARAMETER_NAMES = (
    "expected_profit_rate",
    "segment_stress_sensitivity",
    "segment_duration_pressure",
    "risk_adjusted_expected_profit_rate",
    "segment_opportunity_readiness",
    "segment_fatigue",
)
SEGMENT_PARAMETER_COUNT = len(SEGMENT_PARAMETER_NAMES)

GLOBAL_STATE_NAMES = (
    "cash_ratio",
    "elapsed_horizon_fraction",
    "remaining_horizon_fraction",
    "capital_ratio",
    "required_liquidity_ratio",
    "liquidity_coverage_ratio",
    "risk_weighted_assets_ratio",
    "capital_adequacy_ratio",
    "demand_regime_multiplier",
    "default_hazard_multiplier",
    "good_opportunity_multiplier",
    "recovery_opportunity_multiplier",
    "stress_intensity",
    "risk_weight_multiplier",
    "repayment_delay_multiplier",
    "segment_fatigue_mean",
)
GLOBAL_STATE_COUNT = len(GLOBAL_STATE_NAMES)
OBSERVATION_RATIO_CAP = 10.0


@dataclass(frozen=True)
class SegmentCalibration:
    name: str
    risk_edges: np.ndarray
    amount_edges: np.ndarray
    segment_labels: list[str]
    monthly_demand: np.ndarray
    monthly_counts: np.ndarray
    loan_pd: np.ndarray
    monthly_default_hazard: np.ndarray
    avg_loan_amount: np.ndarray
    annual_interest_rate: np.ndarray
    recovery_rate: np.ndarray
    avg_term_months: np.ndarray
    expected_profit_rate: np.ndarray
    new_loan_bucket: np.ndarray
    funding_cost_annual: float
    servicing_cost_rate: float
    metadata: dict[str, Any]
    risk_bucket_count: int = 5
    amount_tier_count: int = 3
    term_bucket_count: int = 2
    maturity_bucket_count: int = 6

    @property
    def segment_count(self) -> int:
        return len(self.segment_labels)

    @classmethod
    def from_scored(
        cls,
        scored: pd.DataFrame,
        split: str,
        risk_edges: np.ndarray | None = None,
        amount_edges: np.ndarray | None = None,
        risk_bucket_count: int = 5,
        amount_tier_count: int = 3,
        term_bucket_count: int = 2,
        maturity_bucket_count: int = 6,
        recovery_rate: float = 0.10,
        funding_cost_annual: float = 0.03,
        servicing_cost_rate: float = 0.005,
    ) -> "SegmentCalibration":
        frame = scored[scored["split"] == split].copy() if "split" in scored.columns else scored.copy()
        if frame.empty:
            raise ValueError(f"No rows available for segment calibration split: {split}")

        frame = frame.sort_values("issue_month").reset_index(drop=True)
        for column in ["funded_amnt", "int_rate", "term_months", "risk_score", "bad_loan"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["issue_month", "funded_amnt", "int_rate", "term_months", "risk_score", "bad_loan"])
        frame = frame[frame["funded_amnt"] > 0].copy()
        if frame.empty:
            raise ValueError(f"No valid rows available for segment calibration split: {split}")
        if "_sample_weight" not in frame.columns:
            frame["_sample_weight"] = 1.0
        frame["_sample_weight"] = pd.to_numeric(frame["_sample_weight"], errors="coerce").fillna(1.0).clip(lower=1.0)
        frame["_weighted_funded_amnt"] = frame["funded_amnt"] * frame["_sample_weight"]

        if risk_edges is None:
            risk_edges = _quantile_edges(frame["risk_score"], risk_bucket_count)
        if amount_edges is None:
            amount_edges = _quantile_edges(frame["funded_amnt"], amount_tier_count)
        risk_edges = np.asarray(risk_edges, dtype=np.float64)
        amount_edges = np.asarray(amount_edges, dtype=np.float64)

        observed_bad_rate = float(frame["bad_loan"].mean())
        mean_score = max(float(frame["risk_score"].mean()), 1e-6)

        risk_bucket = _bucketize(frame["risk_score"].to_numpy(dtype=np.float64), risk_edges)
        amount_tier = _bucketize(frame["funded_amnt"].to_numpy(dtype=np.float64), amount_edges)
        term_bucket = np.where(frame["term_months"].to_numpy(dtype=np.float64) <= 36.0, 0, 1)
        segment_id = _segment_id(risk_bucket, amount_tier, term_bucket, amount_tier_count, term_bucket_count)
        frame["segment_id"] = segment_id

        segment_count = risk_bucket_count * amount_tier_count * term_bucket_count
        segment_labels = _segment_labels(risk_bucket_count, amount_tier_count, term_bucket_count)
        global_funded = float(frame["funded_amnt"].mean())
        global_rate = float(frame["int_rate"].mean()) / 100.0
        global_term = float(frame["term_months"].mean())
        global_pd = float(np.average(frame["bad_loan"].to_numpy(dtype=np.float64), weights=frame["_sample_weight"]))

        avg_loan_amount = np.full(segment_count, global_funded, dtype=np.float64)
        annual_interest_rate = np.full(segment_count, global_rate, dtype=np.float64)
        avg_term_months = np.full(segment_count, global_term, dtype=np.float64)
        loan_pd = np.full(segment_count, min(max(global_pd, 0.001), 0.95), dtype=np.float64)

        grouped = frame.groupby("segment_id", sort=True)
        for sid, group in grouped:
            sid = int(sid)
            avg_loan_amount[sid] = max(float(group["funded_amnt"].mean()), 1.0)
            annual_interest_rate[sid] = max(float(group["int_rate"].mean()) / 100.0, 0.0)
            avg_term_months[sid] = float(group["term_months"].mean())
            segment_pd = float(
                np.average(group["bad_loan"].to_numpy(dtype=np.float64), weights=group["_sample_weight"])
            )
            loan_pd[sid] = min(max(segment_pd, 0.001), 0.95)

        avg_term_months = np.clip(avg_term_months, 1.0, 84.0)
        monthly_default_hazard = 1.0 - np.power(1.0 - loan_pd, 1.0 / avg_term_months)
        recovery = np.full(segment_count, float(recovery_rate), dtype=np.float64)
        max_term = max(60.0, float(np.nanmax(avg_term_months)))
        new_loan_bucket = np.array(
            [_term_to_bucket(term, maturity_bucket_count, max_term) for term in avg_term_months],
            dtype=np.int64,
        )
        expected_profit_rate = _expected_profit_rate(
            loan_pd,
            annual_interest_rate,
            avg_term_months,
            recovery,
            funding_cost_annual,
            servicing_cost_rate,
            maturity_bucket_count=maturity_bucket_count,
        )

        months = sorted(frame["issue_month"].dropna().unique())
        monthly_demand = np.zeros((len(months), segment_count), dtype=np.float64)
        monthly_counts = np.zeros((len(months), segment_count), dtype=np.float64)
        month_index = {month: idx for idx, month in enumerate(months)}
        for (month, sid), group in frame.groupby(["issue_month", "segment_id"], sort=True):
            row = month_index[month]
            col = int(sid)
            monthly_demand[row, col] = float(group["_weighted_funded_amnt"].sum())
            monthly_counts[row, col] = float(group["_sample_weight"].sum())

        metadata = {
            "split": split,
            "rows": int(len(frame)),
            "months": int(len(months)),
            "first_month": str(pd.Period(frame["issue_month"].min(), freq="M")),
            "last_month": str(pd.Period(frame["issue_month"].max(), freq="M")),
            "segment_count": int(segment_count),
            "risk_bucket_count": int(risk_bucket_count),
            "amount_tier_count": int(amount_tier_count),
            "term_bucket_count": int(term_bucket_count),
            "maturity_bucket_count": int(maturity_bucket_count),
            "observed_bad_rate": observed_bad_rate,
            "mean_risk_score": mean_score,
            "pd_calibration_method": "segment_empirical_bad_rate",
            "expected_profit_rate_model": "deterministic_monthly_runoff",
            "funding_cost_annual": float(funding_cost_annual),
            "servicing_cost_rate": float(servicing_cost_rate),
            "demand_sample_weight_mean": float(frame["_sample_weight"].mean()),
            "demand_sample_weight_max": float(frame["_sample_weight"].max()),
            "mean_monthly_demand": float(monthly_demand.sum(axis=1).mean()),
            "positive_profit_segment_share": float(np.mean(expected_profit_rate > 0.0)),
        }
        return cls(
            name=split,
            risk_edges=risk_edges,
            amount_edges=amount_edges,
            segment_labels=segment_labels,
            monthly_demand=monthly_demand,
            monthly_counts=monthly_counts,
            loan_pd=loan_pd,
            monthly_default_hazard=monthly_default_hazard,
            avg_loan_amount=avg_loan_amount,
            annual_interest_rate=annual_interest_rate,
            recovery_rate=recovery,
            avg_term_months=avg_term_months,
            expected_profit_rate=expected_profit_rate,
            new_loan_bucket=new_loan_bucket,
            funding_cost_annual=float(funding_cost_annual),
            servicing_cost_rate=float(servicing_cost_rate),
            metadata=metadata,
            risk_bucket_count=risk_bucket_count,
            amount_tier_count=amount_tier_count,
            term_bucket_count=term_bucket_count,
            maturity_bucket_count=maturity_bucket_count,
        )

    def stressed(
        self,
        name: str,
        default_multiplier: float = 1.0,
        high_risk_demand_multiplier: float = 1.0,
        recovery_multiplier: float = 1.0,
    ) -> "SegmentCalibration":
        demand_multiplier = np.ones(self.segment_count, dtype=np.float64)
        for segment in range(self.segment_count):
            risk_bucket, _, _ = self.decode_segment(segment)
            if risk_bucket >= self.risk_bucket_count - 2:
                demand_multiplier[segment] = high_risk_demand_multiplier
        metadata = dict(self.metadata)
        metadata.update(
            {
                "stress_name": name,
                "default_multiplier": float(default_multiplier),
                "high_risk_demand_multiplier": float(high_risk_demand_multiplier),
                "recovery_multiplier": float(recovery_multiplier),
                "expected_profit_rate_recomputed": True,
            }
        )
        stressed_loan_pd = np.clip(self.loan_pd * default_multiplier, 0.001, 0.95)
        stressed_recovery = np.clip(self.recovery_rate * recovery_multiplier, 0.0, 1.0)
        stressed_expected_profit_rate = _expected_profit_rate(
            stressed_loan_pd,
            self.annual_interest_rate,
            self.avg_term_months,
            stressed_recovery,
            self.funding_cost_annual,
            self.servicing_cost_rate,
            maturity_bucket_count=self.maturity_bucket_count,
        )
        return SegmentCalibration(
            name=name,
            risk_edges=self.risk_edges,
            amount_edges=self.amount_edges,
            segment_labels=self.segment_labels,
            monthly_demand=self.monthly_demand * demand_multiplier[None, :],
            monthly_counts=self.monthly_counts * demand_multiplier[None, :],
            loan_pd=stressed_loan_pd,
            monthly_default_hazard=1.0 - np.power(1.0 - stressed_loan_pd, 1.0 / self.avg_term_months),
            avg_loan_amount=self.avg_loan_amount,
            annual_interest_rate=self.annual_interest_rate,
            recovery_rate=stressed_recovery,
            avg_term_months=self.avg_term_months,
            expected_profit_rate=stressed_expected_profit_rate,
            new_loan_bucket=self.new_loan_bucket,
            funding_cost_annual=self.funding_cost_annual,
            servicing_cost_rate=self.servicing_cost_rate,
            metadata=metadata,
            risk_bucket_count=self.risk_bucket_count,
            amount_tier_count=self.amount_tier_count,
            term_bucket_count=self.term_bucket_count,
            maturity_bucket_count=self.maturity_bucket_count,
        )

    def decode_segment(self, segment_id: int) -> tuple[int, int, int]:
        term_bucket = int(segment_id % self.term_bucket_count)
        base = int(segment_id // self.term_bucket_count)
        amount_tier = int(base % self.amount_tier_count)
        risk_bucket = int(base // self.amount_tier_count)
        return risk_bucket, amount_tier, term_bucket


class SegmentLendingEnv(gym.Env):
    """Bank-level liquidity-constrained monthly lending allocation environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        calibration: SegmentCalibration,
        initial_cash: float = 50_000_000.0,
        min_cash_ratio: float = 0.20,
        horizon_months: int = 36,
        liquidity_penalty_lambda: float = 10.0,
        seed: int = 42,
        starting_cash: float | None = None,
        enforce_min_cash_constraint: bool = True,
        reserve_exposure_ratio: float = 0.05,
        terminal_runoff_months: int = 60,
        terminal_runoff_discount: float = 0.99,
        terminal_value_weight: float = 1.0,
        initial_loan_book: np.ndarray | None = None,
        scenario_name: str | None = None,
        liquidate_on_breach: bool = False,
        min_capital_ratio: float = 0.08,
        liquidity_liquidation_haircut: float = 0.60,
        expected_inflation_annual: float = 0.025,
        allocation_temperature: float = 1.0,
        segment_epr_prior_scale: float = 0.0,
        adaptive_epr_prior_scale: float = 0.0,
        expected_profit_shaping_weight: float = 0.0,
        allocation_top_k: int = 0,
        direct_allocation_decoder: str = "softmax",
        min_expected_profit_rate: float = 0.0,
        policy_action_mode: str = "direct",
        residual_baseline_policy: str = "one_step_lp_closed_form",
        residual_scale: float = 0.50,
        residual_deploy_scale: float = 0.50,
        residual_base_weight_floor: float = 0.0,
        epr_aware_lp_threshold: float = 0.09,
        residual_use_lend_gate: bool = True,
        residual_lend_gate_threshold: float = 0.02,
        demand_regime_mode: str = "ar1",
        demand_regime_rho: float = 0.85,
        demand_regime_volatility: float = 0.20,
        high_risk_demand_sensitivity: float = 0.50,
        default_regime_sensitivity: float = 0.50,
        observe_regime_state: bool = True,
        opportunity_regime: bool = True,
        good_opportunity_sensitivity: float = 1.25,
        recovery_opportunity_sensitivity: float = 1.00,
        recovery_opportunity_decay: float = 0.70,
        recovery_stress_threshold: float = 0.75,
        normal_opportunity_sensitivity: float = 0.60,
        risk_adjusted_normal_opportunity_weight: float = 0.20,
        stress_shock: bool = True,
        stress_threshold: float = 0.65,
        stress_base_probability: float = 0.01,
        stress_event_probability: float = 0.35,
        stress_event_bonus: float = 0.75,
        liquidity_shock_exposure_frac: float = 0.04,
        funding_shock_exposure_frac: float = 0.01,
        capital_shock_exposure_frac: float = 0.015,
        base_risk_weight_multiplier: float = 10.0,
        stress_risk_weight_inflation: float = 0.75,
        repayment_delay_strength: float = 0.65,
        duration_mismatch_strength: float = 0.35,
        stress_default_cluster_sensitivity: float = 1.25,
        fatigue_decay: float = 0.90,
        fatigue_growth: float = 0.35,
        fatigue_cap: float = 3.0,
        fatigue_default_sensitivity: float = 0.45,
        fatigue_margin_sensitivity: float = 0.10,
        fatigue_demand_sensitivity: float = 0.20,
        fatigue_concentration_growth: float = 0.25,
        fatigue_repayment_delay_sensitivity: float = 0.60,
        fatigue_risk_weight_sensitivity: float = 0.10,
        rotation_opportunity_fatigue_sensitivity: float = 1.50,
        rotation_opportunity_duration_sensitivity: float = 0.35,
        use_risk_adjusted_epr: bool = True,
        risk_adjusted_default_cost_weight: float = 0.65,
        risk_adjusted_shock_cost_weight: float = 1.00,
        risk_adjusted_capital_charge_weight: float = 0.03,
        risk_adjusted_repayment_delay_weight: float = 0.75,
        risk_adjusted_fatigue_cost_weight: float = 1.00,
        opportunity_capture_value_weight: float = 0.0,
        strategic_fatigue_cost_weight: float = 0.0,
        soft_risk_penalty_lambda: float = 0.25,
        soft_risk_lcr_target: float = 1.05,
        soft_risk_car_target: float = 0.10,
    ) -> None:
        super().__init__()
        self.calibration = calibration
        self.initial_cash = float(initial_cash)
        self.starting_cash = float(initial_cash if starting_cash is None else starting_cash)
        self.min_cash = float(initial_cash * min_cash_ratio)
        self.horizon_months = int(horizon_months)
        self.liquidity_penalty_lambda = float(liquidity_penalty_lambda)
        self.enforce_min_cash_constraint = bool(enforce_min_cash_constraint)
        self.reserve_exposure_ratio = max(0.0, float(reserve_exposure_ratio))
        self.terminal_runoff_months = max(0, int(terminal_runoff_months))
        self.terminal_runoff_discount = min(max(float(terminal_runoff_discount), 0.0), 1.0)
        self.terminal_value_weight = float(terminal_value_weight)
        self.liquidate_on_breach = bool(liquidate_on_breach)
        self.min_capital_ratio = max(0.0, float(min_capital_ratio))
        self.liquidity_liquidation_haircut = min(max(float(liquidity_liquidation_haircut), 0.0), 1.0)
        self.expected_inflation_annual = max(0.0, float(expected_inflation_annual))
        self.monthly_inflation_rate = (1.0 + self.expected_inflation_annual) ** (1.0 / 12.0) - 1.0
        self.allocation_temperature = max(float(allocation_temperature), 1e-3)
        self.segment_epr_prior_scale = float(segment_epr_prior_scale)
        self.adaptive_epr_prior_scale = max(0.0, float(adaptive_epr_prior_scale))
        self.expected_profit_shaping_weight = float(expected_profit_shaping_weight)
        self.allocation_top_k = max(0, int(allocation_top_k))
        self.direct_allocation_decoder = str(direct_allocation_decoder)
        self.min_expected_profit_rate = float(min_expected_profit_rate)
        self.policy_action_mode = str(policy_action_mode)
        self.residual_baseline_policy = str(residual_baseline_policy)
        self.residual_scale = max(0.0, float(residual_scale))
        self.residual_deploy_scale = max(0.0, float(residual_deploy_scale))
        self.residual_base_weight_floor = float(np.clip(residual_base_weight_floor, 0.0, 1.0))
        self.epr_aware_lp_threshold = float(epr_aware_lp_threshold)
        self.residual_use_lend_gate = bool(residual_use_lend_gate)
        self.residual_lend_gate_threshold = float(np.clip(residual_lend_gate_threshold, 0.0, 1.0))
        self.demand_regime_mode = str(demand_regime_mode)
        self.demand_regime_rho = float(np.clip(demand_regime_rho, -0.99, 0.99))
        self.demand_regime_volatility = max(0.0, float(demand_regime_volatility))
        self.high_risk_demand_sensitivity = max(0.0, float(high_risk_demand_sensitivity))
        self.default_regime_sensitivity = max(0.0, float(default_regime_sensitivity))
        self.observe_regime_state = bool(observe_regime_state)
        self.opportunity_regime = bool(opportunity_regime)
        self.good_opportunity_sensitivity = max(0.0, float(good_opportunity_sensitivity))
        self.recovery_opportunity_sensitivity = max(0.0, float(recovery_opportunity_sensitivity))
        self.recovery_opportunity_decay = float(np.clip(recovery_opportunity_decay, 0.0, 0.99))
        self.recovery_stress_threshold = max(0.0, float(recovery_stress_threshold))
        self.normal_opportunity_sensitivity = max(0.0, float(normal_opportunity_sensitivity))
        self.risk_adjusted_normal_opportunity_weight = max(0.0, float(risk_adjusted_normal_opportunity_weight))
        self.stress_shock = bool(stress_shock)
        self.stress_threshold = float(stress_threshold)
        self.stress_base_probability = float(np.clip(stress_base_probability, 0.0, 1.0))
        self.stress_event_probability = float(np.clip(stress_event_probability, 0.0, 1.0))
        self.stress_event_bonus = max(0.0, float(stress_event_bonus))
        self.liquidity_shock_exposure_frac = max(0.0, float(liquidity_shock_exposure_frac))
        self.funding_shock_exposure_frac = max(0.0, float(funding_shock_exposure_frac))
        self.capital_shock_exposure_frac = max(0.0, float(capital_shock_exposure_frac))
        self.base_risk_weight_multiplier = max(1e-6, float(base_risk_weight_multiplier))
        self.stress_risk_weight_inflation = max(0.0, float(stress_risk_weight_inflation))
        self.repayment_delay_strength = float(np.clip(repayment_delay_strength, 0.0, 0.95))
        self.duration_mismatch_strength = float(np.clip(duration_mismatch_strength, 0.0, 0.95))
        self.stress_default_cluster_sensitivity = max(0.0, float(stress_default_cluster_sensitivity))
        self.fatigue_decay = float(np.clip(fatigue_decay, 0.0, 0.999))
        self.fatigue_growth = max(0.0, float(fatigue_growth))
        self.fatigue_cap = max(1e-6, float(fatigue_cap))
        self.fatigue_default_sensitivity = max(0.0, float(fatigue_default_sensitivity))
        self.fatigue_margin_sensitivity = max(0.0, float(fatigue_margin_sensitivity))
        self.fatigue_demand_sensitivity = max(0.0, float(fatigue_demand_sensitivity))
        self.fatigue_concentration_growth = max(0.0, float(fatigue_concentration_growth))
        self.fatigue_repayment_delay_sensitivity = max(0.0, float(fatigue_repayment_delay_sensitivity))
        self.fatigue_risk_weight_sensitivity = max(0.0, float(fatigue_risk_weight_sensitivity))
        self.rotation_opportunity_fatigue_sensitivity = max(0.0, float(rotation_opportunity_fatigue_sensitivity))
        self.rotation_opportunity_duration_sensitivity = max(0.0, float(rotation_opportunity_duration_sensitivity))
        self.use_risk_adjusted_epr = bool(use_risk_adjusted_epr)
        self.risk_adjusted_default_cost_weight = max(0.0, float(risk_adjusted_default_cost_weight))
        self.risk_adjusted_shock_cost_weight = max(0.0, float(risk_adjusted_shock_cost_weight))
        self.risk_adjusted_capital_charge_weight = max(0.0, float(risk_adjusted_capital_charge_weight))
        self.risk_adjusted_repayment_delay_weight = max(0.0, float(risk_adjusted_repayment_delay_weight))
        self.risk_adjusted_fatigue_cost_weight = max(0.0, float(risk_adjusted_fatigue_cost_weight))
        self.opportunity_capture_value_weight = max(0.0, float(opportunity_capture_value_weight))
        self.strategic_fatigue_cost_weight = max(0.0, float(strategic_fatigue_cost_weight))
        self.soft_risk_penalty_lambda = max(0.0, float(soft_risk_penalty_lambda))
        self.soft_risk_lcr_target = max(0.0, float(soft_risk_lcr_target))
        self.soft_risk_car_target = max(0.0, float(soft_risk_car_target))
        self.scenario_name = scenario_name or calibration.metadata.get("stress_name") or calibration.name
        self.seed_value = int(seed)
        self._reset_rngs(self.seed_value)
        self.segment_count = calibration.segment_count
        self.maturity_bucket_count = calibration.maturity_bucket_count
        self.global_state_count = GLOBAL_STATE_COUNT
        self.segment_indices = np.arange(self.segment_count, dtype=np.int64)
        self.segment_new_loan_bucket = np.asarray(calibration.new_loan_bucket, dtype=np.int64)
        self.segment_risk_bucket = np.array(
            [calibration.decode_segment(segment)[0] for segment in range(self.segment_count)],
            dtype=np.int64,
        )
        risk_denominator = max(float(calibration.risk_bucket_count - 1), 1.0)
        self.segment_risk_rank = self.segment_risk_bucket.astype(np.float64) / risk_denominator
        term_values = calibration.avg_term_months.astype(np.float64)
        term_range = float(np.nanmax(term_values) - np.nanmin(term_values))
        self.segment_term_rank = (
            (term_values - np.nanmin(term_values)) / term_range
            if term_range > 1e-9
            else np.zeros_like(term_values)
        )
        positive_epr = np.maximum(calibration.expected_profit_rate.astype(np.float64), 0.0)
        epr_range = float(positive_epr.max() - positive_epr.min())
        self.segment_epr_rank = (
            (positive_epr - positive_epr.min()) / epr_range
            if epr_range > 1e-9
            else np.zeros_like(positive_epr)
        )
        self.segment_stress_sensitivity = np.clip(
            0.35 + 0.35 * self.segment_risk_rank + 0.30 * self.segment_term_rank + 0.25 * self.segment_epr_rank,
            0.0,
            1.75,
        )
        self.segment_duration_pressure = np.clip(
            0.25 + 0.50 * self.segment_term_rank + 0.25 * self.segment_epr_rank,
            0.0,
            1.25,
        )
        self.bucket_months = max(1.0, float(np.nanmax(calibration.avg_term_months)) / self.maturity_bucket_count)
        self.static_segment_parameter_state = self._build_static_segment_parameter_state()
        self.initial_loan_book = (
            np.asarray(initial_loan_book, dtype=np.float64).copy()
            if initial_loan_book is not None
            else np.zeros((self.segment_count, self.maturity_bucket_count), dtype=np.float64)
        )

        self.action_dim = self.segment_count + (
            2 if self.policy_action_mode == "residual" and self.residual_use_lend_gate else 1
        )
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                GLOBAL_STATE_COUNT
                + self.segment_count
                + self.segment_count * self.maturity_bucket_count
                + self.segment_count * SEGMENT_PARAMETER_COUNT,
            ),
            dtype=np.float32,
        )
        self.reset(seed=seed)

    def _reset_rngs(self, seed: int) -> None:
        self.demand_rng = np.random.default_rng(int(seed))
        self.default_rng = np.random.default_rng(int(seed) + 10_000_000)
        self.stress_rng = np.random.default_rng(int(seed) + 20_000_000)
        self.seed_value = int(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._reset_rngs(int(seed))
        options = options or {}
        starting_cash = float(options.get("starting_cash", self.starting_cash))
        self.budget = starting_cash
        self.capital = starting_cash
        self.elapsed = 0
        self.loan_book = self.initial_loan_book.copy()
        self.regime_state = self._initial_regime_state()
        self.recovery_opportunity_state = 0.0
        self.segment_fatigue = np.zeros(self.segment_count, dtype=np.float64)
        self.current_stress_intensity = self._sample_stress_intensity()
        self.current_stress_event = bool(self.current_stress_intensity > self._stress_pressure() + 1e-9)
        self.current_demand = self._sample_monthly_demand()
        self.trace: list[dict[str, Any]] = []
        self.done = False
        return self._state(), self._reset_info()

    def step(self, action):
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        allocation = self._allocation_from_actor_action(raw)
        return self.step_allocation(allocation)

    def step_allocation(self, allocation):
        if self.done:
            raise RuntimeError("Cannot step an environment after termination. Call reset().")

        allocation = self._project_allocation(np.asarray(allocation, dtype=np.float64).reshape(-1))
        start_budget = self.budget
        book_metrics = self._simulate_existing_book()
        decision_period_profit = float(book_metrics["period_profit"])
        self.capital += decision_period_profit
        self.budget += book_metrics["cash_inflow"]

        allocation = self._project_allocation(allocation)
        disbursement = float(allocation.sum())
        self.budget -= disbursement
        if disbursement > 0.0:
            self.loan_book[self.segment_indices, self.segment_new_loan_bucket] += allocation

        inflation_metrics = self._simulate_cash_inflation()
        book_metrics = self._merge_book_metrics(book_metrics, inflation_metrics)
        decision_period_profit += float(inflation_metrics["period_profit"])
        self.capital += float(inflation_metrics["period_profit"])
        self.budget += float(inflation_metrics["cash_inflow"])

        stress_metrics = self._simulate_stress_shock()
        book_metrics = self._merge_book_metrics(book_metrics, stress_metrics)
        decision_period_profit += float(stress_metrics["period_profit"])
        self.capital += float(stress_metrics["period_profit"])
        self.budget += float(stress_metrics["cash_inflow"])
        previous_fatigue = self.segment_fatigue.copy()
        self._update_segment_fatigue(allocation)
        franchise_metrics = self._simulate_strategic_franchise(allocation, previous_fatigue)
        book_metrics = self._merge_book_metrics(book_metrics, franchise_metrics)
        decision_period_profit += float(franchise_metrics["period_profit"])
        self.capital += float(franchise_metrics["period_profit"])
        self.budget += float(franchise_metrics["cash_inflow"])
        expected_deployment_profit = float(np.dot(allocation, self.calibration.expected_profit_rate))
        risk_adjusted_deployment_profit = float(np.dot(allocation, self._risk_adjusted_expected_profit_rate()))
        expected_profit_shaping_reward = (
            self.expected_profit_shaping_weight * risk_adjusted_deployment_profit / max(self.initial_cash, 1.0)
        )

        required_liquidity = self._required_liquidity()
        shortfall, immediate_liquidity_penalty = self._liquidity_penalty()
        lcr_buffer_violation = max(0.0, self.soft_risk_lcr_target - self._liquidity_coverage_ratio(required_liquidity))
        current_car_for_penalty = self._capital_adequacy_ratio()
        car_buffer_violation = max(0.0, self.soft_risk_car_target - current_car_for_penalty)
        soft_risk_penalty = self.soft_risk_penalty_lambda * (
            lcr_buffer_violation**2 + car_buffer_violation**2
        )
        terminal_metrics = self._empty_terminal_metrics()
        metrics = dict(book_metrics)
        reward = (
            float(book_metrics["period_profit"]) / max(self.initial_cash, 1.0)
            + expected_profit_shaping_reward
            - immediate_liquidity_penalty
            - soft_risk_penalty
        )
        liquidity_penalty = immediate_liquidity_penalty
        min_budget_observed = self.budget
        max_liquidity_shortfall = shortfall
        liquidity_breach_count = int(shortfall > 0.0)
        liquidated = False
        liquidation_trigger = "none"
        liquidation_loss = 0.0
        liquidation_proceeds = 0.0
        capital_ratio = self._capital_adequacy_ratio()
        capital_shortfall = self._capital_shortfall()
        capital_breach = capital_shortfall > 0.0
        pre_liquidation_budget = float(self.budget)
        pre_liquidation_loan_book_exposure = float(np.maximum(self.loan_book, 0.0).sum())
        pre_liquidation_current_assets = pre_liquidation_budget + pre_liquidation_loan_book_exposure
        pre_liquidation_required_liquidity = float(required_liquidity)
        pre_liquidation_liquidity_shortfall = float(shortfall)
        pre_liquidation_liquidity_coverage_ratio = self._liquidity_coverage_ratio(required_liquidity)
        pre_liquidation_risk_weighted_assets = self._risk_weighted_assets()
        pre_liquidation_capital_adequacy_ratio = float(capital_ratio)
        pre_liquidation_capital_shortfall = float(capital_shortfall)

        self.elapsed += 1
        liquidity_breach = shortfall > 0.0
        if self.liquidate_on_breach and (capital_breach or liquidity_breach):
            liquidated = True
            current_assets = float(max(pre_liquidation_budget, 0.0) + pre_liquidation_loan_book_exposure)
            if capital_breach:
                liquidation_trigger = "capital_adequacy"
                liquidation_loss = float(max(self.capital, 0.0))
                liquidation_proceeds = 0.0
            else:
                liquidation_trigger = "liquidity_shortfall"
                liquidation_loss = float(self.liquidity_liquidation_haircut * current_assets)
                liquidation_proceeds = float(max(0.0, current_assets - liquidation_loss))
            metrics["period_profit"] = float(metrics["period_profit"]) - liquidation_loss
            decision_period_profit -= liquidation_loss
            reward -= liquidation_loss / max(self.initial_cash, 1.0)
            self.capital = max(0.0, self.capital - liquidation_loss)
            self.budget = liquidation_proceeds
            self.loan_book = np.zeros_like(self.loan_book)
            min_budget_observed = min(min_budget_observed, self.budget)
            self.done = True
        else:
            self.done = self.elapsed >= self.horizon_months
        if self.done:
            if not liquidated:
                terminal_metrics = self._simulate_terminal_runoff()
                reward += terminal_metrics["reward"]
                liquidity_penalty += terminal_metrics["liquidity_penalty"]
                min_budget_observed = min(min_budget_observed, terminal_metrics["min_budget"])
                max_liquidity_shortfall = max(max_liquidity_shortfall, terminal_metrics["max_shortfall"])
                liquidity_breach_count += int(terminal_metrics["liquidity_breach_count"])
                for key in BOOK_METRIC_KEYS:
                    metrics[key] = float(metrics.get(key, 0.0)) + float(terminal_metrics.get(key, 0.0))
                metrics["period_profit"] += float(terminal_metrics["terminal_value"])
                self.capital += float(terminal_metrics["period_profit"]) + float(terminal_metrics["terminal_value"])
                reward += float(terminal_metrics["terminal_value_reward"])
            required_liquidity = self._required_liquidity()
            shortfall = self._liquidity_shortfall()
            capital_ratio = self._capital_adequacy_ratio()
            capital_shortfall = self._capital_shortfall()
            capital_breach = capital_shortfall > 0.0
        else:
            self._advance_regime()
            self.current_stress_intensity = self._sample_stress_intensity()
            self.current_stress_event = bool(self.current_stress_intensity > self._stress_pressure() + 1e-9)
            self.current_demand = self._sample_monthly_demand()

        reported_required_liquidity = pre_liquidation_required_liquidity if liquidated else float(required_liquidity)
        reported_liquidity_shortfall = (
            pre_liquidation_liquidity_shortfall if liquidated else float(shortfall)
        )
        reported_liquidity_coverage_ratio = (
            pre_liquidation_liquidity_coverage_ratio
            if liquidated
            else self._liquidity_coverage_ratio(required_liquidity)
        )
        reported_risk_weighted_assets = (
            pre_liquidation_risk_weighted_assets if liquidated else self._risk_weighted_assets()
        )
        reported_capital_adequacy_ratio = (
            pre_liquidation_capital_adequacy_ratio if liquidated else float(capital_ratio)
        )
        reported_capital_shortfall = (
            pre_liquidation_capital_shortfall if liquidated else float(capital_shortfall)
        )
        reported_capital_breach = bool(reported_capital_shortfall > 0.0)
        profit = float(metrics["period_profit"])
        deployable_denominator = max(start_budget - self._required_liquidity(self.loan_book), 1.0)
        info = {
            "step": self.elapsed,
            "start_budget": start_budget,
            "budget": self.budget,
            "capital": self.capital,
            "min_cash": self.min_cash,
            "required_liquidity": reported_required_liquidity,
            "reserve_exposure_ratio": self.reserve_exposure_ratio,
            "liquidity_coverage_ratio": reported_liquidity_coverage_ratio,
            "risk_weighted_assets": reported_risk_weighted_assets,
            "capital_adequacy_ratio": reported_capital_adequacy_ratio,
            "min_capital_ratio": self.min_capital_ratio,
            "capital_shortfall": reported_capital_shortfall,
            "capital_breach": reported_capital_breach,
            "liquidity_shortfall": reported_liquidity_shortfall,
            "max_liquidity_shortfall": max_liquidity_shortfall,
            "min_budget_observed": min_budget_observed,
            "demand": float(self.current_demand.sum()) if not self.done else 0.0,
            "disbursement": disbursement,
            "cash_inflow": float(metrics["cash_inflow"]),
            "gross_cash_inflow": float(metrics["gross_cash_inflow"]),
            "cash_outflow": float(metrics["cash_outflow"]),
            "interest_income": float(metrics["interest_income"]),
            "principal_paid": float(metrics["principal_paid"]),
            "recovery": float(metrics["recovery"]),
            "default_loss": float(metrics["default_loss"]),
            "defaulted_exposure": float(metrics["defaulted_exposure"]),
            "funding_cost": float(metrics["funding_cost"]),
            "servicing_cost": float(metrics["servicing_cost"]),
            "cash_inflation_cost": float(metrics["cash_inflation_cost"]),
            "period_profit": profit,
            "decision_period_profit": decision_period_profit,
            "liquidity_penalty": float(liquidity_penalty),
            "reward": float(reward),
            "expected_deployment_profit": expected_deployment_profit,
            "risk_adjusted_deployment_profit": risk_adjusted_deployment_profit,
            "expected_profit_shaping_reward": expected_profit_shaping_reward,
            "lcr_buffer_violation": lcr_buffer_violation,
            "car_buffer_violation": car_buffer_violation,
            "soft_risk_penalty": soft_risk_penalty,
            "demand_regime_state": float(self.regime_state),
            "demand_regime_multiplier": self._global_demand_regime_multiplier(),
            "default_hazard_multiplier": self._global_default_hazard_multiplier(),
            "good_opportunity_multiplier": self._global_good_opportunity_multiplier(),
            "recovery_opportunity_multiplier": self._global_recovery_opportunity_multiplier(),
            "stress_intensity": float(self.current_stress_intensity),
            "stress_event": bool(self.current_stress_event),
            "risk_weight_multiplier": self._global_risk_weight_multiplier(),
            "repayment_delay_multiplier": self._global_repayment_delay_multiplier(),
            "segment_fatigue_mean": float(self.segment_fatigue.mean()),
            "liquidity_shock_loss": float(metrics["liquidity_shock_loss"]),
            "funding_shock_cost": float(metrics["funding_shock_cost"]),
            "capital_shock_loss": float(metrics["capital_shock_loss"]),
            "opportunity_capture_value": float(metrics["opportunity_capture_value"]),
            "fatigue_franchise_cost": float(metrics["fatigue_franchise_cost"]),
            "capital_utilization": disbursement / deployable_denominator,
            "liquidity_breach": bool(liquidity_breach_count > 0 or shortfall > 0.0),
            "liquidity_breach_count": liquidity_breach_count,
            "liquidated": bool(liquidated),
            "liquidation_trigger": liquidation_trigger,
            "capital_liquidation": bool(liquidated and liquidation_trigger == "capital_adequacy"),
            "liquidity_liquidation": bool(liquidated and liquidation_trigger == "liquidity_shortfall"),
            "liquidation_loss": float(liquidation_loss),
            "liquidation_proceeds": float(liquidation_proceeds),
            "liquidation_asset_base": pre_liquidation_current_assets if liquidated else 0.0,
            "liquidity_liquidation_haircut": self.liquidity_liquidation_haircut,
            "pre_liquidation_budget": pre_liquidation_budget if liquidated else 0.0,
            "pre_liquidation_loan_book_exposure": (
                pre_liquidation_loan_book_exposure if liquidated else 0.0
            ),
            "pre_liquidation_required_liquidity": (
                pre_liquidation_required_liquidity if liquidated else 0.0
            ),
            "pre_liquidation_liquidity_coverage_ratio": (
                pre_liquidation_liquidity_coverage_ratio if liquidated else float("nan")
            ),
            "allocation": allocation.astype(np.float32),
            "allocation_by_risk": self._allocation_by_risk(allocation),
            "loan_book_exposure": float(self.loan_book.sum()),
            "terminal_runoff_months": int(terminal_metrics["runoff_months"]),
            "terminal_runoff_profit": float(terminal_metrics["period_profit"]),
            "terminal_value": float(terminal_metrics["terminal_value"]),
            "terminal_value_reward": float(terminal_metrics["terminal_value_reward"]),
        }
        self.trace.append(info)
        return self._state(), float(reward), bool(self.done), False, info

    def _empty_book_metrics(self) -> dict[str, float]:
        return {key: 0.0 for key in BOOK_METRIC_KEYS}

    def _empty_terminal_metrics(self) -> dict[str, float]:
        metrics = self._empty_book_metrics()
        metrics.update(
            {
                "reward": 0.0,
                "liquidity_penalty": 0.0,
                "min_budget": float(self.budget),
                "max_shortfall": 0.0,
                "liquidity_breach_count": 0,
                "runoff_months": 0,
                "terminal_value": 0.0,
                "terminal_value_reward": 0.0,
            }
        )
        return metrics

    def _simulate_cash_inflation(self) -> dict[str, float]:
        metrics = self._empty_book_metrics()
        if self.monthly_inflation_rate <= 0.0:
            return metrics
        cost = float(max(self.budget, 0.0) * self.monthly_inflation_rate)
        metrics["cash_inflow"] = -cost
        metrics["cash_outflow"] = cost
        metrics["cash_inflation_cost"] = cost
        metrics["period_profit"] = -cost
        return metrics

    def _simulate_stress_shock(self) -> dict[str, float]:
        metrics = self._empty_book_metrics()
        if not self.stress_shock or self.current_stress_intensity <= 0.0:
            return metrics
        exposure_by_segment = np.maximum(self.loan_book, 0.0).sum(axis=1)
        if float(exposure_by_segment.sum()) <= 1e-9:
            return metrics
        weighted_exposure = float(np.dot(exposure_by_segment, 0.50 + self.segment_stress_sensitivity))
        intensity = float(self.current_stress_intensity)
        liquidity_loss = float(self.liquidity_shock_exposure_frac * intensity * weighted_exposure)
        funding_cost = float(self.funding_shock_exposure_frac * intensity * weighted_exposure)
        capital_loss = float(self.capital_shock_exposure_frac * intensity * weighted_exposure)
        cash_loss = liquidity_loss + funding_cost
        metrics["cash_inflow"] = -cash_loss
        metrics["cash_outflow"] = cash_loss
        metrics["liquidity_shock_loss"] = liquidity_loss
        metrics["funding_shock_cost"] = funding_cost
        metrics["capital_shock_loss"] = capital_loss
        metrics["period_profit"] = -(cash_loss + capital_loss)
        return metrics

    def _simulate_strategic_franchise(
        self,
        allocation: np.ndarray,
        previous_fatigue: np.ndarray,
    ) -> dict[str, float]:
        metrics = self._empty_book_metrics()
        if self.opportunity_capture_value_weight <= 0.0 and self.strategic_fatigue_cost_weight <= 0.0:
            return metrics

        fatigue_cap = max(self.fatigue_cap, 1e-6)
        fatigue_delta = np.maximum(self.segment_fatigue - previous_fatigue, 0.0) / fatigue_cap
        fatigue_load = (
            (0.50 + self.segment_duration_pressure)
            * (0.50 + self.segment_stress_sensitivity)
            * (0.50 + self.segment_epr_rank)
        )
        fatigue_cost = float(
            self.initial_cash
            * self.strategic_fatigue_cost_weight
            * np.mean(fatigue_delta * fatigue_load)
        )

        opportunity_intensity = 0.0
        if self.opportunity_regime:
            opportunity_intensity = float(
                np.clip(self._good_opportunity_state() + self.recovery_opportunity_state, 0.0, 2.0)
            )
        risk_adjusted_quality = np.clip(self._risk_adjusted_expected_profit_rate(), 0.0, None)
        quality = (0.25 + self.segment_epr_rank) * (0.25 + np.minimum(risk_adjusted_quality, 0.50))
        capture_basis = allocation * self._segment_opportunity_readiness() * quality
        opportunity_value = float(
            self.opportunity_capture_value_weight
            * opportunity_intensity
            * np.sum(capture_basis)
        )

        net_profit = opportunity_value - fatigue_cost
        metrics["cash_inflow"] = net_profit
        metrics["gross_cash_inflow"] = max(opportunity_value, 0.0)
        metrics["cash_outflow"] = fatigue_cost
        metrics["opportunity_capture_value"] = opportunity_value
        metrics["fatigue_franchise_cost"] = fatigue_cost
        metrics["period_profit"] = net_profit
        return metrics

    def _merge_book_metrics(self, left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
        return {key: float(left.get(key, 0.0)) + float(right.get(key, 0.0)) for key in BOOK_METRIC_KEYS}

    def _required_liquidity(self, loan_book: np.ndarray | None = None) -> float:
        exposure = self.loan_book if loan_book is None else loan_book
        return float(self.min_cash + self.reserve_exposure_ratio * np.maximum(exposure, 0.0).sum())

    def _liquidity_shortfall(self) -> float:
        shortfall = max(0.0, self._required_liquidity() - self.budget)
        return 0.0 if shortfall <= 1e-6 else shortfall

    def _liquidity_penalty(self) -> tuple[float, float]:
        shortfall = self._liquidity_shortfall()
        penalty = self.liquidity_penalty_lambda * (shortfall / max(self.initial_cash, 1.0)) ** 2
        return shortfall, penalty

    def _liquidity_coverage_ratio(self, required_liquidity: float | None = None) -> float:
        required = self._required_liquidity() if required_liquidity is None else float(required_liquidity)
        if required <= 1e-9:
            return float("inf")
        return float(self.budget / required)

    def _risk_weighted_assets(self) -> float:
        exposure_by_segment = np.maximum(self.loan_book, 0.0).sum(axis=1)
        return float(np.dot(exposure_by_segment, self._risk_weight_multiplier_by_segment()))

    def _capital_adequacy_ratio(self) -> float:
        risk_weighted_assets = self._risk_weighted_assets()
        if risk_weighted_assets <= 1e-9:
            return float("inf")
        return float(self.capital / risk_weighted_assets)

    def _capital_shortfall(self) -> float:
        risk_weighted_assets = self._risk_weighted_assets()
        if risk_weighted_assets <= 1e-9:
            return 0.0
        return max(0.0, self.min_capital_ratio * risk_weighted_assets - self.capital)

    def _simulate_terminal_runoff(self) -> dict[str, float]:
        metrics = self._empty_terminal_metrics()
        discount = 1.0
        for month in range(self.terminal_runoff_months):
            if not np.any(self.loan_book > 1e-6):
                break
            # Terminal runoff settles the outstanding loan book after the
            # decision horizon. Inflation drag is charged during decision
            # months only, keeping reject-all and lending policies on the same
            # inflation horizon.
            book_metrics = self._simulate_existing_book()
            self.budget += book_metrics["cash_inflow"]
            shortfall, penalty = self._liquidity_penalty()
            for key in BOOK_METRIC_KEYS:
                metrics[key] += float(book_metrics[key])
            metrics["reward"] += discount * (float(book_metrics["period_profit"]) / max(self.initial_cash, 1.0) - penalty)
            metrics["liquidity_penalty"] += float(discount * penalty)
            metrics["min_budget"] = min(float(metrics["min_budget"]), float(self.budget))
            metrics["max_shortfall"] = max(float(metrics["max_shortfall"]), float(shortfall))
            metrics["liquidity_breach_count"] += int(shortfall > 0.0)
            metrics["runoff_months"] = month + 1
            discount *= self.terminal_runoff_discount

        terminal_value = self.terminal_value_weight * self._terminal_expected_profit()
        metrics["terminal_value"] = float(terminal_value)
        metrics["terminal_value_reward"] = float(discount * terminal_value / max(self.initial_cash, 1.0))
        return metrics

    def _terminal_expected_profit(self) -> float:
        if self.terminal_value_weight == 0.0 or not np.any(self.loan_book > 0.0):
            return 0.0
        remaining_months = (np.arange(self.maturity_bucket_count, dtype=np.float64) + 0.5) * self.bucket_months
        term_scale = np.minimum(1.0, remaining_months[None, :] / np.maximum(self.calibration.avg_term_months[:, None], 1.0))
        expected_profit = self.loan_book * self.calibration.expected_profit_rate[:, None] * term_scale
        return float(expected_profit.sum())

    def _simulate_existing_book(self) -> dict[str, float]:
        exposure = self.loan_book
        if not np.any(exposure > 0.0):
            self.loan_book = np.zeros_like(exposure)
            return self._empty_book_metrics()

        hazard = (self.calibration.monthly_default_hazard * self._default_hazard_multiplier_by_segment())[:, None]
        avg_amount = np.maximum(self.calibration.avg_loan_amount[:, None], 1.0)
        approximate_count = np.maximum(1.0, np.floor(exposure / avg_amount))
        default_counts = self.default_rng.binomial(approximate_count.astype(np.int64), np.clip(hazard, 0.0, 0.95))
        default_fraction = default_counts / approximate_count
        defaults = exposure * default_fraction
        surviving = np.maximum(exposure - defaults, 0.0)

        remaining_months = (np.arange(self.maturity_bucket_count, dtype=np.float64) + 0.5) * self.bucket_months
        base_principal_rate = np.clip(1.0 / np.maximum(remaining_months, 1.0), 0.0, 1.0)
        repayment_multiplier = self._repayment_multiplier_by_segment()
        principal_rate = base_principal_rate[None, :] * repayment_multiplier[:, None]
        principal_paid = surviving * principal_rate
        post_principal = np.maximum(surviving - principal_paid, 0.0)

        # Monthly interest and carrying costs use beginning-of-month exposure,
        # equivalent to assuming default events settle at month end.
        margin_multiplier = np.exp(-self.fatigue_margin_sensitivity * self.segment_fatigue).clip(0.50, 1.0)
        interest_income = exposure * (self.calibration.annual_interest_rate[:, None] / 12.0) * margin_multiplier[:, None]
        recovery = defaults * self.calibration.recovery_rate[:, None]
        default_loss = defaults - recovery
        funding_cost = exposure * (self.calibration.funding_cost_annual / 12.0)
        servicing_cost = exposure * (self.calibration.servicing_cost_rate / 12.0)

        next_book = np.zeros_like(exposure)
        matured_principal = np.zeros(self.segment_count, dtype=np.float64)
        aging_rate = min(1.0, 1.0 / self.bucket_months)
        segment_aging_rate = np.clip(aging_rate * repayment_multiplier, 0.0, 1.0)
        for h in range(self.maturity_bucket_count):
            stay = post_principal[:, h] * (1.0 - segment_aging_rate)
            advance = post_principal[:, h] * segment_aging_rate
            next_book[:, h] += stay
            if h > 0:
                next_book[:, h - 1] += advance
            else:
                matured_principal += advance
        self.loan_book = next_book

        principal_paid_total = float(principal_paid.sum() + matured_principal.sum())
        gross_cash_inflow = float(principal_paid_total + interest_income.sum() + recovery.sum())
        cash_outflow = float(funding_cost.sum() + servicing_cost.sum())
        cash_inflow = gross_cash_inflow - cash_outflow
        period_profit = float(interest_income.sum() - default_loss.sum() - funding_cost.sum() - servicing_cost.sum())
        return {
            "cash_inflow": cash_inflow,
            "gross_cash_inflow": gross_cash_inflow,
            "cash_outflow": cash_outflow,
            "interest_income": float(interest_income.sum()),
            "principal_paid": principal_paid_total,
            "recovery": float(recovery.sum()),
            "default_loss": float(default_loss.sum()),
            "defaulted_exposure": float(defaults.sum()),
            "funding_cost": float(funding_cost.sum()),
            "servicing_cost": float(servicing_cost.sum()),
            "cash_inflation_cost": 0.0,
            "liquidity_shock_loss": 0.0,
            "funding_shock_cost": 0.0,
            "capital_shock_loss": 0.0,
            "period_profit": period_profit,
        }

    def _allocation_from_actor_action(self, raw: np.ndarray) -> np.ndarray:
        if raw.size == self.segment_count:
            return self._project_allocation(raw)
        if raw.size != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {raw.size}")
        if self.policy_action_mode == "residual":
            return self._residual_allocation_from_actor_action(raw)
        if self.policy_action_mode != "direct":
            raise ValueError(f"Unknown policy_action_mode: {self.policy_action_mode}")
        return self._direct_allocation_from_actor_action(raw)

    def _direct_allocation_from_actor_action(self, raw: np.ndarray) -> np.ndarray:
        deploy_fraction = 1.0 / (1.0 + np.exp(-np.clip(raw[0], -20.0, 20.0)))
        segment_logits = raw[1:] / self.allocation_temperature
        risk_adjusted_epr = self._risk_adjusted_expected_profit_rate()
        segment_logits = segment_logits + self._effective_epr_prior_scale(risk_adjusted_epr) * risk_adjusted_epr
        active_mask = risk_adjusted_epr > self.min_expected_profit_rate
        if 0 < self.allocation_top_k < self.segment_count:
            eligible_logits = np.where(active_mask, segment_logits, -1.0e9)
            top_indices = np.argpartition(eligible_logits, -self.allocation_top_k)[-self.allocation_top_k :]
            top_mask = np.zeros(self.segment_count, dtype=bool)
            top_mask[top_indices] = True
            active_mask = active_mask & top_mask
            if not np.any(active_mask):
                active_mask = risk_adjusted_epr > self.min_expected_profit_rate
        if not np.any(active_mask):
            return np.zeros(self.segment_count, dtype=np.float64)
        target = deploy_fraction * self._deployable_capacity()
        if self.direct_allocation_decoder == "rank_fill":
            return self._allocate_by_priority_scores(target, np.where(active_mask, segment_logits, -np.inf))
        if self.direct_allocation_decoder != "softmax":
            raise ValueError(f"Unknown direct_allocation_decoder: {self.direct_allocation_decoder}")
        weights = _softmax(np.where(active_mask, segment_logits, -1.0e9))
        return self._allocate_by_weights(target, weights, active_mask)

    def _residual_allocation_from_actor_action(self, raw: np.ndarray) -> np.ndarray:
        base_allocation = self._residual_baseline_allocation()
        base_total = float(base_allocation.sum())
        capacity = self._deployable_capacity()
        risk_adjusted_epr = self._risk_adjusted_expected_profit_rate()
        active_mask = risk_adjusted_epr > self.min_expected_profit_rate
        if not np.any(active_mask):
            return np.zeros(self.segment_count, dtype=np.float64)
        if base_total <= 1e-9:
            return np.zeros(self.segment_count, dtype=np.float64)

        if self.residual_use_lend_gate:
            lend_gate = 1.0 / (1.0 + np.exp(-np.clip(raw[0], -20.0, 20.0)))
            lend_gate = 0.0 if lend_gate < self.residual_lend_gate_threshold else lend_gate
            deploy_raw = raw[1]
            residual_raw = raw[2:]
        else:
            lend_gate = 1.0
            deploy_raw = raw[0]
            residual_raw = raw[1:]
        deploy_multiplier = np.exp(
            np.clip(self.residual_deploy_scale * np.tanh(np.clip(deploy_raw, -20.0, 20.0)), -2.0, 2.0)
        )
        target = lend_gate * min(capacity, float(self.current_demand[active_mask].sum()), base_total * deploy_multiplier)
        base_weights = self._residual_base_weights(base_allocation, active_mask, risk_adjusted_epr)
        residual_logits = residual_raw / self.allocation_temperature
        adjusted_logits = (
            np.log(np.clip(base_weights, 1e-8, 1.0))
            + self.residual_scale * residual_logits
            + self._effective_epr_prior_scale(risk_adjusted_epr) * risk_adjusted_epr
        )
        if 0 < self.allocation_top_k < self.segment_count:
            eligible_logits = np.where(active_mask, adjusted_logits, -1.0e9)
            top_indices = np.argpartition(eligible_logits, -self.allocation_top_k)[-self.allocation_top_k :]
            top_mask = np.zeros(self.segment_count, dtype=bool)
            top_mask[top_indices] = True
            active_mask = active_mask & top_mask
            if not np.any(active_mask):
                active_mask = risk_adjusted_epr > self.min_expected_profit_rate
        weights = _softmax(np.where(active_mask, adjusted_logits, -1.0e9))
        return self._allocate_by_weights(target, weights, active_mask)

    def _residual_base_weights(
        self,
        base_allocation: np.ndarray,
        active_mask: np.ndarray,
        risk_adjusted_epr: np.ndarray,
    ) -> np.ndarray:
        base_total = max(float(base_allocation.sum()), 1e-9)
        base_weights = np.asarray(base_allocation, dtype=np.float64) / base_total
        floor = self.residual_base_weight_floor
        if floor <= 0.0:
            return base_weights
        active = np.asarray(active_mask, dtype=bool)
        uniform_prior = np.where(active, 1.0, 0.0)
        uniform_total = float(uniform_prior.sum())
        uniform_prior = uniform_prior / uniform_total if uniform_total > 0.0 else uniform_prior
        epr_prior = np.where(active, np.maximum(risk_adjusted_epr, 0.0) + 1e-4, 0.0)
        epr_total = float(epr_prior.sum())
        prior = epr_prior / epr_total if epr_total > 1e-9 else uniform_prior
        weights = (1.0 - floor) * base_weights + floor * prior
        return weights / max(float(weights.sum()), 1e-9)

    def _effective_epr_prior_scale(self, risk_adjusted_epr: np.ndarray) -> float:
        base = float(self.segment_epr_prior_scale)
        adaptive = float(self.adaptive_epr_prior_scale)
        if adaptive <= 0.0:
            return base
        positive_share = float(np.mean(np.asarray(risk_adjusted_epr) > self.min_expected_profit_rate))
        risk_pressure = float(np.clip((0.65 - positive_share) / 0.65, 0.0, 1.0))
        return base + adaptive * risk_pressure

    def _residual_baseline_allocation(self) -> np.ndarray:
        if self.residual_baseline_policy == "budget_aware_heuristic":
            return budget_aware_segment_policy(self)
        if self.residual_baseline_policy == "fixed_aggressive":
            return fixed_aggressive_policy(self)
        if self.residual_baseline_policy == "fixed_conservative":
            return fixed_conservative_policy(self)
        if self.residual_baseline_policy == "adaptive_lp_conservative":
            return adaptive_lp_conservative_policy(self)
        if self.residual_baseline_policy == "stress_aware_lp_conservative":
            return stress_aware_lp_conservative_policy(self)
        if self.residual_baseline_policy == "epr_aware_lp_conservative":
            return epr_aware_lp_conservative_policy(self)
        if self.residual_baseline_policy == "one_step_lp_closed_form":
            return one_step_lp_closed_form_policy(self)
        if self.residual_baseline_policy == "greedy_expected_profit":
            return greedy_expected_profit_policy(self)
        if self.residual_baseline_policy == "reject_all":
            return reject_all_policy(self)
        raise ValueError(f"Unknown residual_baseline_policy: {self.residual_baseline_policy}")

    def _allocate_by_weights(
        self,
        target: float,
        weights: np.ndarray,
        eligibility_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        allocation = np.zeros(self.segment_count, dtype=np.float64)
        eligible = (
            np.ones(self.segment_count, dtype=bool)
            if eligibility_mask is None
            else np.asarray(eligibility_mask, dtype=bool)
        )
        eligible_demand = np.where(eligible, self.current_demand, 0.0)
        remaining = min(float(target), float(eligible_demand.sum()), self._deployable_capacity())
        active = eligible_demand > 1e-9
        weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
        while remaining > 1e-6 and np.any(active):
            active_weights = np.where(active, weights, 0.0)
            if active_weights.sum() <= 0.0:
                active_weights = active.astype(np.float64)
            proposed = remaining * active_weights / active_weights.sum()
            room = np.maximum(self.current_demand - allocation, 0.0)
            increment = np.minimum(proposed, room)
            if increment.sum() <= 1e-9:
                break
            allocation += increment
            remaining -= float(increment.sum())
            active = (room - increment > 1e-9) & eligible
        return self._project_allocation(allocation)

    def _allocate_by_priority_scores(self, target: float, scores: np.ndarray) -> np.ndarray:
        allocation = np.zeros(self.segment_count, dtype=np.float64)
        remaining = min(float(target), self._deployable_capacity())
        for segment in np.argsort(scores)[::-1]:
            if remaining <= 1e-9:
                break
            if not np.isfinite(scores[segment]):
                continue
            amount = min(float(self.current_demand[segment]), remaining)
            if amount <= 0.0:
                continue
            allocation[segment] = amount
            remaining -= amount
        return self._project_allocation(allocation)

    def _project_allocation(self, allocation: np.ndarray) -> np.ndarray:
        if allocation.size != self.segment_count:
            raise ValueError(f"Expected allocation dimension {self.segment_count}, got {allocation.size}")
        projected = np.minimum(np.maximum(allocation, 0.0), self.current_demand)
        capacity = self._deployable_capacity()
        total = float(projected.sum())
        if total > capacity > 0.0:
            projected *= capacity / total
        elif capacity <= 0.0:
            projected[:] = 0.0
        return projected

    def _deployable_capacity(self) -> float:
        if self.enforce_min_cash_constraint:
            required_before_new_lending = self._required_liquidity()
            return max(0.0, (self.budget - required_before_new_lending) / (1.0 + self.reserve_exposure_ratio))
        return max(0.0, self.budget)

    def _sample_monthly_demand(self) -> np.ndarray:
        index = int(self.demand_rng.integers(0, len(self.calibration.monthly_demand)))
        demand = self.calibration.monthly_demand[index].astype(np.float64, copy=True)
        return demand * self._demand_multiplier_by_segment()

    def _initial_regime_state(self) -> float:
        if self.demand_regime_mode != "ar1" or self.demand_regime_volatility <= 0.0:
            return 0.0
        stationary_scale = self.demand_regime_volatility / max((1.0 - self.demand_regime_rho**2) ** 0.5, 1e-6)
        return float(self.demand_rng.normal(0.0, stationary_scale))

    def _advance_regime(self) -> None:
        if self.demand_regime_mode != "ar1" or self.demand_regime_volatility <= 0.0:
            self.regime_state = 0.0
            self.recovery_opportunity_state = 0.0
            return
        previous = float(self.regime_state)
        innovation = float(self.demand_rng.normal(0.0, self.demand_regime_volatility))
        self.regime_state = float(np.clip(self.demand_regime_rho * self.regime_state + innovation, -2.0, 2.0))
        if self.opportunity_regime:
            release = 0.0
            if previous > self.recovery_stress_threshold and self.regime_state < previous:
                release = max(0.0, previous - max(self.regime_state, 0.0))
            self.recovery_opportunity_state = float(
                np.clip(self.recovery_opportunity_decay * self.recovery_opportunity_state + release, 0.0, 2.0)
            )
        else:
            self.recovery_opportunity_state = 0.0

    def _demand_multiplier_by_segment(self) -> np.ndarray:
        if self.demand_regime_mode != "ar1":
            base = np.ones(self.segment_count, dtype=np.float64)
            return base * np.exp(-self.fatigue_demand_sensitivity * self.segment_fatigue)
        load = 1.0 + self.high_risk_demand_sensitivity * self.segment_risk_rank
        multiplier = np.exp(np.clip(self.regime_state * load, -1.5, 1.5))
        if self.opportunity_regime:
            good = self._good_opportunity_state()
            low_risk_rank = 1.0 - self.segment_risk_rank
            fatigue_ratio = np.clip(self.segment_fatigue / self.fatigue_cap, 0.0, 1.0)
            readiness = np.exp(-self.rotation_opportunity_fatigue_sensitivity * fatigue_ratio)
            readiness *= np.exp(-self.rotation_opportunity_duration_sensitivity * self.segment_duration_pressure)
            high_epr_boost = np.exp(
                np.clip(self.good_opportunity_sensitivity * good * self.segment_epr_rank * readiness, 0.0, 2.0)
            )
            recovery_boost = np.exp(
                np.clip(
                    self.recovery_opportunity_sensitivity
                    * self.recovery_opportunity_state
                    * low_risk_rank
                    * (0.5 + 0.5 * self.segment_epr_rank)
                    * readiness,
                    0.0,
                    2.0,
                )
            )
            multiplier = multiplier * high_epr_boost * recovery_boost
        multiplier = multiplier * np.exp(-self.fatigue_demand_sensitivity * self.segment_fatigue)
        return multiplier

    def _default_hazard_multiplier_by_segment(self) -> np.ndarray:
        if self.demand_regime_mode != "ar1":
            regime_multiplier = np.ones(self.segment_count, dtype=np.float64)
        else:
            load = 1.0 + 0.5 * self.segment_risk_rank
            regime_multiplier = np.exp(np.clip(self.default_regime_sensitivity * self.regime_state * load, -1.0, 1.5))
        stress_cluster = np.exp(
            np.clip(
                self.stress_default_cluster_sensitivity
                * float(self.current_stress_intensity)
                * self.segment_stress_sensitivity,
                0.0,
                2.5,
            )
        )
        fatigue_multiplier = np.exp(np.clip(self.fatigue_default_sensitivity * self.segment_fatigue, 0.0, 2.0))
        return regime_multiplier * stress_cluster * fatigue_multiplier

    def _global_demand_regime_multiplier(self) -> float:
        if self.demand_regime_mode != "ar1":
            return 1.0
        return float(np.exp(np.clip(self.regime_state, -1.5, 1.5)))

    def _global_default_hazard_multiplier(self) -> float:
        if self.demand_regime_mode != "ar1":
            regime_multiplier = 1.0
        else:
            regime_multiplier = float(np.exp(np.clip(self.default_regime_sensitivity * self.regime_state, -1.0, 1.5)))
        stress_multiplier = float(np.exp(np.clip(self.stress_default_cluster_sensitivity * self.current_stress_intensity, 0.0, 2.5)))
        fatigue_multiplier = float(np.exp(np.clip(self.fatigue_default_sensitivity * self.segment_fatigue.mean(), 0.0, 2.0)))
        return regime_multiplier * stress_multiplier * fatigue_multiplier

    def _expected_stress_intensity(self) -> float:
        if self.demand_regime_mode != "ar1" or not self.stress_shock:
            return 0.0
        pressure = self._stress_pressure()
        probability = float(
            np.clip(self.stress_base_probability + self.stress_event_probability * pressure, 0.0, 1.0)
        )
        expected_event = self.stress_event_bonus * probability
        return float(np.clip(max(float(self.current_stress_intensity), pressure + expected_event), 0.0, 2.0))

    def _risk_adjusted_expected_profit_rate(self) -> np.ndarray:
        base = self.calibration.expected_profit_rate.astype(np.float64)
        if not self.use_risk_adjusted_epr:
            return base.copy()

        expected_stress = self._expected_stress_intensity()
        default_multiplier = self._default_hazard_multiplier_by_segment()
        incremental_default_multiplier = np.maximum(default_multiplier - 1.0, 0.0)
        default_cluster_cost = (
            self.risk_adjusted_default_cost_weight
            * self.calibration.loan_pd
            * (1.0 - self.calibration.recovery_rate)
            * incremental_default_multiplier
        )

        shock_exposure_cost = (
            self.liquidity_shock_exposure_frac
            + self.funding_shock_exposure_frac
            + self.capital_shock_exposure_frac
        )
        shock_cost = (
            self.risk_adjusted_shock_cost_weight
            * shock_exposure_cost
            * expected_stress
            * (0.50 + self.segment_stress_sensitivity)
        )

        capital_charge = (
            self.risk_adjusted_capital_charge_weight
            * self.min_capital_ratio
            * self._risk_weight_multiplier_by_segment()
            * (0.50 + 0.50 * self.segment_duration_pressure)
        )

        repayment_delay = np.maximum(1.0 - self._repayment_multiplier_by_segment(), 0.0)
        carry_rate = self.calibration.funding_cost_annual + self.calibration.servicing_cost_rate + 0.02
        duration_years = np.clip(self.calibration.avg_term_months / 12.0, 0.25, 7.0)
        repayment_delay_cost = (
            self.risk_adjusted_repayment_delay_weight
            * repayment_delay
            * carry_rate
            * duration_years
        )

        fatigue_default_cost = (
            self.fatigue_default_sensitivity
            * self.segment_fatigue
            * self.calibration.loan_pd
            * (1.0 - self.calibration.recovery_rate)
        )
        fatigue_margin_cost = (
            self.fatigue_margin_sensitivity
            * self.segment_fatigue
            * self.calibration.annual_interest_rate
            * duration_years
        )
        fatigue_cost = self.risk_adjusted_fatigue_cost_weight * (fatigue_default_cost + fatigue_margin_cost)

        return base - default_cluster_cost - shock_cost - capital_charge - repayment_delay_cost - fatigue_cost

    def _stress_pressure(self) -> float:
        if self.demand_regime_mode != "ar1" or not self.stress_shock:
            return 0.0
        denominator = max(2.0 - self.stress_threshold, 1e-6)
        return float(np.clip((self.regime_state - self.stress_threshold) / denominator, 0.0, 1.0))

    def _sample_stress_intensity(self) -> float:
        pressure = self._stress_pressure()
        if pressure <= 0.0 and self.stress_base_probability <= 0.0:
            return 0.0
        probability = float(np.clip(self.stress_base_probability + self.stress_event_probability * pressure, 0.0, 1.0))
        event = bool(self.stress_rng.random() < probability)
        return float(np.clip(pressure + (self.stress_event_bonus * event), 0.0, 2.0))

    def _repayment_multiplier_by_segment(self) -> np.ndarray:
        duration_drag = self.duration_mismatch_strength * self.segment_duration_pressure
        stress_drag = self.repayment_delay_strength * float(self.current_stress_intensity) * self.segment_stress_sensitivity
        fatigue_drag = (
            self.fatigue_repayment_delay_sensitivity
            * np.clip(self.segment_fatigue / self.fatigue_cap, 0.0, 1.0)
            * (0.50 + 0.50 * self.segment_duration_pressure)
        )
        return np.clip(1.0 - duration_drag - stress_drag - fatigue_drag, 0.05, 1.0)

    def _global_repayment_delay_multiplier(self) -> float:
        exposure_by_segment = np.maximum(self.loan_book, 0.0).sum(axis=1)
        multiplier = self._repayment_multiplier_by_segment()
        total = float(exposure_by_segment.sum())
        if total <= 1e-9:
            return float(multiplier.mean())
        return float(np.dot(exposure_by_segment, multiplier) / total)

    def _risk_weight_multiplier_by_segment(self) -> np.ndarray:
        stress_load = self.stress_risk_weight_inflation * float(self.current_stress_intensity) * self.segment_stress_sensitivity
        fatigue_load = self.fatigue_risk_weight_sensitivity * self.segment_fatigue
        return self.base_risk_weight_multiplier * (1.0 + stress_load + fatigue_load)

    def _global_risk_weight_multiplier(self) -> float:
        exposure_by_segment = np.maximum(self.loan_book, 0.0).sum(axis=1)
        multiplier = self._risk_weight_multiplier_by_segment()
        total = float(exposure_by_segment.sum())
        if total <= 1e-9:
            return float(multiplier.mean())
        return float(np.dot(exposure_by_segment, multiplier) / total)

    def _update_segment_fatigue(self, allocation: np.ndarray) -> None:
        demand_denominator = np.maximum(self.current_demand, self.calibration.avg_loan_amount)
        fill_ratio = np.clip(np.asarray(allocation, dtype=np.float64) / np.maximum(demand_denominator, 1.0), 0.0, 1.0)
        exposure_pressure = np.clip(
            np.maximum(self.loan_book, 0.0).sum(axis=1) / max(self.initial_cash, 1.0),
            0.0,
            2.0,
        )
        disbursement = max(float(np.sum(allocation)), 1.0)
        concentration_pressure = np.clip(np.asarray(allocation, dtype=np.float64) / disbursement, 0.0, 1.0)
        growth = self.fatigue_growth * (
            0.70 * fill_ratio
            + 0.30 * exposure_pressure
            + self.fatigue_concentration_growth * concentration_pressure
        ) * (
            0.50 + self.segment_stress_sensitivity
        )
        self.segment_fatigue = np.clip(
            self.fatigue_decay * self.segment_fatigue + growth,
            0.0,
            self.fatigue_cap,
        )

    def _good_opportunity_state(self) -> float:
        if self.demand_regime_mode != "ar1" or not self.opportunity_regime:
            return 0.0
        return max(0.0, -float(self.regime_state))

    def _global_good_opportunity_multiplier(self) -> float:
        return float(np.exp(np.clip(self.good_opportunity_sensitivity * self._good_opportunity_state(), 0.0, 2.0)))

    def _global_recovery_opportunity_multiplier(self) -> float:
        if self.demand_regime_mode != "ar1" or not self.opportunity_regime:
            return 1.0
        return float(np.exp(np.clip(self.recovery_opportunity_sensitivity * self.recovery_opportunity_state, 0.0, 2.0)))

    def _state(self) -> np.ndarray:
        global_state = self._global_state()
        demand = self.current_demand / max(self.initial_cash, 1.0)
        exposure = self.loan_book.reshape(-1) / max(self.initial_cash, 1.0)
        return np.concatenate([global_state, demand, exposure, self._segment_parameter_state()]).astype(np.float32)

    def _global_state(self) -> np.ndarray:
        scale = max(self.initial_cash, 1.0)
        horizon = max(float(self.horizon_months), 1.0)
        required_liquidity = self._required_liquidity()
        risk_weighted_assets = self._risk_weighted_assets()
        return np.array(
            [
                self.budget / scale,
                self.elapsed / horizon,
                max(self.horizon_months - self.elapsed, 0) / horizon,
                self.capital / scale,
                required_liquidity / scale,
                self._bounded_observation_ratio(self._liquidity_coverage_ratio(required_liquidity)),
                risk_weighted_assets / scale,
                self._bounded_observation_ratio(self._capital_adequacy_ratio()),
                self._global_demand_regime_multiplier() if self.observe_regime_state else 0.0,
                self._global_default_hazard_multiplier() if self.observe_regime_state else 0.0,
                self._global_good_opportunity_multiplier() if self.observe_regime_state else 0.0,
                self._global_recovery_opportunity_multiplier() if self.observe_regime_state else 0.0,
                float(self.current_stress_intensity) if self.observe_regime_state else 0.0,
                self._global_risk_weight_multiplier() if self.observe_regime_state else 0.0,
                self._global_repayment_delay_multiplier() if self.observe_regime_state else 0.0,
                float(self.segment_fatigue.mean()),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _bounded_observation_ratio(value: float) -> float:
        if not np.isfinite(value):
            return OBSERVATION_RATIO_CAP if value >= 0.0 else -OBSERVATION_RATIO_CAP
        return float(np.clip(value, -OBSERVATION_RATIO_CAP, OBSERVATION_RATIO_CAP))

    def _build_static_segment_parameter_state(self) -> np.ndarray:
        params = np.stack(
            [
                np.clip(self.calibration.expected_profit_rate, -1.0, 1.0),
                np.clip(self.segment_stress_sensitivity / 1.75, 0.0, 1.0),
                np.clip(self.segment_duration_pressure / 1.25, 0.0, 1.0),
            ],
            axis=1,
        )
        return params.astype(np.float64)

    def _segment_opportunity_readiness(self) -> np.ndarray:
        fatigue_ratio = self.segment_fatigue / max(self.fatigue_cap, 1e-6)
        readiness = np.exp(
            -self.rotation_opportunity_fatigue_sensitivity * fatigue_ratio
            - self.rotation_opportunity_duration_sensitivity * self.segment_duration_pressure
        )
        return np.clip(readiness, 0.0, 1.0)

    def _segment_parameter_state(self) -> np.ndarray:
        risk_adjusted_epr = np.clip(self._risk_adjusted_expected_profit_rate(), -1.0, 1.0)[:, None]
        opportunity_readiness = self._segment_opportunity_readiness()[:, None]
        fatigue = (self.segment_fatigue / self.fatigue_cap).clip(0.0, 1.0)[:, None]
        return (
            np.concatenate([self.static_segment_parameter_state, risk_adjusted_epr, opportunity_readiness, fatigue], axis=1)
            .reshape(-1)
            .astype(np.float64)
        )

    def _allocation_by_risk(self, allocation: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.segment_risk_bucket,
            weights=np.asarray(allocation, dtype=np.float64),
            minlength=self.calibration.risk_bucket_count,
        ).astype(np.float32)

    def _reset_info(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "capital": self.capital,
            "min_cash": self.min_cash,
            "required_liquidity": self._required_liquidity(),
            "reserve_exposure_ratio": self.reserve_exposure_ratio,
            "residual_use_lend_gate": self.residual_use_lend_gate,
            "residual_lend_gate_threshold": self.residual_lend_gate_threshold,
            "liquidity_coverage_ratio": self._liquidity_coverage_ratio(),
            "risk_weighted_assets": self._risk_weighted_assets(),
            "capital_adequacy_ratio": self._capital_adequacy_ratio(),
            "min_capital_ratio": self.min_capital_ratio,
            "capital_shortfall": self._capital_shortfall(),
            "expected_inflation_annual": self.expected_inflation_annual,
            "monthly_inflation_rate": self.monthly_inflation_rate,
            "segment_epr_prior_scale": self.segment_epr_prior_scale,
            "adaptive_epr_prior_scale": self.adaptive_epr_prior_scale,
            "demand_regime_mode": self.demand_regime_mode,
            "demand_regime_rho": self.demand_regime_rho,
            "demand_regime_volatility": self.demand_regime_volatility,
            "high_risk_demand_sensitivity": self.high_risk_demand_sensitivity,
            "default_regime_sensitivity": self.default_regime_sensitivity,
            "observe_regime_state": self.observe_regime_state,
            "opportunity_regime": self.opportunity_regime,
            "good_opportunity_sensitivity": self.good_opportunity_sensitivity,
            "recovery_opportunity_sensitivity": self.recovery_opportunity_sensitivity,
            "recovery_opportunity_decay": self.recovery_opportunity_decay,
            "recovery_stress_threshold": self.recovery_stress_threshold,
            "stress_shock": self.stress_shock,
            "stress_threshold": self.stress_threshold,
            "stress_base_probability": self.stress_base_probability,
            "stress_event_probability": self.stress_event_probability,
            "stress_event_bonus": self.stress_event_bonus,
            "liquidity_shock_exposure_frac": self.liquidity_shock_exposure_frac,
            "funding_shock_exposure_frac": self.funding_shock_exposure_frac,
            "capital_shock_exposure_frac": self.capital_shock_exposure_frac,
            "base_risk_weight_multiplier": self.base_risk_weight_multiplier,
            "stress_risk_weight_inflation": self.stress_risk_weight_inflation,
            "repayment_delay_strength": self.repayment_delay_strength,
            "duration_mismatch_strength": self.duration_mismatch_strength,
            "stress_default_cluster_sensitivity": self.stress_default_cluster_sensitivity,
            "fatigue_decay": self.fatigue_decay,
            "fatigue_growth": self.fatigue_growth,
            "fatigue_cap": self.fatigue_cap,
            "fatigue_default_sensitivity": self.fatigue_default_sensitivity,
            "fatigue_margin_sensitivity": self.fatigue_margin_sensitivity,
            "fatigue_demand_sensitivity": self.fatigue_demand_sensitivity,
            "fatigue_concentration_growth": self.fatigue_concentration_growth,
            "fatigue_repayment_delay_sensitivity": self.fatigue_repayment_delay_sensitivity,
            "fatigue_risk_weight_sensitivity": self.fatigue_risk_weight_sensitivity,
            "rotation_opportunity_fatigue_sensitivity": self.rotation_opportunity_fatigue_sensitivity,
            "rotation_opportunity_duration_sensitivity": self.rotation_opportunity_duration_sensitivity,
            "use_risk_adjusted_epr": self.use_risk_adjusted_epr,
            "risk_adjusted_default_cost_weight": self.risk_adjusted_default_cost_weight,
            "risk_adjusted_shock_cost_weight": self.risk_adjusted_shock_cost_weight,
            "risk_adjusted_capital_charge_weight": self.risk_adjusted_capital_charge_weight,
            "risk_adjusted_repayment_delay_weight": self.risk_adjusted_repayment_delay_weight,
            "risk_adjusted_fatigue_cost_weight": self.risk_adjusted_fatigue_cost_weight,
            "opportunity_capture_value_weight": self.opportunity_capture_value_weight,
            "strategic_fatigue_cost_weight": self.strategic_fatigue_cost_weight,
            "soft_risk_penalty_lambda": self.soft_risk_penalty_lambda,
            "soft_risk_lcr_target": self.soft_risk_lcr_target,
            "soft_risk_car_target": self.soft_risk_car_target,
            "segment_count": self.segment_count,
            "maturity_bucket_count": self.maturity_bucket_count,
            "calibration_split": self.calibration.name,
            "scenario_name": self.scenario_name,
            "global_state_names": GLOBAL_STATE_NAMES,
            "segment_parameter_names": SEGMENT_PARAMETER_NAMES,
        }


def reject_all_policy(env: SegmentLendingEnv) -> np.ndarray:
    return np.zeros(env.segment_count, dtype=np.float64)


def fixed_conservative_policy(env: SegmentLendingEnv) -> np.ndarray:
    risk_adjusted_epr = env._risk_adjusted_expected_profit_rate()
    scores = np.full(env.segment_count, -np.inf, dtype=np.float64)
    mask = env.segment_risk_bucket <= 1
    scores[mask] = risk_adjusted_epr[mask] + 0.05
    return allocate_by_scores(env, scores, deploy_fraction=0.55)


def fixed_aggressive_policy(env: SegmentLendingEnv) -> np.ndarray:
    risk_adjusted_epr = env._risk_adjusted_expected_profit_rate()
    risk_weights = np.array([0.40, 0.30, 0.20, 0.10, 0.0], dtype=np.float64)
    scores = risk_weights[env.segment_risk_bucket] + 0.01 * risk_adjusted_epr
    return allocate_by_scores(env, scores, deploy_fraction=0.90)


def adaptive_lp_conservative_policy(env: SegmentLendingEnv) -> np.ndarray:
    """Use LP carry in calm states and conservative allocation under observed stress."""

    lp = one_step_lp_closed_form_policy(env)
    conservative = fixed_conservative_policy(env)
    stress = float(np.clip(getattr(env, "current_stress_intensity", 0.0), 0.0, 2.0))
    stress_weight = float(np.clip(stress / 0.35, 0.0, 1.0))
    return (1.0 - stress_weight) * lp + stress_weight * conservative


def stress_aware_lp_conservative_policy(env: SegmentLendingEnv) -> np.ndarray:
    """Use LP ranking only in explicitly no-stress environments.

    In the stress-enabled MDP, the residual actor receives a conservative base
    and must learn any extra deployment. In explicit no-stress evaluation or
    mixed no-stress training batches, it starts from LP carry so ordinary-regime
    upside is not hidden behind a conservative segment mask.
    """

    if not bool(getattr(env, "stress_shock", True)):
        return one_step_lp_closed_form_policy(env)
    return fixed_conservative_policy(env)


def epr_aware_lp_conservative_policy(env: SegmentLendingEnv) -> np.ndarray:
    """Use LP in no-stress high-carry universes and conservative otherwise."""

    if bool(getattr(env, "stress_shock", True)):
        return fixed_conservative_policy(env)
    risk_adjusted_epr = env._risk_adjusted_expected_profit_rate()
    demand = np.maximum(env.current_demand, 0.0)
    demand_total = float(demand.sum())
    if demand_total <= 1e-9:
        return reject_all_policy(env)
    demand_weighted_epr = float(np.dot(demand, risk_adjusted_epr) / demand_total)
    if demand_weighted_epr >= float(getattr(env, "epr_aware_lp_threshold", 0.09)):
        return one_step_lp_closed_form_policy(env)
    return fixed_conservative_policy(env)


def greedy_expected_profit_policy(env: SegmentLendingEnv) -> np.ndarray:
    risk_adjusted_epr = env._risk_adjusted_expected_profit_rate()
    scores = np.where(risk_adjusted_epr > 0.0, risk_adjusted_epr, -np.inf)
    return allocate_by_scores(env, scores, deploy_fraction=1.0)


def one_step_lp_closed_form_policy(env: SegmentLendingEnv) -> np.ndarray:
    """Closed-form solution of the one-period linear allocation problem.

    With linear per-dollar profit coefficients and a single budget constraint,
    the myopic LP is solved by sorting segments by expected profit rate and
    filling feasible demand greedily.
    """

    risk_adjusted_epr = env._risk_adjusted_expected_profit_rate()
    scores = np.where(risk_adjusted_epr > 0.0, risk_adjusted_epr, -np.inf)
    return allocate_by_scores(env, scores, deploy_fraction=1.0)


def one_step_lp_policy(env: SegmentLendingEnv) -> np.ndarray:
    return one_step_lp_closed_form_policy(env)


def budget_aware_segment_policy(env: SegmentLendingEnv) -> np.ndarray:
    risk_adjusted_epr = env._risk_adjusted_expected_profit_rate()
    reserve_ratio = env.budget / max(env._required_liquidity(), 1.0)
    scores = np.full(env.segment_count, -np.inf, dtype=np.float64)
    max_risk = 1 if reserve_ratio <= 1.2 else 2 if reserve_ratio <= 2.0 else 3
    deploy_fraction = 0.35 if reserve_ratio <= 1.2 else 0.65 if reserve_ratio <= 2.0 else 0.90
    mask = (env.segment_risk_bucket <= max_risk) & (risk_adjusted_epr > 0.0)
    scores[mask] = (
        risk_adjusted_epr[mask]
        + (max_risk - env.segment_risk_bucket[mask]) * 0.01
    )
    return allocate_by_scores(env, scores, deploy_fraction=deploy_fraction)


def allocate_by_scores(env: SegmentLendingEnv, scores: np.ndarray, deploy_fraction: float = 1.0) -> np.ndarray:
    required_before_new_lending = env._required_liquidity()
    safe_capacity = max(0.0, (env.budget - required_before_new_lending) / (1.0 + env.reserve_exposure_ratio))
    safe_capacity *= 0.99  # 1% buffer for intra-step inflation drag and negative cash flows
    capacity = safe_capacity * float(deploy_fraction)
    allocation = np.zeros(env.segment_count, dtype=np.float64)
    for segment in np.argsort(scores)[::-1]:
        if capacity <= 1e-9:
            break
        if not np.isfinite(scores[segment]):
            continue
        amount = min(float(env.current_demand[segment]), capacity)
        if amount <= 0.0:
            continue
        allocation[segment] = amount
        capacity -= amount
    return env._project_allocation(allocation)


def evaluate_segment_policy(
    env_factory: Callable[[int], SegmentLendingEnv],
    policy_fn: Callable[[SegmentLendingEnv, np.ndarray], np.ndarray] | None = None,
    actor_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    episodes: int = 100,
    seed: int = 1_000,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    allocation_rows = []
    for episode in range(int(episodes)):
        env = env_factory(seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        while True:
            if actor_fn is not None:
                action = actor_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                if policy_fn is None:
                    raise ValueError("Either policy_fn or actor_fn must be provided.")
                allocation = policy_fn(env, obs)
                obs, reward, terminated, truncated, info = env.step_allocation(allocation)
            total_reward += float(reward)
            if terminated or truncated:
                break

        trace = pd.DataFrame(env.trace)
        min_budget_source = "min_budget_observed" if not trace.empty and "min_budget_observed" in trace.columns else "budget"
        min_budget = float(trace[min_budget_source].min()) if not trace.empty else env.budget
        final_budget = float(env.budget)
        final_capital = float(env.capital)
        ending_exposure = float(env.loan_book.sum())
        profit = float(trace["period_profit"].sum()) if not trace.empty else 0.0
        terminal_runoff_profit = float(trace["terminal_runoff_profit"].sum()) if not trace.empty else 0.0
        terminal_value = float(trace["terminal_value"].sum()) if not trace.empty else 0.0
        if not trace.empty and "decision_period_profit" in trace.columns:
            decision_period_profit = float(trace["decision_period_profit"].sum())
        else:
            decision_period_profit = profit - terminal_runoff_profit - terminal_value
        if not trace.empty and "liquidity_breach_count" in trace.columns:
            breach_count = int(trace["liquidity_breach_count"].sum())
        else:
            breach_count = int(trace["liquidity_breach"].sum()) if not trace.empty else 0
        if not trace.empty and "max_liquidity_shortfall" in trace.columns:
            shortfall = trace["max_liquidity_shortfall"].to_numpy(dtype=np.float64)
        elif not trace.empty and "liquidity_shortfall" in trace.columns:
            shortfall = trace["liquidity_shortfall"].to_numpy(dtype=np.float64)
        else:
            shortfall = (
                np.maximum(env.min_cash - trace["budget"].to_numpy(dtype=np.float64), 0.0)
                if not trace.empty
                else np.array([0.0])
            )
        liquidity_ratios = (
            trace["liquidity_coverage_ratio"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
            if not trace.empty and "liquidity_coverage_ratio" in trace.columns
            else np.array([], dtype=np.float64)
        )
        min_liquidity_coverage_ratio = float(liquidity_ratios.min()) if liquidity_ratios.size else float("inf")
        disbursement = float(trace["disbursement"].sum()) if not trace.empty else 0.0
        default_loss = float(trace["default_loss"].sum()) if not trace.empty else 0.0
        funding_cost = float(trace["funding_cost"].sum()) if not trace.empty else 0.0
        servicing_cost = float(trace["servicing_cost"].sum()) if not trace.empty else 0.0
        cash_inflation_cost = float(trace["cash_inflation_cost"].sum()) if not trace.empty else 0.0
        expected_deployment_profit = (
            float(trace["expected_deployment_profit"].sum())
            if not trace.empty and "expected_deployment_profit" in trace.columns
            else 0.0
        )
        risk_adjusted_deployment_profit = (
            float(trace["risk_adjusted_deployment_profit"].sum())
            if not trace.empty and "risk_adjusted_deployment_profit" in trace.columns
            else 0.0
        )
        expected_profit_shaping_reward = (
            float(trace["expected_profit_shaping_reward"].sum())
            if not trace.empty and "expected_profit_shaping_reward" in trace.columns
            else 0.0
        )
        lcr_buffer_violation = (
            float(trace["lcr_buffer_violation"].max())
            if not trace.empty and "lcr_buffer_violation" in trace.columns
            else 0.0
        )
        car_buffer_violation = (
            float(trace["car_buffer_violation"].max())
            if not trace.empty and "car_buffer_violation" in trace.columns
            else 0.0
        )
        soft_risk_penalty = (
            float(trace["soft_risk_penalty"].sum())
            if not trace.empty and "soft_risk_penalty" in trace.columns
            else 0.0
        )
        mean_stress_intensity = (
            float(trace["stress_intensity"].mean())
            if not trace.empty and "stress_intensity" in trace.columns
            else 0.0
        )
        capital_shortfall = (
            trace["capital_shortfall"].to_numpy(dtype=np.float64)
            if not trace.empty and "capital_shortfall" in trace.columns
            else np.array([0.0])
        )
        capital_breach_count = (
            int(trace["capital_breach"].sum()) if not trace.empty and "capital_breach" in trace.columns else 0
        )
        capital_ratios = (
            trace["capital_adequacy_ratio"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
            if not trace.empty and "capital_adequacy_ratio" in trace.columns
            else np.array([], dtype=np.float64)
        )
        min_capital_adequacy_ratio = float(capital_ratios.min()) if capital_ratios.size else float("inf")
        liquidated = float(trace["liquidated"].any()) if not trace.empty and "liquidated" in trace.columns else 0.0
        capital_liquidation = (
            float(trace["capital_liquidation"].any())
            if not trace.empty and "capital_liquidation" in trace.columns
            else 0.0
        )
        liquidity_liquidation = (
            float(trace["liquidity_liquidation"].any())
            if not trace.empty and "liquidity_liquidation" in trace.columns
            else 0.0
        )
        liquidation_trigger = "none"
        if not trace.empty and "liquidation_trigger" in trace.columns:
            triggers = [value for value in trace["liquidation_trigger"].astype(str).tolist() if value != "none"]
            liquidation_trigger = triggers[0] if triggers else "none"
        liquidation_loss = float(trace["liquidation_loss"].sum()) if not trace.empty and "liquidation_loss" in trace.columns else 0.0
        liquidation_proceeds = (
            float(trace["liquidation_proceeds"].sum())
            if not trace.empty and "liquidation_proceeds" in trace.columns
            else 0.0
        )
        liquidation_asset_base = (
            float(trace["liquidation_asset_base"].sum())
            if not trace.empty and "liquidation_asset_base" in trace.columns
            else 0.0
        )
        gross_cash_inflow = float(trace["gross_cash_inflow"].sum()) if not trace.empty else 0.0
        cash_outflow = float(trace["cash_outflow"].sum()) if not trace.empty else 0.0
        cash_inflow = float(trace["cash_inflow"].sum()) if not trace.empty else 0.0
        decision_disbursements = (
            trace["disbursement"].to_numpy(dtype=np.float64)
            if not trace.empty and "disbursement" in trace.columns
            else np.array([], dtype=np.float64)
        )
        final_decision_disbursement = float(decision_disbursements[-1]) if decision_disbursements.size else 0.0
        preterminal_disbursements = decision_disbursements[:-1]
        mean_preterminal_disbursement = (
            float(preterminal_disbursements.mean()) if preterminal_disbursements.size else 0.0
        )
        final_disbursement_ratio = final_decision_disbursement / max(mean_preterminal_disbursement, 1.0)
        final_disbursement_share = final_decision_disbursement / max(disbursement, 1.0)
        rows.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "cumulative_profit": profit,
                "decision_period_profit": decision_period_profit,
                "terminal_runoff_profit": terminal_runoff_profit,
                "terminal_value": terminal_value,
                "terminal_budget": final_budget,
                "terminal_capital": final_capital,
                "ending_exposure": ending_exposure,
                "min_budget": min_budget,
                "liquidity_breach": float(breach_count > 0),
                "liquidity_breach_count": breach_count,
                "min_liquidity_coverage_ratio": min_liquidity_coverage_ratio,
                "capital_breach": float(capital_breach_count > 0),
                "capital_breach_count": capital_breach_count,
                "capital_shortfall": float(capital_shortfall.max()),
                "min_capital_adequacy_ratio": min_capital_adequacy_ratio,
                "expected_shortfall": float(shortfall.mean()),
                "max_shortfall": float(shortfall.max()),
                "default_loss": default_loss,
                "funding_cost": funding_cost,
                "servicing_cost": servicing_cost,
                "cash_inflation_cost": cash_inflation_cost,
                "expected_deployment_profit": expected_deployment_profit,
                "risk_adjusted_deployment_profit": risk_adjusted_deployment_profit,
                "expected_profit_shaping_reward": expected_profit_shaping_reward,
                "lcr_buffer_violation": lcr_buffer_violation,
                "car_buffer_violation": car_buffer_violation,
                "soft_risk_penalty": soft_risk_penalty,
                "mean_stress_intensity": mean_stress_intensity,
                "liquidated": liquidated,
                "capital_liquidation": capital_liquidation,
                "liquidity_liquidation": liquidity_liquidation,
                "liquidation_trigger": liquidation_trigger,
                "liquidation_asset_base": liquidation_asset_base,
                "liquidation_loss": liquidation_loss,
                "liquidation_proceeds": liquidation_proceeds,
                "gross_cash_inflow": gross_cash_inflow,
                "cash_outflow": cash_outflow,
                "cash_inflow": cash_inflow,
                "disbursement": disbursement,
                "final_decision_disbursement": final_decision_disbursement,
                "mean_preterminal_disbursement": mean_preterminal_disbursement,
                "final_disbursement_ratio": final_disbursement_ratio,
                "final_disbursement_share": final_disbursement_share,
                "capital_utilization": disbursement / max(env.initial_cash * env.horizon_months, 1.0),
                "max_drawdown": max(0.0, env.initial_cash - min_budget),
            }
        )
        if not trace.empty:
            allocations = np.stack(trace["allocation"].to_list())
            allocation_rows.append(allocations.mean(axis=0))

    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for column in frame.columns:
        if column in {"episode", "liquidation_trigger"}:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        summary[f"{column}_mean"] = float(series.mean()) if not series.dropna().empty else float("nan")
        summary[f"{column}_std"] = float(series.std(ddof=0)) if not series.dropna().empty else float("nan")
    summary["episode_count"] = int(len(frame))
    if "cumulative_profit" in frame.columns:
        profit_series = pd.to_numeric(frame["cumulative_profit"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not profit_series.empty:
            ordered_profit = profit_series.sort_values()
            cvar_count = max(1, int(np.ceil(0.05 * len(ordered_profit))))
            worst_decile_count = max(1, int(np.ceil(0.10 * len(ordered_profit))))
            summary["cumulative_profit_cvar_5_mean"] = float(ordered_profit.iloc[:cvar_count].mean())
            summary["cumulative_profit_worst_decile_mean"] = float(ordered_profit.iloc[:worst_decile_count].mean())
            summary["cumulative_profit_p05"] = float(profit_series.quantile(0.05))
            summary["cumulative_profit_p10"] = float(profit_series.quantile(0.10))
    if "mean_stress_intensity" in frame.columns and "cumulative_profit" in frame.columns:
        stress_series = pd.to_numeric(frame["mean_stress_intensity"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        profit_series = pd.to_numeric(frame["cumulative_profit"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        stress_profit = profit_series[stress_series > 0.10].dropna()
        summary["stress_regime_episode_count"] = int(stress_profit.shape[0])
        summary["stress_regime_profit_mean"] = float(stress_profit.mean()) if not stress_profit.empty else float("nan")
    if "liquidated" in frame.columns:
        summary["liquidated_episode_count"] = int(frame["liquidated"].sum())
    if "capital_liquidation" in frame.columns:
        summary["capital_liquidation_episode_count"] = int(frame["capital_liquidation"].sum())
    if "liquidity_liquidation" in frame.columns:
        summary["liquidity_liquidation_episode_count"] = int(frame["liquidity_liquidation"].sum())
    if "capital_breach" in frame.columns:
        summary["capital_breach_episode_count"] = int(frame["capital_breach"].sum())
    if "liquidity_breach" in frame.columns:
        summary["liquidity_breach_episode_count"] = int(frame["liquidity_breach"].sum())
    if allocation_rows:
        mean_allocation = np.mean(np.stack(allocation_rows), axis=0)
        for risk_bucket in range(env_factory(seed).calibration.risk_bucket_count):
            total = 0.0
            for segment, amount in enumerate(mean_allocation):
                decoded_risk, _, _ = env_factory(seed).calibration.decode_segment(segment)
                if decoded_risk == risk_bucket:
                    total += float(amount)
            summary[f"mean_allocation_risk_{risk_bucket}"] = total
    return summary, frame


def _quantile_edges(series: pd.Series, bucket_count: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bucket_count + 1)
    values = series.astype(float).quantile(quantiles).to_numpy(dtype=np.float64).copy()
    values[0] = -np.inf
    values[-1] = np.inf
    for i in range(1, len(values) - 1):
        if values[i] <= values[i - 1]:
            values[i] = np.nextafter(values[i - 1], np.inf)
    return values


def _bucketize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], values, side="right").clip(0, len(edges) - 2).astype(np.int64)


def _segment_id(
    risk_bucket: np.ndarray,
    amount_tier: np.ndarray,
    term_bucket: np.ndarray,
    amount_tier_count: int,
    term_bucket_count: int,
) -> np.ndarray:
    return ((risk_bucket * amount_tier_count + amount_tier) * term_bucket_count + term_bucket).astype(np.int64)


def _segment_labels(risk_count: int, amount_count: int, term_count: int) -> list[str]:
    labels = []
    for risk in range(risk_count):
        for amount in range(amount_count):
            for term in range(term_count):
                term_name = "36m" if term == 0 else "60m"
                labels.append(f"risk{risk}_amount{amount}_{term_name}")
    return labels


def _expected_profit_rate(
    loan_pd: np.ndarray,
    annual_interest_rate: np.ndarray,
    avg_term_months: np.ndarray,
    recovery_rate: np.ndarray,
    funding_cost_annual: float,
    servicing_cost_rate: float,
    maturity_bucket_count: int = 6,
    runoff_months: int = 120,
) -> np.ndarray:
    """Simulator-consistent expected lifetime profit per dollar of new lending.

    The simulator does not charge lifetime loss on the original principal in
    one shot. It amortizes exposure monthly, samples defaults from the
    remaining loan book, charges carrying costs on beginning-of-month exposure,
    and ages maturity buckets. This deterministic counterpart uses the same
    accounting with expected defaults, so heuristic scores and PPO observations
    see the same segment economics as rollout evaluation.
    """
    loan_pd = np.asarray(loan_pd, dtype=np.float64)
    annual_interest_rate = np.asarray(annual_interest_rate, dtype=np.float64)
    avg_term_months = np.clip(np.asarray(avg_term_months, dtype=np.float64), 1.0, 84.0)
    recovery_rate = np.asarray(recovery_rate, dtype=np.float64)

    segment_count = int(loan_pd.shape[0])
    bucket_count = int(max(1, maturity_bucket_count))
    max_term = max(60.0, float(np.nanmax(avg_term_months)))
    bucket_months = max(1.0, max_term / bucket_count)
    remaining_months = (np.arange(bucket_count, dtype=np.float64) + 0.5) * bucket_months
    principal_rate = np.clip(1.0 / np.maximum(remaining_months, 1.0), 0.0, 1.0)
    aging_rate = min(1.0, 1.0 / bucket_months)

    monthly_hazard = 1.0 - np.power(1.0 - np.clip(loan_pd, 0.001, 0.95), 1.0 / avg_term_months)
    new_loan_bucket = np.array(
        [_term_to_bucket(float(term), bucket_count, max_term) for term in avg_term_months],
        dtype=np.int64,
    )

    book = np.zeros((segment_count, bucket_count), dtype=np.float64)
    book[np.arange(segment_count), new_loan_bucket] = 1.0
    profit = np.zeros(segment_count, dtype=np.float64)

    hazard = np.clip(monthly_hazard, 0.0, 0.95)[:, None]
    annual_rate = annual_interest_rate[:, None]
    recovery = recovery_rate[:, None]
    funding_rate = float(funding_cost_annual) / 12.0
    servicing_rate = float(servicing_cost_rate) / 12.0

    for _ in range(int(max(runoff_months, max_term))):
        if not np.any(book > 1e-10):
            break
        defaults = book * hazard
        surviving = np.maximum(book - defaults, 0.0)
        principal_paid = surviving * principal_rate[None, :]
        post_principal = np.maximum(surviving - principal_paid, 0.0)

        # Match simulator timing: defaults settle at month end, so interest and
        # carrying costs use beginning-of-month exposure.
        interest_income = book * (annual_rate / 12.0)
        recovered = defaults * recovery
        default_loss = defaults - recovered
        funding_cost = book * funding_rate
        servicing_cost = book * servicing_rate
        period_profit = interest_income - default_loss - funding_cost - servicing_cost
        profit += period_profit.sum(axis=1)

        next_book = np.zeros_like(book)
        next_book[:, 0] += post_principal[:, 0] * (1.0 - aging_rate)
        for h in range(1, bucket_count):
            stay = post_principal[:, h] * (1.0 - aging_rate)
            advance = post_principal[:, h] * aging_rate
            next_book[:, h] += stay
            next_book[:, h - 1] += advance
        book = next_book

    return profit


def _term_to_bucket(term_months: float, bucket_count: int, max_term_months: float) -> int:
    bucket_width = max(1.0, max_term_months / bucket_count)
    return int(np.clip(np.ceil(term_months / bucket_width) - 1, 0, bucket_count - 1))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    total = exp.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(values, 1.0 / len(values), dtype=np.float64)
    return exp / total

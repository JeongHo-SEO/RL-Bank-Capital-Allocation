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
    "period_profit",
)

SEGMENT_PARAMETER_NAMES = (
    "loan_pd",
    "annual_interest_rate",
    "recovery_rate",
    "expected_profit_rate",
)
SEGMENT_PARAMETER_COUNT = len(SEGMENT_PARAMETER_NAMES)


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
        expected_profit_shaping_weight: float = 0.0,
        allocation_top_k: int = 0,
        min_expected_profit_rate: float = -1.0,
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
        self.expected_profit_shaping_weight = float(expected_profit_shaping_weight)
        self.allocation_top_k = max(0, int(allocation_top_k))
        self.min_expected_profit_rate = float(min_expected_profit_rate)
        self.scenario_name = scenario_name or calibration.metadata.get("stress_name") or calibration.name
        self.seed_value = int(seed)
        self._reset_rngs(self.seed_value)
        self.segment_count = calibration.segment_count
        self.maturity_bucket_count = calibration.maturity_bucket_count
        self.segment_indices = np.arange(self.segment_count, dtype=np.int64)
        self.segment_new_loan_bucket = np.asarray(calibration.new_loan_bucket, dtype=np.int64)
        self.segment_risk_bucket = np.array(
            [calibration.decode_segment(segment)[0] for segment in range(self.segment_count)],
            dtype=np.int64,
        )
        self.bucket_months = max(1.0, float(np.nanmax(calibration.avg_term_months)) / self.maturity_bucket_count)
        self.segment_parameter_state = self._build_segment_parameter_state()
        self.initial_loan_book = (
            np.asarray(initial_loan_book, dtype=np.float64).copy()
            if initial_loan_book is not None
            else np.zeros((self.segment_count, self.maturity_bucket_count), dtype=np.float64)
        )

        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.segment_count + 1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                1
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
        expected_deployment_profit = float(np.dot(allocation, self.calibration.expected_profit_rate))
        expected_profit_shaping_reward = (
            self.expected_profit_shaping_weight * expected_deployment_profit / max(self.initial_cash, 1.0)
        )

        required_liquidity = self._required_liquidity()
        shortfall, immediate_liquidity_penalty = self._liquidity_penalty()
        terminal_metrics = self._empty_terminal_metrics()
        metrics = dict(book_metrics)
        reward = (
            float(book_metrics["period_profit"]) / max(self.initial_cash, 1.0)
            + expected_profit_shaping_reward
            - immediate_liquidity_penalty
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
            "expected_profit_shaping_reward": expected_profit_shaping_reward,
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
        return float(np.maximum(self.loan_book, 0.0).sum())

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

        hazard = self.calibration.monthly_default_hazard[:, None]
        avg_amount = np.maximum(self.calibration.avg_loan_amount[:, None], 1.0)
        approximate_count = np.maximum(1.0, np.floor(exposure / avg_amount))
        default_counts = self.default_rng.binomial(approximate_count.astype(np.int64), np.clip(hazard, 0.0, 0.95))
        default_fraction = default_counts / approximate_count
        defaults = exposure * default_fraction
        surviving = np.maximum(exposure - defaults, 0.0)

        remaining_months = (np.arange(self.maturity_bucket_count, dtype=np.float64) + 0.5) * self.bucket_months
        principal_rate = np.clip(1.0 / np.maximum(remaining_months, 1.0), 0.0, 1.0)
        principal_paid = surviving * principal_rate[None, :]
        post_principal = np.maximum(surviving - principal_paid, 0.0)

        # Monthly interest and carrying costs use beginning-of-month exposure,
        # equivalent to assuming default events settle at month end.
        interest_income = exposure * (self.calibration.annual_interest_rate[:, None] / 12.0)
        recovery = defaults * self.calibration.recovery_rate[:, None]
        default_loss = defaults - recovery
        funding_cost = exposure * (self.calibration.funding_cost_annual / 12.0)
        servicing_cost = exposure * (self.calibration.servicing_cost_rate / 12.0)

        next_book = np.zeros_like(exposure)
        matured_principal = np.zeros(self.segment_count, dtype=np.float64)
        aging_rate = min(1.0, 1.0 / self.bucket_months)
        for h in range(self.maturity_bucket_count):
            stay = post_principal[:, h] * (1.0 - aging_rate)
            advance = post_principal[:, h] * aging_rate
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
            "period_profit": period_profit,
        }

    def _allocation_from_actor_action(self, raw: np.ndarray) -> np.ndarray:
        if raw.size == self.segment_count:
            return self._project_allocation(raw)
        if raw.size != self.segment_count + 1:
            raise ValueError(f"Expected action dimension {self.segment_count + 1}, got {raw.size}")
        deploy_fraction = 1.0 / (1.0 + np.exp(-np.clip(raw[0], -20.0, 20.0)))
        segment_logits = raw[1:] / self.allocation_temperature
        segment_logits = segment_logits + self.segment_epr_prior_scale * self.calibration.expected_profit_rate
        active_mask = self.calibration.expected_profit_rate > self.min_expected_profit_rate
        if 0 < self.allocation_top_k < self.segment_count:
            eligible_logits = np.where(active_mask, segment_logits, -1.0e9)
            top_indices = np.argpartition(eligible_logits, -self.allocation_top_k)[-self.allocation_top_k :]
            top_mask = np.zeros(self.segment_count, dtype=bool)
            top_mask[top_indices] = True
            active_mask = active_mask & top_mask
            if not np.any(active_mask):
                active_mask = self.calibration.expected_profit_rate > self.min_expected_profit_rate
        weights = _softmax(np.where(active_mask, segment_logits, -1.0e9))
        capacity = self._deployable_capacity()
        return self._allocate_by_weights(deploy_fraction * capacity, weights)

    def _allocate_by_weights(self, target: float, weights: np.ndarray) -> np.ndarray:
        allocation = np.zeros(self.segment_count, dtype=np.float64)
        remaining = min(float(target), float(self.current_demand.sum()), self._deployable_capacity())
        active = self.current_demand > 1e-9
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
            active = room - increment > 1e-9
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
        return self.calibration.monthly_demand[index].astype(np.float64, copy=True)

    def _state(self) -> np.ndarray:
        cash = np.array([self.budget / max(self.initial_cash, 1.0)], dtype=np.float64)
        demand = self.current_demand / max(self.initial_cash, 1.0)
        exposure = self.loan_book.reshape(-1) / max(self.initial_cash, 1.0)
        return np.concatenate([cash, demand, exposure, self.segment_parameter_state]).astype(np.float32)

    def _build_segment_parameter_state(self) -> np.ndarray:
        params = np.stack(
            [
                np.clip(self.calibration.loan_pd, 0.0, 1.0),
                np.clip(self.calibration.annual_interest_rate, 0.0, 1.0),
                np.clip(self.calibration.recovery_rate, 0.0, 1.0),
                np.clip(self.calibration.expected_profit_rate, -1.0, 1.0),
            ],
            axis=1,
        )
        return params.reshape(-1).astype(np.float64)

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
            "liquidity_coverage_ratio": self._liquidity_coverage_ratio(),
            "risk_weighted_assets": self._risk_weighted_assets(),
            "capital_adequacy_ratio": self._capital_adequacy_ratio(),
            "min_capital_ratio": self.min_capital_ratio,
            "capital_shortfall": self._capital_shortfall(),
            "expected_inflation_annual": self.expected_inflation_annual,
            "monthly_inflation_rate": self.monthly_inflation_rate,
            "segment_count": self.segment_count,
            "maturity_bucket_count": self.maturity_bucket_count,
            "calibration_split": self.calibration.name,
            "scenario_name": self.scenario_name,
            "segment_parameter_names": SEGMENT_PARAMETER_NAMES,
        }


def reject_all_policy(env: SegmentLendingEnv) -> np.ndarray:
    return np.zeros(env.segment_count, dtype=np.float64)


def fixed_conservative_policy(env: SegmentLendingEnv) -> np.ndarray:
    scores = np.full(env.segment_count, -np.inf, dtype=np.float64)
    mask = env.segment_risk_bucket <= 1
    scores[mask] = env.calibration.expected_profit_rate[mask] + 0.05
    return allocate_by_scores(env, scores, deploy_fraction=0.55)


def fixed_aggressive_policy(env: SegmentLendingEnv) -> np.ndarray:
    risk_weights = np.array([0.40, 0.30, 0.20, 0.10, 0.0], dtype=np.float64)
    scores = risk_weights[env.segment_risk_bucket] + 0.01 * env.calibration.expected_profit_rate
    return allocate_by_scores(env, scores, deploy_fraction=0.90)


def greedy_expected_profit_policy(env: SegmentLendingEnv) -> np.ndarray:
    scores = np.where(env.calibration.expected_profit_rate > 0.0, env.calibration.expected_profit_rate, -np.inf)
    return allocate_by_scores(env, scores, deploy_fraction=1.0)


def one_step_lp_closed_form_policy(env: SegmentLendingEnv) -> np.ndarray:
    """Closed-form solution of the one-period linear allocation problem.

    With linear per-dollar profit coefficients and a single budget constraint,
    the myopic LP is solved by sorting segments by expected profit rate and
    filling feasible demand greedily.
    """

    scores = np.where(env.calibration.expected_profit_rate > 0.0, env.calibration.expected_profit_rate, -np.inf)
    return allocate_by_scores(env, scores, deploy_fraction=1.0)


def one_step_lp_policy(env: SegmentLendingEnv) -> np.ndarray:
    return one_step_lp_closed_form_policy(env)


def budget_aware_segment_policy(env: SegmentLendingEnv) -> np.ndarray:
    reserve_ratio = env.budget / max(env._required_liquidity(), 1.0)
    scores = np.full(env.segment_count, -np.inf, dtype=np.float64)
    max_risk = 1 if reserve_ratio <= 1.2 else 2 if reserve_ratio <= 2.0 else 3
    deploy_fraction = 0.35 if reserve_ratio <= 1.2 else 0.65 if reserve_ratio <= 2.0 else 0.90
    mask = env.segment_risk_bucket <= max_risk
    scores[mask] = (
        env.calibration.expected_profit_rate[mask]
        + (max_risk - env.segment_risk_bucket[mask]) * 0.01
    )
    return allocate_by_scores(env, scores, deploy_fraction=deploy_fraction)


def allocate_by_scores(env: SegmentLendingEnv, scores: np.ndarray, deploy_fraction: float = 1.0) -> np.ndarray:
    required_before_new_lending = env._required_liquidity()
    safe_capacity = max(0.0, (env.budget - required_before_new_lending) / (1.0 + env.reserve_exposure_ratio))
    safe_capacity *= 0.99  # 1% 버퍼
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
        expected_profit_shaping_reward = (
            float(trace["expected_profit_shaping_reward"].sum())
            if not trace.empty and "expected_profit_shaping_reward" in trace.columns
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
                "expected_profit_shaping_reward": expected_profit_shaping_reward,
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

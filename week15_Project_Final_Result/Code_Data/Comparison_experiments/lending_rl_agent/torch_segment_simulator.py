from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .segment_simulator import SEGMENT_PARAMETER_COUNT, SEGMENT_PARAMETER_NAMES, SegmentCalibration


BOOK_KEYS = (
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


@dataclass
class TorchBatchConfig:
    initial_cash: float
    min_cash_ratio: float
    horizon_months: int
    liquidity_penalty_lambda: float
    reserve_exposure_ratio: float
    terminal_runoff_months: int
    terminal_runoff_discount: float
    terminal_value_weight: float
    expected_inflation_annual: float
    min_capital_ratio: float
    liquidity_liquidation_haircut: float
    enforce_min_cash_constraint: bool = False
    default_mode: str = "normal"
    min_expected_profit_rate: float = 0.0
    negative_epr_penalty: float = 0.0
    allocation_temperature: float = 1.0
    segment_epr_prior_scale: float = 0.0
    expected_profit_shaping_weight: float = 0.0
    allocation_top_k: int = 0


class TorchSegmentBatchEnv:
    """Vectorized segment-lending simulator that keeps rollout math on torch device."""

    def __init__(
        self,
        calibrations: list[SegmentCalibration],
        config: TorchBatchConfig,
        device: str | torch.device,
        starting_cash: list[float | None] | None = None,
        scenario_names: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        if not calibrations:
            raise ValueError("TorchSegmentBatchEnv requires at least one calibration.")
        self.calibrations = calibrations
        self.config = config
        self.device = torch.device(device)
        self.batch_size = len(calibrations)
        self.seed = int(seed)
        torch.manual_seed(self.seed)

        first = calibrations[0]
        self.segment_count = first.segment_count
        self.maturity_bucket_count = first.maturity_bucket_count
        self.observation_dim = (
            1
            + self.segment_count
            + self.segment_count * self.maturity_bucket_count
            + self.segment_count * SEGMENT_PARAMETER_COUNT
        )
        self.action_dim = self.segment_count + 1
        self.scenario_names = [
            scenario_names[i] if scenario_names else calibrations[i].name for i in range(self.batch_size)
        ]
        self.starting_cash_values = [
            float(config.initial_cash if starting_cash is None or starting_cash[i] is None else starting_cash[i])
            for i in range(self.batch_size)
        ]

        month_counts = {cal.monthly_demand.shape[0] for cal in calibrations}
        if len(month_counts) != 1:
            raise ValueError("Vectorized torch simulator requires same month count within a batch.")
        self.month_count = next(iter(month_counts))

        self.monthly_demand = self._stack("monthly_demand")
        self.loan_pd = self._stack("loan_pd")
        self.monthly_default_hazard = self._stack("monthly_default_hazard")
        self.avg_loan_amount = self._stack("avg_loan_amount").clamp_min(1.0)
        self.annual_interest_rate = self._stack("annual_interest_rate")
        self.recovery_rate = self._stack("recovery_rate")
        self.funding_cost_annual = torch.as_tensor(
            [cal.funding_cost_annual for cal in calibrations],
            dtype=torch.float32,
            device=self.device,
        )
        self.servicing_cost_rate = torch.as_tensor(
            [cal.servicing_cost_rate for cal in calibrations],
            dtype=torch.float32,
            device=self.device,
        )
        self.avg_term_months = self._stack("avg_term_months").clamp(1.0, 84.0)
        self.expected_profit_rate = self._stack("expected_profit_rate")
        self.profitable_mask = self.expected_profit_rate > float(config.min_expected_profit_rate)
        self.new_loan_bucket = torch.as_tensor(
            np.stack([cal.new_loan_bucket for cal in calibrations]),
            dtype=torch.long,
            device=self.device,
        )
        self.segment_index = torch.arange(self.segment_count, device=self.device).view(1, -1).expand(self.batch_size, -1)
        self.segment_risk_bucket = torch.as_tensor(
            np.stack(
                [
                    np.array([cal.decode_segment(segment)[0] for segment in range(self.segment_count)])
                    for cal in calibrations
                ]
            ),
            dtype=torch.long,
            device=self.device,
        )
        self.bucket_months = max(1.0, float(np.nanmax(first.avg_term_months)) / self.maturity_bucket_count)
        self.remaining_months = (
            (torch.arange(self.maturity_bucket_count, dtype=torch.float32, device=self.device) + 0.5)
            * float(self.bucket_months)
        )
        self.principal_rate = (1.0 / self.remaining_months.clamp_min(1.0)).clamp(0.0, 1.0)
        self.aging_rate = min(1.0, 1.0 / self.bucket_months)
        self.segment_parameter_state = self._build_segment_parameter_state()
        self.reset()

    def _stack(self, attr: str) -> torch.Tensor:
        return torch.as_tensor(
            np.stack([getattr(cal, attr) for cal in self.calibrations]),
            dtype=torch.float32,
            device=self.device,
        )

    def _zeros(self) -> torch.Tensor:
        return torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

    def reset(self) -> torch.Tensor:
        self.budget = torch.as_tensor(self.starting_cash_values, dtype=torch.float32, device=self.device)
        self.capital = self.budget.clone()
        self.elapsed = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.demand_step = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.demand_indices = self._build_demand_indices()
        self.loan_book = torch.zeros(
            (self.batch_size, self.segment_count, self.maturity_bucket_count),
            dtype=torch.float32,
            device=self.device,
        )
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.current_demand = self._sample_monthly_demand()
        self.invalid_weight_mass = self._zeros()
        self.expected_deployment_profit = self._zeros()
        self.expected_profit_shaping_reward = self._zeros()
        self.episode = self._empty_episode_metrics()
        return self.state()

    def _empty_episode_metrics(self) -> dict[str, torch.Tensor]:
        metrics = {key: self._zeros() for key in BOOK_KEYS}
        metrics.update(
            {
                "total_reward": self._zeros(),
                "decision_period_profit": self._zeros(),
                "terminal_runoff_profit": self._zeros(),
                "terminal_value": self._zeros(),
                "liquidity_breach_count": self._zeros(),
                "liquidated": self._zeros(),
                "capital_liquidation": self._zeros(),
                "liquidity_liquidation": self._zeros(),
                "capital_breach_count": self._zeros(),
                "capital_shortfall": self._zeros(),
                "liquidation_asset_base": self._zeros(),
                "liquidation_loss": self._zeros(),
                "liquidation_proceeds": self._zeros(),
                "max_liquidity_shortfall": self._zeros(),
                "min_liquidity_coverage_ratio": torch.full(
                    (self.batch_size,), float("inf"), dtype=torch.float32, device=self.device
                ),
                "min_capital_adequacy_ratio": torch.full(
                    (self.batch_size,), float("inf"), dtype=torch.float32, device=self.device
                ),
                "disbursement": self._zeros(),
                "invalid_weight_mass": self._zeros(),
                "negative_epr_penalty": self._zeros(),
                "expected_deployment_profit": self._zeros(),
                "expected_profit_shaping_reward": self._zeros(),
                "ending_exposure": self._zeros(),
                "min_budget": self.budget.clone(),
            }
        )
        return metrics

    def state(self) -> torch.Tensor:
        scale = max(float(self.config.initial_cash), 1.0)
        return torch.cat(
            [
                (self.budget / scale).unsqueeze(1),
                self.current_demand / scale,
                self.loan_book.reshape(self.batch_size, -1) / scale,
                self.segment_parameter_state,
            ],
            dim=1,
        )

    def step(self, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        allocation = self._allocation_from_actor_action(raw_action)
        return self.step_allocation(allocation)

    def step_allocation(
        self, allocation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        active = ~self.done
        reward = self._zeros()
        if not bool(active.any().item()):
            return self.state(), reward, self.done.clone(), {}

        allocation = self._project_allocation(allocation) * active.float().unsqueeze(1)
        book = self._simulate_existing_book(active)
        decision_period_profit = book["period_profit"].clone()
        self.capital = torch.where(active, self.capital + decision_period_profit, self.capital)
        self.budget = torch.where(active, self.budget + book["cash_inflow"], self.budget)

        allocation = self._project_allocation(allocation) * active.float().unsqueeze(1)
        disbursement = allocation.sum(dim=1)
        self.budget = torch.where(active, self.budget - disbursement, self.budget)
        self.loan_book.scatter_add_(2, self.new_loan_bucket.unsqueeze(-1), allocation.unsqueeze(-1))

        inflation = self._simulate_cash_inflation(active)
        book = self._merge_book_metrics(book, inflation)
        decision_period_profit = decision_period_profit + inflation["period_profit"]
        self.capital = torch.where(active, self.capital + inflation["period_profit"], self.capital)
        self.budget = torch.where(active, self.budget + inflation["cash_inflow"], self.budget)

        required = self.required_liquidity()
        shortfall = self.liquidity_shortfall(required)
        penalty = self.config.liquidity_penalty_lambda * (shortfall / max(self.config.initial_cash, 1.0)).square()
        unprofitable_penalty = float(self.config.negative_epr_penalty) * self.invalid_weight_mass
        expected_deployment_profit = (allocation * self.expected_profit_rate).sum(dim=1)
        expected_profit_shaping_reward = (
            float(self.config.expected_profit_shaping_weight)
            * expected_deployment_profit
            / max(self.config.initial_cash, 1.0)
        )
        reward = torch.where(
            active,
            book["period_profit"] / max(self.config.initial_cash, 1.0)
            + expected_profit_shaping_reward
            - penalty
            - unprofitable_penalty,
            reward,
        )

        capital_ratio = self.capital_adequacy_ratio()
        capital_shortfall = self.capital_shortfall()
        capital_breach = capital_shortfall > 0.0
        liquidity_breach = shortfall > 0.0
        liquidate = active & (capital_breach | liquidity_breach)
        liquidity_liquidation = liquidate & ~capital_breach
        capital_liquidation = liquidate & capital_breach

        pre_liquidation_budget = self.budget.clone()
        pre_liquidation_capital_ratio = capital_ratio.clone()
        current_assets = self.budget.clamp_min(0.0) + self.loan_book.clamp_min(0.0).sum(dim=(1, 2))
        liquidity_loss = self.config.liquidity_liquidation_haircut * current_assets
        capital_loss = self.capital.clamp_min(0.0)
        liquidation_loss = torch.where(capital_liquidation, capital_loss, torch.where(liquidity_liquidation, liquidity_loss, self._zeros()))
        liquidation_proceeds = torch.where(liquidity_liquidation, (current_assets - liquidity_loss).clamp_min(0.0), self._zeros())

        book["period_profit"] = book["period_profit"] - liquidation_loss
        decision_period_profit = decision_period_profit - liquidation_loss
        reward = reward - liquidation_loss / max(self.config.initial_cash, 1.0)
        self.capital = torch.where(liquidate, (self.capital - liquidation_loss).clamp_min(0.0), self.capital)
        self.budget = torch.where(liquidate, liquidation_proceeds, self.budget)
        self.loan_book = torch.where(liquidate.view(-1, 1, 1), torch.zeros_like(self.loan_book), self.loan_book)
        self.elapsed = torch.where(active, self.elapsed + 1, self.elapsed)

        horizon_done = active & ~liquidate & (self.elapsed >= int(self.config.horizon_months))
        terminal = self._simulate_terminal_runoff(horizon_done)
        reward = reward + terminal["reward"]
        book["period_profit"] = book["period_profit"] + terminal["period_profit"] + terminal["terminal_value"]
        self.capital = self.capital + terminal["period_profit"] + terminal["terminal_value"]

        done_now = liquidate | horizon_done
        self.done = self.done | done_now
        need_next = ~self.done
        if bool(need_next.any().item()):
            sampled = self._sample_monthly_demand()
            self.current_demand = torch.where(need_next.unsqueeze(1), sampled, torch.zeros_like(self.current_demand))
        else:
            self.current_demand = torch.zeros_like(self.current_demand)

        final_required = torch.where(liquidate, required, self.required_liquidity())
        final_shortfall = torch.where(liquidate, shortfall, self.liquidity_shortfall(final_required))
        final_shortfall = torch.maximum(final_shortfall, terminal["max_shortfall"])
        post_lcr = self.liquidity_coverage_ratio(final_required)
        pre_lcr = pre_liquidation_budget / required.clamp_min(1e-9)
        final_lcr = torch.where(liquidate, pre_lcr, post_lcr)
        final_car = torch.where(liquidate, pre_liquidation_capital_ratio, self.capital_adequacy_ratio())

        self._accumulate_episode(
            book=book,
            reward=reward,
            decision_period_profit=decision_period_profit,
            terminal=terminal,
            disbursement=disbursement,
            shortfall=final_shortfall,
            liquidity_breach=liquidity_breach | (terminal["liquidity_breach_count"] > 0),
            final_lcr=final_lcr,
            final_car=final_car,
            liquidate=liquidate,
            capital_breach=capital_breach,
            capital_shortfall=capital_shortfall,
            capital_liquidation=capital_liquidation,
            liquidity_liquidation=liquidity_liquidation,
            liquidation_asset_base=torch.where(liquidate, current_assets, self._zeros()),
            liquidation_loss=liquidation_loss,
            liquidation_proceeds=liquidation_proceeds,
            invalid_weight_mass=self.invalid_weight_mass,
            negative_epr_penalty=unprofitable_penalty,
            expected_deployment_profit=expected_deployment_profit,
            expected_profit_shaping_reward=expected_profit_shaping_reward,
        )

        info = {
            "period_profit": book["period_profit"],
            "liquidity_breach": liquidity_breach,
            "liquidated": liquidate,
            "liquidity_liquidation": liquidity_liquidation,
            "capital_liquidation": capital_liquidation,
            "liquidation_loss": liquidation_loss,
            "liquidity_coverage_ratio": final_lcr,
            "capital_adequacy_ratio": final_car,
            "disbursement": disbursement,
            "invalid_weight_mass": self.invalid_weight_mass,
            "negative_epr_penalty": unprofitable_penalty,
            "expected_deployment_profit": expected_deployment_profit,
            "expected_profit_shaping_reward": expected_profit_shaping_reward,
        }
        return self.state(), reward, self.done.clone(), info

    def _accumulate_episode(self, **kwargs: torch.Tensor | dict[str, torch.Tensor]) -> None:
        book = kwargs["book"]
        assert isinstance(book, dict)
        for key in BOOK_KEYS:
            self.episode[key] += book[key].detach()
        self.episode["total_reward"] += kwargs["reward"].detach()
        self.episode["decision_period_profit"] += kwargs["decision_period_profit"].detach()
        terminal = kwargs["terminal"]
        assert isinstance(terminal, dict)
        self.episode["terminal_runoff_profit"] += terminal["period_profit"].detach()
        self.episode["terminal_value"] += terminal["terminal_value"].detach()
        self.episode["disbursement"] += kwargs["disbursement"].detach()
        self.episode["invalid_weight_mass"] += kwargs["invalid_weight_mass"].detach()
        self.episode["negative_epr_penalty"] += kwargs["negative_epr_penalty"].detach()
        self.episode["expected_deployment_profit"] += kwargs["expected_deployment_profit"].detach()
        self.episode["expected_profit_shaping_reward"] += kwargs["expected_profit_shaping_reward"].detach()
        self.episode["max_liquidity_shortfall"] = torch.maximum(
            self.episode["max_liquidity_shortfall"], kwargs["shortfall"].detach()
        )
        self.episode["liquidity_breach_count"] += kwargs["liquidity_breach"].float().detach()
        self.episode["liquidated"] = torch.maximum(self.episode["liquidated"], kwargs["liquidate"].float().detach())
        self.episode["capital_breach_count"] += kwargs["capital_breach"].float().detach()
        self.episode["capital_shortfall"] = torch.maximum(
            self.episode["capital_shortfall"], kwargs["capital_shortfall"].detach()
        )
        self.episode["capital_liquidation"] = torch.maximum(
            self.episode["capital_liquidation"], kwargs["capital_liquidation"].float().detach()
        )
        self.episode["liquidity_liquidation"] = torch.maximum(
            self.episode["liquidity_liquidation"], kwargs["liquidity_liquidation"].float().detach()
        )
        self.episode["liquidation_asset_base"] += kwargs["liquidation_asset_base"].detach()
        self.episode["liquidation_loss"] += kwargs["liquidation_loss"].detach()
        self.episode["liquidation_proceeds"] += kwargs["liquidation_proceeds"].detach()
        self.episode["min_liquidity_coverage_ratio"] = torch.minimum(
            self.episode["min_liquidity_coverage_ratio"], kwargs["final_lcr"].detach()
        )
        self.episode["min_capital_adequacy_ratio"] = torch.minimum(
            self.episode["min_capital_adequacy_ratio"], kwargs["final_car"].detach()
        )
        self.episode["ending_exposure"] = self.loan_book.sum(dim=(1, 2)).detach()
        self.episode["min_budget"] = torch.minimum(self.episode["min_budget"], self.budget.detach())

    def _simulate_existing_book(self, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        exposure = self.loan_book
        metrics = {key: self._zeros() for key in BOOK_KEYS}
        if not bool((mask & (exposure.sum(dim=(1, 2)) > 1e-6)).any().item()):
            self.loan_book = torch.where(mask.view(-1, 1, 1), torch.zeros_like(exposure), exposure)
            return metrics

        hazard = self.monthly_default_hazard.unsqueeze(2).clamp(0.0, 0.95)
        count = torch.floor(exposure / self.avg_loan_amount.unsqueeze(2).clamp_min(1.0)).clamp_min(1.0)
        if self.config.default_mode == "expected":
            default_fraction = hazard.expand_as(exposure)
        else:
            std = torch.sqrt((hazard * (1.0 - hazard) / count).clamp_min(0.0))
            default_fraction = (hazard + torch.randn_like(exposure) * std).clamp(0.0, 1.0)
        default_fraction = torch.where(exposure > 1e-6, default_fraction, torch.zeros_like(default_fraction))
        defaults = exposure * default_fraction
        surviving = (exposure - defaults).clamp_min(0.0)
        principal_paid = surviving * self.principal_rate.view(1, 1, -1)
        post_principal = (surviving - principal_paid).clamp_min(0.0)

        interest_income = exposure * (self.annual_interest_rate.unsqueeze(2) / 12.0)
        recovery = defaults * self.recovery_rate.unsqueeze(2)
        default_loss = defaults - recovery
        funding_cost = exposure * (self.funding_cost_annual.view(-1, 1, 1) / 12.0)
        servicing_cost = exposure * (self.servicing_cost_rate.view(-1, 1, 1) / 12.0)

        next_book = torch.zeros_like(exposure)
        matured_principal = post_principal[:, :, 0] * float(self.aging_rate)
        next_book[:, :, 0] += post_principal[:, :, 0] * (1.0 - float(self.aging_rate))
        for h in range(1, self.maturity_bucket_count):
            stay = post_principal[:, :, h] * (1.0 - float(self.aging_rate))
            advance = post_principal[:, :, h] * float(self.aging_rate)
            next_book[:, :, h] += stay
            next_book[:, :, h - 1] += advance
        self.loan_book = torch.where(mask.view(-1, 1, 1), next_book, exposure)

        principal_total = principal_paid.sum(dim=(1, 2)) + matured_principal.sum(dim=1)
        gross_cash_inflow = principal_total + interest_income.sum(dim=(1, 2)) + recovery.sum(dim=(1, 2))
        cash_outflow = funding_cost.sum(dim=(1, 2)) + servicing_cost.sum(dim=(1, 2))
        metrics["cash_inflow"] = torch.where(mask, gross_cash_inflow - cash_outflow, metrics["cash_inflow"])
        metrics["gross_cash_inflow"] = torch.where(mask, gross_cash_inflow, metrics["gross_cash_inflow"])
        metrics["cash_outflow"] = torch.where(mask, cash_outflow, metrics["cash_outflow"])
        metrics["interest_income"] = torch.where(mask, interest_income.sum(dim=(1, 2)), metrics["interest_income"])
        metrics["principal_paid"] = torch.where(mask, principal_total, metrics["principal_paid"])
        metrics["recovery"] = torch.where(mask, recovery.sum(dim=(1, 2)), metrics["recovery"])
        metrics["default_loss"] = torch.where(mask, default_loss.sum(dim=(1, 2)), metrics["default_loss"])
        metrics["defaulted_exposure"] = torch.where(mask, defaults.sum(dim=(1, 2)), metrics["defaulted_exposure"])
        metrics["funding_cost"] = torch.where(mask, funding_cost.sum(dim=(1, 2)), metrics["funding_cost"])
        metrics["servicing_cost"] = torch.where(mask, servicing_cost.sum(dim=(1, 2)), metrics["servicing_cost"])
        profit = interest_income.sum(dim=(1, 2)) - default_loss.sum(dim=(1, 2)) - cash_outflow
        metrics["period_profit"] = torch.where(mask, profit, metrics["period_profit"])
        return metrics

    def _monthly_inflation_rate(self) -> float:
        annual = max(float(self.config.expected_inflation_annual), 0.0)
        return (1.0 + annual) ** (1.0 / 12.0) - 1.0

    def _simulate_cash_inflation(self, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        metrics = {key: self._zeros() for key in BOOK_KEYS}
        monthly_rate = self._monthly_inflation_rate()
        if monthly_rate <= 0.0:
            return metrics
        cost = self.budget.clamp_min(0.0) * float(monthly_rate)
        metrics["cash_inflow"] = torch.where(mask, -cost, metrics["cash_inflow"])
        metrics["cash_outflow"] = torch.where(mask, cost, metrics["cash_outflow"])
        metrics["cash_inflation_cost"] = torch.where(mask, cost, metrics["cash_inflation_cost"])
        metrics["period_profit"] = torch.where(mask, -cost, metrics["period_profit"])
        return metrics

    def _merge_book_metrics(
        self, left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {key: left[key] + right[key] for key in BOOK_KEYS}

    def _simulate_terminal_runoff(self, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        metrics = {key: self._zeros() for key in BOOK_KEYS}
        metrics.update(
            {
                "reward": self._zeros(),
                "liquidity_penalty": self._zeros(),
                "liquidity_breach_count": self._zeros(),
                "max_shortfall": self._zeros(),
                "terminal_value": self._zeros(),
            }
        )
        discount = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
        active = mask.clone()
        for _ in range(int(self.config.terminal_runoff_months)):
            active = active & (self.loan_book.sum(dim=(1, 2)) > 1e-6)
            if not bool(active.any().item()):
                break
            # Terminal runoff settles the existing loan book after the decision
            # horizon. Cash inflation is charged during decision months only so
            # reject-all and lending policies are compared over the same policy
            # horizon rather than giving reject-all a shorter inflation window.
            book = self._simulate_existing_book(active)
            self.budget = torch.where(active, self.budget + book["cash_inflow"], self.budget)
            shortfall = self.liquidity_shortfall()
            penalty = self.config.liquidity_penalty_lambda * (shortfall / max(self.config.initial_cash, 1.0)).square()
            for key in BOOK_KEYS:
                metrics[key] += book[key] * discount
            metrics["reward"] += discount * (book["period_profit"] / max(self.config.initial_cash, 1.0) - penalty)
            metrics["liquidity_penalty"] += discount * penalty
            metrics["liquidity_breach_count"] += (shortfall > 0.0).float()
            metrics["max_shortfall"] = torch.maximum(metrics["max_shortfall"], shortfall)
            discount = discount * float(self.config.terminal_runoff_discount)
        terminal_value = self.terminal_expected_profit()
        metrics["terminal_value"] = terminal_value
        metrics["reward"] += discount * terminal_value / max(self.config.initial_cash, 1.0)
        return metrics

    def terminal_expected_profit(self) -> torch.Tensor:
        if self.config.terminal_value_weight == 0.0:
            return self._zeros()
        term_scale = torch.minimum(
            torch.ones_like(self.loan_book),
            self.remaining_months.view(1, 1, -1) / self.avg_term_months.unsqueeze(2).clamp_min(1.0),
        )
        return (
            self.loan_book
            * self.expected_profit_rate.unsqueeze(2)
            * term_scale
            * float(self.config.terminal_value_weight)
        ).sum(dim=(1, 2))

    def _allocation_from_actor_action(self, raw: torch.Tensor) -> torch.Tensor:
        deploy_fraction = torch.sigmoid(raw[:, 0].clamp(-20.0, 20.0))
        temperature = max(float(self.config.allocation_temperature), 1e-3)
        segment_logits = raw[:, 1:] / temperature
        segment_logits = segment_logits + float(self.config.segment_epr_prior_scale) * self.expected_profit_rate
        base_weights = F.softmax(segment_logits, dim=1)
        self.invalid_weight_mass = (base_weights * (~self.profitable_mask).float()).sum(dim=1).detach()
        active_mask = self.profitable_mask
        top_k = int(self.config.allocation_top_k)
        if 0 < top_k < self.segment_count:
            eligible_logits = torch.where(active_mask, segment_logits, torch.full_like(segment_logits, -1.0e9))
            top_indices = torch.topk(eligible_logits, k=top_k, dim=1).indices
            top_mask = torch.zeros_like(active_mask)
            top_mask.scatter_(1, top_indices, True)
            active_mask = active_mask & top_mask
            active_mask = torch.where(active_mask.any(dim=1, keepdim=True), active_mask, self.profitable_mask)
        masked_logits = torch.where(active_mask, segment_logits, torch.full_like(segment_logits, -1.0e9))
        weights = F.softmax(masked_logits, dim=1)
        weights = torch.where(active_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        capacity = self.deployable_capacity()
        return self._allocate_by_weights(deploy_fraction * capacity, weights)

    def _allocate_by_weights(self, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        allocation = torch.zeros((self.batch_size, self.segment_count), dtype=torch.float32, device=self.device)
        eligible_demand = torch.where(self.profitable_mask, self.current_demand, torch.zeros_like(self.current_demand))
        remaining = torch.minimum(torch.minimum(target, eligible_demand.sum(dim=1)), self.deployable_capacity())
        active = eligible_demand > 1e-9
        weights = weights.clamp_min(0.0)
        for _ in range(self.segment_count):
            active_weights = torch.where(active, weights, torch.zeros_like(weights))
            weight_sum = active_weights.sum(dim=1, keepdim=True)
            fallback = active.float()
            active_weights = torch.where(weight_sum > 1e-9, active_weights, fallback)
            weight_sum = active_weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
            proposed = remaining.unsqueeze(1) * active_weights / weight_sum
            room = (eligible_demand - allocation).clamp_min(0.0)
            increment = torch.minimum(proposed, room)
            allocation = allocation + increment
            remaining = (remaining - increment.sum(dim=1)).clamp_min(0.0)
            active = (room - increment > 1e-9) & self.profitable_mask
        return self._project_allocation(allocation)

    def _project_allocation(self, allocation: torch.Tensor) -> torch.Tensor:
        demand_cap = torch.where(self.profitable_mask, self.current_demand, torch.zeros_like(self.current_demand))
        projected = torch.minimum(allocation.clamp_min(0.0), demand_cap)
        capacity = self.deployable_capacity()
        total = projected.sum(dim=1)
        scale = torch.where(total > capacity, capacity / total.clamp_min(1e-9), torch.ones_like(total))
        projected = projected * scale.unsqueeze(1)
        projected = torch.where((capacity > 0.0).unsqueeze(1), projected, torch.zeros_like(projected))
        return projected

    def deployable_capacity(self) -> torch.Tensor:
        if self.config.enforce_min_cash_constraint:
            required = self.required_liquidity()
            return ((self.budget - required) / (1.0 + self.config.reserve_exposure_ratio)).clamp_min(0.0)
        return self.budget.clamp_min(0.0)

    def _sample_monthly_demand(self) -> torch.Tensor:
        batch = torch.arange(self.batch_size, device=self.device)
        position = self.demand_step.clamp_max(self.demand_indices.shape[1] - 1)
        index = self.demand_indices[batch, position]
        self.demand_step = self.demand_step + 1
        return self.monthly_demand[batch, index, :].clone()

    def _build_demand_indices(self) -> torch.Tensor:
        horizon = max(int(self.config.horizon_months) + 1, 1)
        rng = np.random.default_rng(self.seed)
        indices = rng.integers(0, self.month_count, size=(self.batch_size, horizon), endpoint=False)
        return torch.as_tensor(indices, dtype=torch.long, device=self.device)

    def required_liquidity(self) -> torch.Tensor:
        return float(self.config.initial_cash * self.config.min_cash_ratio) + float(
            self.config.reserve_exposure_ratio
        ) * self.loan_book.clamp_min(0.0).sum(dim=(1, 2))

    def liquidity_shortfall(self, required: torch.Tensor | None = None) -> torch.Tensor:
        required = self.required_liquidity() if required is None else required
        return (required - self.budget).clamp_min(0.0)

    def liquidity_coverage_ratio(self, required: torch.Tensor | None = None) -> torch.Tensor:
        required = self.required_liquidity() if required is None else required
        return self.budget / required.clamp_min(1e-9)

    def risk_weighted_assets(self) -> torch.Tensor:
        return self.loan_book.clamp_min(0.0).sum(dim=(1, 2))

    def capital_adequacy_ratio(self) -> torch.Tensor:
        rwa = self.risk_weighted_assets()
        return torch.where(rwa > 1e-9, self.capital / rwa.clamp_min(1e-9), torch.full_like(rwa, float("inf")))

    def capital_shortfall(self) -> torch.Tensor:
        rwa = self.risk_weighted_assets()
        return (float(self.config.min_capital_ratio) * rwa - self.capital).clamp_min(0.0)

    def _build_segment_parameter_state(self) -> torch.Tensor:
        params = torch.stack(
            [
                self.loan_pd.clamp(0.0, 1.0),
                self.annual_interest_rate.clamp(0.0, 1.0),
                self.recovery_rate.clamp(0.0, 1.0),
                self.expected_profit_rate.clamp(-1.0, 1.0),
            ],
            dim=2,
        )
        return params.reshape(self.batch_size, -1)

    def episode_frame(self) -> pd.DataFrame:
        data = {key: value.detach().cpu().numpy() for key, value in self.episode.items()}
        data["cumulative_profit"] = data["period_profit"]
        data["liquidity_breach"] = (data["liquidity_breach_count"] > 0).astype(float)
        data["capital_breach"] = (data["capital_breach_count"] > 0).astype(float)
        data["terminal_budget"] = self.budget.detach().cpu().numpy()
        data["terminal_capital"] = self.capital.detach().cpu().numpy()
        data["ending_exposure"] = self.loan_book.sum(dim=(1, 2)).detach().cpu().numpy()
        data["expected_shortfall"] = data["max_liquidity_shortfall"]
        data["max_shortfall"] = data["max_liquidity_shortfall"]
        data["max_drawdown"] = np.maximum(0.0, float(self.config.initial_cash) - data["min_budget"])
        data["capital_utilization"] = data["disbursement"] / max(
            float(self.config.initial_cash) * float(self.config.horizon_months), 1.0
        )
        data["final_decision_disbursement"] = np.zeros(self.batch_size, dtype=float)
        data["mean_preterminal_disbursement"] = np.zeros(self.batch_size, dtype=float)
        data["final_disbursement_ratio"] = np.zeros(self.batch_size, dtype=float)
        data["final_disbursement_share"] = np.zeros(self.batch_size, dtype=float)
        data["episode"] = np.arange(self.batch_size)
        return pd.DataFrame(data)


def build_torch_batch_config(args: Any) -> TorchBatchConfig:
    return TorchBatchConfig(
        initial_cash=float(args.initial_cash),
        min_cash_ratio=float(args.min_cash_ratio),
        horizon_months=int(args.horizon_months),
        liquidity_penalty_lambda=float(args.liquidity_penalty_lambda),
        reserve_exposure_ratio=float(args.reserve_exposure_ratio),
        terminal_runoff_months=int(args.terminal_runoff_months),
        terminal_runoff_discount=float(getattr(args, "terminal_runoff_discount", 0.99)),
        terminal_value_weight=float(getattr(args, "terminal_value_weight", 1.0)),
        expected_inflation_annual=float(getattr(args, "expected_inflation_annual", 0.025)),
        min_capital_ratio=float(getattr(args, "min_capital_ratio", 0.08)),
        liquidity_liquidation_haircut=float(getattr(args, "liquidity_liquidation_haircut", 0.60)),
        enforce_min_cash_constraint=False,
        default_mode=str(getattr(args, "gpu_default_mode", "normal")),
        min_expected_profit_rate=float(getattr(args, "min_expected_profit_rate", 0.0)),
        negative_epr_penalty=float(getattr(args, "negative_epr_penalty", 0.0)),
        allocation_temperature=float(getattr(args, "allocation_temperature", 1.0)),
        segment_epr_prior_scale=float(getattr(args, "segment_epr_prior_scale", 0.0)),
        expected_profit_shaping_weight=float(getattr(args, "expected_profit_shaping_weight", 0.0)),
        allocation_top_k=int(getattr(args, "allocation_top_k", 0)),
    )


def summarize_episode_frame(frame: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    for column in frame.columns:
        if column in {"episode", "policy"}:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        summary[f"{column}_mean"] = float(series.mean()) if not series.dropna().empty else float("nan")
        summary[f"{column}_std"] = float(series.std(ddof=0)) if not series.dropna().empty else float("nan")
    summary["episode_count"] = int(len(frame))
    for column in ["liquidated", "capital_liquidation", "liquidity_liquidation", "capital_breach", "liquidity_breach"]:
        if column in frame.columns:
            summary[f"{column}_episode_count"] = int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
    return summary


def torch_allocate_by_scores(env: TorchSegmentBatchEnv, scores: torch.Tensor, deploy_fraction: torch.Tensor) -> torch.Tensor:
    required = env.required_liquidity()
    safe_capacity = ((env.budget - required) / (1.0 + env.config.reserve_exposure_ratio)).clamp_min(0.0)
    safe_capacity = safe_capacity * 0.99  # 1% 버퍼
    capacity = safe_capacity * deploy_fraction
    order = torch.argsort(scores, dim=1, descending=True)
    sorted_scores = torch.gather(scores, 1, order)
    sorted_demand = torch.gather(env.current_demand, 1, order)
    valid = torch.isfinite(sorted_scores)
    sorted_demand = torch.where(valid, sorted_demand, torch.zeros_like(sorted_demand))
    cumulative = torch.cumsum(sorted_demand, dim=1)
    previous = cumulative - sorted_demand
    sorted_alloc = (capacity.unsqueeze(1) - previous).clamp_min(0.0)
    sorted_alloc = torch.minimum(sorted_alloc, sorted_demand)
    allocation = torch.zeros_like(sorted_alloc)
    allocation.scatter_(1, order, sorted_alloc)
    return env._project_allocation(allocation)


def torch_policy_allocation(env: TorchSegmentBatchEnv, policy: str) -> torch.Tensor:
    b, j = env.batch_size, env.segment_count
    if policy == "reject_all":
        return torch.zeros((b, j), dtype=torch.float32, device=env.device)
    if policy == "ppo_placeholder":
        raise ValueError("PPO policy is handled through actor raw actions.")
    neg_inf = torch.full((b, j), -float("inf"), dtype=torch.float32, device=env.device)
    if policy == "fixed_conservative":
        scores = torch.where(env.segment_risk_bucket <= 1, env.expected_profit_rate + 0.05, neg_inf)
        deploy = torch.full((b,), 0.55, dtype=torch.float32, device=env.device)
    elif policy == "fixed_aggressive":
        risk_weights = torch.tensor([0.40, 0.30, 0.20, 0.10, 0.0], dtype=torch.float32, device=env.device)
        scores = risk_weights[env.segment_risk_bucket] + 0.01 * env.expected_profit_rate
        deploy = torch.full((b,), 0.90, dtype=torch.float32, device=env.device)
    elif policy in {"greedy_expected_profit", "one_step_lp_closed_form"}:
        scores = torch.where(env.expected_profit_rate > 0.0, env.expected_profit_rate, neg_inf)
        deploy = torch.ones((b,), dtype=torch.float32, device=env.device)
    elif policy == "budget_aware_heuristic":
        reserve_ratio = env.budget / env.required_liquidity().clamp_min(1.0)
        max_risk = torch.where(reserve_ratio <= 1.2, 1, torch.where(reserve_ratio <= 2.0, 2, 3))
        deploy = torch.where(
            reserve_ratio <= 1.2,
            torch.full_like(reserve_ratio, 0.35),
            torch.where(reserve_ratio <= 2.0, torch.full_like(reserve_ratio, 0.65), torch.full_like(reserve_ratio, 0.90)),
        )
        mask = env.segment_risk_bucket <= max_risk.unsqueeze(1)
        scores = torch.where(mask, env.expected_profit_rate + (max_risk.unsqueeze(1) - env.segment_risk_bucket) * 0.01, neg_inf)
    else:
        raise ValueError(f"Unknown torch policy: {policy}")
    return torch_allocate_by_scores(env, scores, deploy)


def evaluate_torch_policy_batch(
    make_env: Callable[[int], TorchSegmentBatchEnv],
    policy: str,
    agent: Any | None,
    episodes: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    env = make_env(seed)
    obs = env.reset()
    while not bool(env.done.all().item()):
        active = ~env.done
        if policy == "ppo_rl":
            with torch.no_grad():
                dist = agent.model.distribution(obs)
                raw = dist.mean
            _, _, _, _ = env.step(raw)
        else:
            allocation = torch_policy_allocation(env, policy)
            _, _, _, _ = env.step_allocation(allocation)
        obs = env.state()
    frame = env.episode_frame().iloc[:episodes].copy()
    frame.insert(0, "policy", policy)
    return summarize_episode_frame(frame), frame

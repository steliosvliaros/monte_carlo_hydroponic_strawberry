from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================
# FROZEN MODEL LABEL
# ============================================================
# ioannina_dc_strategy_v1_cashflows
# Purpose:
# Create detailed annual cash-flow calculations by scenario for the
# single 12 MW Ioannina data center project across:
# - Turnkey
# - Powered Shell
# - Build-to-exit
#
# The script exports:
# 1) A summary table by strategy/scenario
# 2) A detailed annual cashflow table for every strategy/scenario
# 3) Monte Carlo summary outputs
# 4) Decision matrix and risk register
# 5) An Excel workbook with one sheet per strategy/scenario cashflow
#
# IMPORTANT:
# - These are underwriting assumptions, not market facts.
# - Update the assumptions when you have real EPC budgets, tenant terms,
#   lender term sheets, or utility milestone dates.
# ============================================================


@dataclass
class Strategy:
    name: str
    tdc_eur: float
    rent_eur_per_kw_month: float
    opex_year1_eur: float
    exit_yield: float
    debt_cost: float
    debt_ratio: float
    occupancy_ramp: List[float]
    lease_escalation: float = 0.02
    opex_escalation: float = 0.025
    debt_tenor_years: int = 15
    build_years: int = 2
    op_years: int = 10
    sale_year_index: int | None = None
    capex_split: Tuple[float, float] = (0.60, 0.40)
    debt_draw_split: Tuple[float, float] = (0.60, 0.40)
    exit_transaction_cost_pct: float = 0.06
    replacement_cost_exit_ratio: float = 1.00
    exit_income_weight: float = 0.30
    bankability_with_anchor: int = 3
    bankability_without_anchor: int = 3
    fit_with_ioannina: int = 3
    optionality: int = 3
    notes: str = ""


@dataclass
class GlobalAssumptions:
    it_load_mw: float = 12.0
    discount_rate_project: float = 0.09
    discount_rate_equity: float = 0.13
    hurdle_equity_irr: float = 0.15
    iterations: int = 2000
    random_seed: int = 42
    start_year: int = 2026
    energization_year: int = 2027
    energization_month: int = 7


@dataclass
class Scenario:
    name: str
    rent_multiplier: float = 1.0
    capex_multiplier: float = 1.0
    opex_multiplier: float = 1.0
    exit_yield_shift: float = 0.0
    debt_cost_shift: float = 0.0
    occupancy_shift: float = 0.0


def kw_from_mw(mw: float) -> float:
    return mw * 1000.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def npv(rate: float, values: np.ndarray) -> float:
    return float(sum(v / ((1.0 + rate) ** i) for i, v in enumerate(values)))


def irr(values: np.ndarray) -> float:
    if np.all(values >= 0) or np.all(values <= 0):
        return float("nan")

    def f(r: float) -> float:
        return sum(v / ((1.0 + r) ** i) for i, v in enumerate(values))

    lo, hi = -0.99, 5.0
    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo * fhi > 0 and tries < 30:
        hi *= 2.0
        fhi = f(hi)
        tries += 1
    if flo * fhi > 0:
        return float("nan")

    for _ in range(200):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if abs(fm) < 1e-10:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return mid


def annuity_payment(principal: float, annual_rate: float, tenor_years: int) -> float:
    if principal <= 0:
        return 0.0
    if annual_rate == 0:
        return principal / tenor_years
    r = annual_rate
    return principal * r / (1.0 - (1.0 + r) ** (-tenor_years))


def blended_terminal_value(next_noi: float, exit_yield: float, tdc: float, strategy: Strategy) -> float:
    income_value = next_noi / exit_yield
    replacement_cost_value = tdc * strategy.replacement_cost_exit_ratio
    income_weight = clamp(strategy.exit_income_weight, 0.0, 1.0)
    gross_terminal_value = (
        income_weight * income_value
        + (1.0 - income_weight) * replacement_cost_value
    )
    sale_cost_pct = clamp(strategy.exit_transaction_cost_pct, 0.0, 0.25)
    return gross_terminal_value * (1.0 - sale_cost_pct)


def _calendar_year_for_row(ga: GlobalAssumptions, phase: str, op_year: int | None = None, build_index: int | None = None) -> int:
    if phase == "Build":
        assert build_index is not None
        return ga.start_year + build_index - 1
    assert op_year is not None
    return ga.start_year + 1 + op_year


def build_detailed_cashflows(strategy: Strategy, ga: GlobalAssumptions, scenario: Scenario, include_break_even: bool = True) -> Dict[str, object]:
    it_kw = kw_from_mw(ga.it_load_mw)

    tdc = strategy.tdc_eur * scenario.capex_multiplier
    rent = strategy.rent_eur_per_kw_month * scenario.rent_multiplier
    opex_year1 = strategy.opex_year1_eur * scenario.opex_multiplier
    exit_yield = strategy.exit_yield + scenario.exit_yield_shift
    debt_cost = strategy.debt_cost + scenario.debt_cost_shift

    if exit_yield <= 0:
        raise ValueError("exit_yield must remain positive")
    if debt_cost < 0:
        raise ValueError("debt_cost must remain non-negative")
    if strategy.replacement_cost_exit_ratio <= 0:
        raise ValueError("replacement_cost_exit_ratio must remain positive")

    debt_committed = tdc * strategy.debt_ratio
    equity_committed = tdc - debt_committed
    annual_debt_service = annuity_payment(debt_committed, debt_cost, strategy.debt_tenor_years)

    rows: List[dict] = []
    project_cf_list: List[float] = []
    equity_cf_list: List[float] = []
    debt_balance = 0.0

    # Build years
    for build_idx in range(1, strategy.build_years + 1):
        capex = -tdc * strategy.capex_split[build_idx - 1]
        debt_draw = debt_committed * strategy.debt_draw_split[build_idx - 1]
        equity_draw = equity_committed * strategy.debt_draw_split[build_idx - 1]
        debt_balance += debt_draw
        interest = 0.0
        principal = 0.0
        debt_service = 0.0
        revenue = 0.0
        opex = 0.0
        noi = 0.0
        terminal_value = 0.0
        project_cf = capex
        equity_cf = -equity_draw

        project_cf_list.append(project_cf)
        equity_cf_list.append(equity_cf)

        rows.append(
            {
                "strategy": strategy.name,
                "scenario": scenario.name,
                "phase": "Build",
                "period_index": build_idx - 1,
                "calendar_year": _calendar_year_for_row(ga, "Build", build_index=build_idx),
                "build_year": build_idx,
                "op_year": np.nan,
                "occupancy": 0.0,
                "it_kw": it_kw,
                "rent_eur_per_kw_month": rent,
                "gross_potential_revenue": 0.0,
                "revenue": revenue,
                "opex": opex,
                "noi": noi,
                "capex": capex,
                "debt_draw": debt_draw,
                "equity_draw": equity_draw,
                "interest": interest,
                "principal": principal,
                "debt_service": debt_service,
                "debt_balance_end": debt_balance,
                "terminal_or_sale_value": terminal_value,
                "project_cashflow": project_cf,
                "equity_cashflow": equity_cf,
            }
        )

    # Operating years
    min_dscr = float("inf")
    sale_row_period = None
    for op_year in range(1, strategy.op_years + 1):
        occ = strategy.occupancy_ramp[min(op_year - 1, len(strategy.occupancy_ramp) - 1)]
        occ = clamp(occ + scenario.occupancy_shift, 0.0, 1.0)

        gross_potential_revenue = it_kw * rent * 12.0 * ((1.0 + strategy.lease_escalation) ** (op_year - 1))
        revenue = gross_potential_revenue * occ
        opex = opex_year1 * ((1.0 + strategy.opex_escalation) ** (op_year - 1))
        noi = revenue - opex

        interest = debt_balance * debt_cost
        principal = max(0.0, annual_debt_service - interest)
        principal = min(principal, debt_balance)
        debt_service = interest + principal
        debt_balance = max(0.0, debt_balance - principal)

        dscr = noi / debt_service if debt_service > 0 else np.inf
        min_dscr = min(min_dscr, dscr)

        terminal_value = 0.0
        if strategy.sale_year_index is not None:
            if op_year == strategy.sale_year_index:
                next_occ = occ
                next_gpr = it_kw * rent * 12.0 * ((1.0 + strategy.lease_escalation) ** op_year)
                next_revenue = next_gpr * next_occ
                next_opex = opex_year1 * ((1.0 + strategy.opex_escalation) ** op_year)
                next_noi = next_revenue - next_opex
                terminal_value = blended_terminal_value(next_noi, exit_yield, tdc, strategy)
                sale_row_period = strategy.build_years + op_year - 1
        elif op_year == strategy.op_years:
            next_occ = strategy.occupancy_ramp[min(op_year, len(strategy.occupancy_ramp) - 1)]
            next_occ = clamp(next_occ + scenario.occupancy_shift, 0.0, 1.0)
            next_gpr = it_kw * rent * 12.0 * ((1.0 + strategy.lease_escalation) ** op_year)
            next_revenue = next_gpr * next_occ
            next_opex = opex_year1 * ((1.0 + strategy.opex_escalation) ** op_year)
            next_noi = next_revenue - next_opex
            terminal_value = blended_terminal_value(next_noi, exit_yield, tdc, strategy)

        project_cf = noi + terminal_value
        equity_cf = noi - debt_service + max(0.0, terminal_value - debt_balance)

        project_cf_list.append(project_cf)
        equity_cf_list.append(equity_cf)

        rows.append(
            {
                "strategy": strategy.name,
                "scenario": scenario.name,
                "phase": "Operate",
                "period_index": strategy.build_years + op_year - 1,
                "calendar_year": _calendar_year_for_row(ga, "Operate", op_year=op_year),
                "build_year": np.nan,
                "op_year": op_year,
                "occupancy": occ,
                "it_kw": it_kw,
                "rent_eur_per_kw_month": rent,
                "gross_potential_revenue": gross_potential_revenue,
                "revenue": revenue,
                "opex": opex,
                "noi": noi,
                "capex": 0.0,
                "debt_draw": 0.0,
                "equity_draw": 0.0,
                "interest": interest,
                "principal": principal,
                "debt_service": debt_service,
                "debt_balance_end": debt_balance,
                "terminal_or_sale_value": terminal_value,
                "project_cashflow": project_cf,
                "equity_cashflow": equity_cf,
                "dscr": dscr,
            }
        )

        if strategy.sale_year_index is not None and op_year == strategy.sale_year_index:
            break

    project_cf = np.array(project_cf_list, dtype=float)
    equity_cf = np.array(equity_cf_list, dtype=float)

    annual_df = pd.DataFrame(rows)
    annual_df["discount_factor_project"] = [1.0 / ((1.0 + ga.discount_rate_project) ** i) for i in annual_df["period_index"]]
    annual_df["discount_factor_equity"] = [1.0 / ((1.0 + ga.discount_rate_equity) ** i) for i in annual_df["period_index"]]
    annual_df["discounted_project_cf"] = annual_df["project_cashflow"] * annual_df["discount_factor_project"]
    annual_df["discounted_equity_cf"] = annual_df["equity_cashflow"] * annual_df["discount_factor_equity"]
    annual_df["cum_project_cf"] = annual_df["project_cashflow"].cumsum()
    annual_df["cum_equity_cf"] = annual_df["equity_cashflow"].cumsum()

    project_npv = npv(ga.discount_rate_project, project_cf)
    equity_npv = npv(ga.discount_rate_equity, equity_cf)
    project_irr = irr(project_cf)
    equity_irr = irr(equity_cf)

    stabilized_noi = float(annual_df.loc[annual_df["phase"] == "Operate", "noi"].iloc[-1])
    terminal_value = float(annual_df["terminal_or_sale_value"].iloc[-1])
    if include_break_even:
        break_even_rent = find_break_even_rent(strategy, ga, scenario)
        break_even_occ = find_break_even_occupancy(strategy, ga, scenario)
    else:
        break_even_rent = float("nan")
        break_even_occ = float("nan")

    return {
        "annual_table": annual_df,
        "project_cf": project_cf,
        "equity_cf": equity_cf,
        "tdc": tdc,
        "rent": rent,
        "exit_yield": exit_yield,
        "project_npv": project_npv,
        "equity_npv": equity_npv,
        "project_irr": project_irr,
        "equity_irr": equity_irr,
        "min_dscr": min_dscr,
        "stabilized_noi": stabilized_noi,
        "terminal_value": terminal_value,
        "break_even_rent": break_even_rent,
        "break_even_occ": break_even_occ,
        "sale_period_index": sale_row_period,
    }


def find_break_even_rent(strategy: Strategy, ga: GlobalAssumptions, scenario: Scenario) -> float:
    lo, hi = 1.0, 300.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        temp = Strategy(**{**strategy.__dict__, "rent_eur_per_kw_month": mid})
        test_npv = build_detailed_cashflows(temp, ga, scenario, include_break_even=False)["project_npv"]
        if test_npv >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def find_break_even_occupancy(strategy: Strategy, ga: GlobalAssumptions, scenario: Scenario) -> float:
    lo, hi = 0.05, 0.99
    for _ in range(120):
        mid = (lo + hi) / 2.0
        temp_ramp = [min(mid, x) for x in strategy.occupancy_ramp]
        temp = Strategy(**{**strategy.__dict__, "occupancy_ramp": temp_ramp})
        test_npv = build_detailed_cashflows(temp, ga, scenario, include_break_even=False)["project_npv"]
        if test_npv >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def summarize_strategy(strategy: Strategy, ga: GlobalAssumptions, scenario: Scenario) -> Dict[str, float]:
    res = build_detailed_cashflows(strategy, ga, scenario)
    return {
        "Strategy": strategy.name,
        "Scenario": scenario.name,
        "TDC_EUR": res["tdc"],
        "Rent_EUR_per_kW_month": res["rent"],
        "Exit_Yield": res["exit_yield"],
        "Project_NPV_9pct": res["project_npv"],
        "Equity_NPV_13pct": res["equity_npv"],
        "Project_IRR": res["project_irr"],
        "Equity_IRR": res["equity_irr"],
        "Min_DSCR": res["min_dscr"],
        "Stabilized_NOI": res["stabilized_noi"],
        "Terminal_or_Sale_Value": res["terminal_value"],
        "Break_even_Rent_EUR_per_kW_month": res["break_even_rent"],
        "Break_even_Stabilized_Occupancy": res["break_even_occ"],
    }


def monte_carlo(strategy: Strategy, ga: GlobalAssumptions) -> pd.DataFrame:
    rng = np.random.default_rng(ga.random_seed)
    rows = []
    for _ in range(ga.iterations):
        if strategy.name == "Turnkey":
            rent_mult = rng.triangular(0.90, 1.00, 1.10)
            capex_mult = rng.triangular(0.97, 1.00, 1.09)
            exit_shift = rng.triangular(-0.0025, 0.0, 0.0100)
            debt_shift = rng.triangular(-0.0040, 0.0, 0.0100)
            occ_shift = rng.triangular(-0.08, 0.0, 0.04)
            opex_mult = rng.triangular(0.98, 1.00, 1.12)
        elif strategy.name == "Powered Shell":
            rent_mult = rng.triangular(0.90, 1.00, 1.10)
            capex_mult = rng.triangular(0.96, 1.00, 1.08)
            exit_shift = rng.triangular(-0.0025, 0.0, 0.0090)
            debt_shift = rng.triangular(-0.0030, 0.0, 0.0100)
            occ_shift = rng.triangular(-0.06, 0.0, 0.05)
            opex_mult = rng.triangular(0.98, 1.00, 1.10)
        else:
            rent_mult = rng.triangular(0.88, 1.00, 1.10)
            capex_mult = rng.triangular(0.97, 1.00, 1.10)
            exit_shift = rng.triangular(-0.0025, 0.0050, 0.0175)
            debt_shift = rng.triangular(-0.0030, 0.0, 0.0110)
            occ_shift = rng.triangular(-0.10, 0.0, 0.03)
            opex_mult = rng.triangular(0.98, 1.00, 1.10)

        scenario = Scenario(
            name="MC",
            rent_multiplier=rent_mult,
            capex_multiplier=capex_mult,
            opex_multiplier=opex_mult,
            exit_yield_shift=exit_shift,
            debt_cost_shift=debt_shift,
            occupancy_shift=occ_shift,
        )
        res = build_detailed_cashflows(strategy, ga, scenario, include_break_even=False)
        rows.append(
            {
                "npv": res["project_npv"],
                "project_irr": res["project_irr"],
                "equity_irr": res["equity_irr"],
                "rent_mult": rent_mult,
                "capex_mult": capex_mult,
                "opex_mult": opex_mult,
                "exit_shift": exit_shift,
                "debt_shift": debt_shift,
                "occ_shift": occ_shift,
            }
        )
    return pd.DataFrame(rows)


def summarize_mc(df: pd.DataFrame, hurdle: float) -> Dict[str, float]:
    return {
        "NPV_P10": float(df["npv"].quantile(0.10)),
        "NPV_P50": float(df["npv"].quantile(0.50)),
        "NPV_P90": float(df["npv"].quantile(0.90)),
        "EqIRR_P10": float(df["equity_irr"].quantile(0.10)),
        "EqIRR_P50": float(df["equity_irr"].quantile(0.50)),
        "EqIRR_P90": float(df["equity_irr"].quantile(0.90)),
        "Prob_NPV_lt_0": float((df["npv"] < 0).mean()),
        "Prob_EqIRR_lt_Hurdle": float((df["equity_irr"] < hurdle).mean()),
    }


def one_way_sensitivity(strategy: Strategy, ga: GlobalAssumptions, base_scenario: Scenario) -> pd.DataFrame:
    base = build_detailed_cashflows(strategy, ga, base_scenario)["project_npv"]
    tests = {
        "rent_-10pct": Scenario("rent_-10pct", rent_multiplier=0.90),
        "rent_+10pct": Scenario("rent_+10pct", rent_multiplier=1.10),
        "capex_+10pct": Scenario("capex_+10pct", capex_multiplier=1.10),
        "capex_-5pct": Scenario("capex_-5pct", capex_multiplier=0.95),
        "exit_yield_+100bps": Scenario("exit_yield_+100bps", exit_yield_shift=0.01),
        "exit_yield_-50bps": Scenario("exit_yield_-50bps", exit_yield_shift=-0.005),
        "debt_+100bps": Scenario("debt_+100bps", debt_cost_shift=0.01),
        "occupancy_-5pts": Scenario("occupancy_-5pts", occupancy_shift=-0.05),
    }
    rows = []
    for name, sc in tests.items():
        out = build_detailed_cashflows(strategy, ga, sc, include_break_even=False)
        rows.append({"variable": name, "npv_delta": out["project_npv"] - base})
    return pd.DataFrame(rows).sort_values("npv_delta")


def score_option(strategy: Strategy, downside_npv: float, upside_eq_irr: float, mc_summary: Dict[str, float]) -> Dict[str, float]:
    risk_adjusted_return = 5 if mc_summary["Prob_EqIRR_lt_Hurdle"] < 0.05 else 4 if mc_summary["Prob_EqIRR_lt_Hurdle"] < 0.15 else 3 if mc_summary["Prob_EqIRR_lt_Hurdle"] < 0.30 else 2
    downside_resilience = 5 if downside_npv > 0 else 3 if downside_npv > -5_000_000 else 2
    absolute_upside = 5 if upside_eq_irr > 0.30 else 4 if upside_eq_irr > 0.22 else 3
    exit_sensitivity = 1 if strategy.name == "Build-to-exit" else 4 if strategy.name == "Powered Shell" else 3

    weights = {
        "risk_adjusted_return": 20,
        "downside_resilience": 15,
        "absolute_upside": 10,
        "bankability_without_anchor": 10,
        "bankability_with_anchor": 10,
        "fit_with_ioannina": 15,
        "optionality": 10,
        "exit_sensitivity": 10,
    }
    scores = {
        "risk_adjusted_return": risk_adjusted_return,
        "downside_resilience": downside_resilience,
        "absolute_upside": absolute_upside,
        "bankability_without_anchor": strategy.bankability_without_anchor,
        "bankability_with_anchor": strategy.bankability_with_anchor,
        "fit_with_ioannina": strategy.fit_with_ioannina,
        "optionality": strategy.optionality,
        "exit_sensitivity": exit_sensitivity,
    }

    weighted_score = sum(weights[k] * scores[k] for k in weights) / 5.0
    out = {"Strategy": strategy.name, "Weighted_Score_100": weighted_score}
    out.update({f"score_{k}": v for k, v in scores.items()})
    return out


def make_strategies() -> Dict[str, Strategy]:
    return {
        "Turnkey": Strategy(
            name="Turnkey",
            tdc_eur=119_000_000,
            rent_eur_per_kw_month=90.0,
            opex_year1_eur=1_500_000,
            exit_yield=0.0675,
            debt_cost=0.058,
            debt_ratio=0.60,
            occupancy_ramp=[0.40, 0.70, 0.85, 0.92, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
            bankability_with_anchor=5,
            bankability_without_anchor=2,
            fit_with_ioannina=2,
            optionality=2,
            notes="Best only with a strong anchor pre-let.",
        ),
        "Powered Shell": Strategy(
            name="Powered Shell",
            tdc_eur=57_000_000,
            rent_eur_per_kw_month=52.0,
            opex_year1_eur=850_000,
            exit_yield=0.0750,
            debt_cost=0.061,
            debt_ratio=0.60,
            occupancy_ramp=[0.28, 0.52, 0.72, 0.84, 0.90, 0.93, 0.95, 0.95, 0.95, 0.95],
            bankability_with_anchor=3,
            bankability_without_anchor=3,
            fit_with_ioannina=5,
            optionality=5,
            notes="Preferred base-case option for Ioannina now.",
        ),
        "Build-to-exit": Strategy(
            name="Build-to-exit",
            tdc_eur=62_000_000,
            rent_eur_per_kw_month=57.0,
            opex_year1_eur=950_000,
            exit_yield=0.0725,
            debt_cost=0.062,
            debt_ratio=0.60,
            occupancy_ramp=[0.30, 0.58, 0.80, 0.90],
            op_years=4,
            sale_year_index=4,
            bankability_with_anchor=4,
            bankability_without_anchor=2,
            fit_with_ioannina=3,
            optionality=3,
            notes="Attractive deterministic IRR, but higher exit risk.",
        ),
    }


def make_scenarios() -> Dict[str, Scenario]:
    return {
        "Base": Scenario(name="Base"),
        "Downside": Scenario(
            name="Downside",
            rent_multiplier=0.92,
            capex_multiplier=1.08,
            opex_multiplier=1.08,
            exit_yield_shift=0.0075,
            debt_cost_shift=0.0080,
            occupancy_shift=-0.06,
        ),
        "Upside": Scenario(
            name="Upside",
            rent_multiplier=1.08,
            capex_multiplier=0.97,
            opex_multiplier=0.98,
            exit_yield_shift=-0.0050,
            debt_cost_shift=-0.0040,
            occupancy_shift=0.03,
        ),
    }


def make_risk_register() -> pd.DataFrame:
    rows = [
        ["Lease-up delay", "Turnkey / Build-to-exit", "Ioannina has less public multi-MW absorption evidence than primary hubs", "Anchor-first marketing, phased scope release"],
        ["Pricing miss", "Turnkey", "Higher rent hurdle needed to justify fit-out capex", "Do not spec-fit without tenant visibility"],
        ["Exit-yield widening", "Build-to-exit", "Sale thesis is sensitive to buyer pricing", "Underwrite hold as fallback"],
        ["Bankability / leverage", "Powered Shell / Build-to-exit", "Speculative income profile may need lower leverage", "Model lower leverage until pre-let"],
        ["Scope mismatch", "Turnkey", "Wrong fit-out design can trap capex", "Keep base design modular"],
        ["Counterparty weakness", "Build-to-exit / Powered Shell", "Secondary-city occupier pool may be weaker", "Guarantees, deposits, covenants"],
        ["Power / schedule slip", "All", "July 2027 target still carries execution risk", "Utility milestones and float"],
        ["Thin buyer pool", "Build-to-exit", "Ioannina is not a proven prime exit market", "Do not force a sale date"],
        ["Under-monetization", "Powered Shell", "Too conservative if a strong anchor appears", "Pre-engineer turnkey conversion"],
    ]
    return pd.DataFrame(rows, columns=["Risk", "Most_Exposed_Option", "Why_It_Matters", "Mitigation"])


def export_outputs(outputs: Dict[str, pd.DataFrame], annual_tables: Dict[Tuple[str, str], pd.DataFrame], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for name, df in outputs.items():
        file_name = name.lower().replace(" ", "_") + ".csv"
        df.to_csv(outdir / file_name, index=False)

    for (strategy, scenario), df in annual_tables.items():
        safe = f"cashflow_{strategy}_{scenario}".lower().replace(" ", "_").replace("-", "_")
        df.to_csv(outdir / f"{safe}.csv", index=False)

    workbook_path = outdir / "ioannina_dc_detailed_cashflows.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, df in outputs.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
        for (strategy, scenario), df in annual_tables.items():
            sheet = f"{strategy[:18]}_{scenario[:10]}"[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)


def run_model() -> tuple[Dict[str, pd.DataFrame], Dict[Tuple[str, str], pd.DataFrame]]:
    ga = GlobalAssumptions()
    strategies = make_strategies()
    scenarios = make_scenarios()

    summary_rows = []
    decision_rows = []
    mc_rows = []
    annual_tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    sensitivities: Dict[str, pd.DataFrame] = {}

    for strategy in strategies.values():
        base_summary = None
        downside_summary = None
        upside_summary = None

        for scenario in scenarios.values():
            detailed = build_detailed_cashflows(strategy, ga, scenario)
            annual_tables[(strategy.name, scenario.name)] = detailed["annual_table"]
            summary = summarize_strategy(strategy, ga, scenario)
            summary_rows.append(summary)
            if scenario.name == "Base":
                base_summary = summary
            elif scenario.name == "Downside":
                downside_summary = summary
            elif scenario.name == "Upside":
                upside_summary = summary

        mc_df = monte_carlo(strategy, ga)
        mc_summary = summarize_mc(mc_df, ga.hurdle_equity_irr)
        mc_rows.append({"Strategy": strategy.name, **mc_summary})

        assert downside_summary is not None and upside_summary is not None
        decision_rows.append(score_option(strategy, downside_summary["Project_NPV_9pct"], upside_summary["Equity_IRR"], mc_summary))
        sensitivities[strategy.name] = one_way_sensitivity(strategy, ga, scenarios["Base"])

    outputs = {
        "scenario_summary": pd.DataFrame(summary_rows),
        "mc_summary": pd.DataFrame(mc_rows),
        "decision_matrix": pd.DataFrame(decision_rows).sort_values("Weighted_Score_100", ascending=False),
        "risk_register": make_risk_register(),
        "sensitivity_turnkey": sensitivities["Turnkey"],
        "sensitivity_powered_shell": sensitivities["Powered Shell"],
        "sensitivity_build_to_exit": sensitivities["Build-to-exit"],
    }
    return outputs, annual_tables


if __name__ == "__main__":
    outdir = Path.cwd() / "outputs"
    outputs, annual_tables = run_model()
    export_outputs(outputs, annual_tables, outdir)

    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", lambda x: f"{x:,.2f}"):
        print("\nSCENARIO SUMMARY")
        print(outputs["scenario_summary"])
        print("\nDECISION MATRIX")
        print(outputs["decision_matrix"])
        print("\nMONTE CARLO SUMMARY")
        print(outputs["mc_summary"])

    best = outputs["decision_matrix"].iloc[0]
    print(f"\nBest option now: {best['Strategy']} ({best['Weighted_Score_100']:.1f}/100)")
    print(f"Detailed annual cashflow CSVs and Excel workbook exported to {outdir.resolve()}")

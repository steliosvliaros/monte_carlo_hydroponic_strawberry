from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# FROZEN MODEL LABEL
# ============================================================
# ioannina_dc_strategy_v1
# This script reproduces the logic behind the frozen recommendation:
# 1) Best base-case option now: Powered Shell
# 2) Build-to-exit is attractive on deterministic IRR, but fragile under
#    exit-yield / lease-up uncertainty
# 3) Speculative Turnkey only wins if anchor demand is strong enough to justify
#    the extra capex and design-specific risk
#
# IMPORTANT:
# - These are underwriting assumptions, not market facts.
# - The numbers are calibrated to approximate the previously frozen answer.
# - Update the assumptions below when you have better tenant, lender, or EPC data.
# ============================================================


@dataclass
class Strategy:
    """Dogtags: #data-model #strategy #capital-plan.

    Describes one development strategy and the full set of underwriting inputs
    used to evaluate it.
    """

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
    sale_year_index: int | None = None  # 1-indexed op year for exit strategy
    capex_split: Tuple[float, float] = (0.7, 0.3)  # build year 1 / build year 2
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
    """Dogtags: #shared-inputs #discounting #simulation.

    Holds assumptions that apply across all strategies and scenario runs.
    """

    it_load_mw: float = 12.0
    discount_rate_project: float = 0.09
    hurdle_equity_irr: float = 0.15
    energy_pass_through: bool = True
    iterations: int = 5000
    random_seed: int = 42


@dataclass
class Scenario:
    """Dogtags: #scenario #stress-test #market-variation.

    Stores scenario-level shifts and multipliers layered on top of a strategy.
    """

    name: str
    rent_multiplier: float = 1.0
    capex_multiplier: float = 1.0
    opex_multiplier: float = 1.0
    exit_yield_shift: float = 0.0
    debt_cost_shift: float = 0.0
    occupancy_shift: float = 0.0  # absolute deduction/addition to each ramp point


def kw_from_mw(mw: float) -> float:
    """Dogtags: #utility #unit-conversion.

    Convert megawatts to kilowatts.

    Args:
        mw: IT load in megawatts.

    Returns:
        Equivalent load in kilowatts.
    """
    return mw * 1000.0


def clamp(x: float, lo: float, hi: float) -> float:
    """Dogtags: #utility #bounds-check.

    Restrict a value to the inclusive range ``[lo, hi]``.
    """
    return max(lo, min(hi, x))


def annuity_payment(principal: float, annual_rate: float, tenor_years: int) -> float:
    """Dogtags: #debt #finance #amortization.

    Calculate the annual payment for a fully amortizing loan.

    Args:
        principal: Debt amount drawn at closing.
        annual_rate: Annual interest rate as a decimal.
        tenor_years: Loan tenor in years.

    Returns:
        Annual debt service payment.
    """
    if principal <= 0:
        return 0.0
    if annual_rate == 0:
        return principal / tenor_years
    r = annual_rate
    return principal * r / (1.0 - (1.0 + r) ** (-tenor_years))


def blended_terminal_value(next_noi: float, exit_yield: float, tdc: float, strategy: Strategy) -> float:
    """Dogtags: #valuation #terminal-value #exit.

    Estimate exit value from forward NOI using the income-capitalization
    approach, with replacement cost acting only as a floor.
    """
    income_value = next_noi / exit_yield
    replacement_cost_value = tdc * strategy.replacement_cost_exit_ratio
    gross_terminal_value = max(income_value, replacement_cost_value)
    sale_cost_pct = clamp(strategy.exit_transaction_cost_pct, 0.0, 0.25)
    return gross_terminal_value * (1.0 - sale_cost_pct)


def build_cashflows(strategy: Strategy, global_assumptions: GlobalAssumptions, scenario: Scenario) -> Dict[str, object]:
    """Dogtags: #core-engine #cashflows #valuation.

    Build project-level and equity-level cash flows across the construction,
    operating, financing, and terminal-value periods.

    Timeline convention:
        t=0 and t=1 are build years (negative capex).
        t=2 onward are operating years 1..N.

    Args:
        strategy: Development option being underwritten.
        global_assumptions: Shared project-wide inputs.
        scenario: Case-specific stress or upside/downside adjustments.

    Returns:
        A dictionary containing cash-flow arrays, KPIs, and an annual detail
        table for further analysis.
    """
    it_kw = kw_from_mw(global_assumptions.it_load_mw)

    tdc = strategy.tdc_eur * scenario.capex_multiplier
    rent = strategy.rent_eur_per_kw_month * scenario.rent_multiplier
    opex_year1 = strategy.opex_year1_eur * scenario.opex_multiplier
    exit_yield = strategy.exit_yield + scenario.exit_yield_shift
    debt_cost = strategy.debt_cost + scenario.debt_cost_shift

    if exit_yield <= 0:
        raise ValueError("Exit yield must stay positive.")
    if debt_cost < 0:
        raise ValueError("Debt cost must stay non-negative.")
    if strategy.replacement_cost_exit_ratio <= 0:
        raise ValueError("Replacement-cost exit ratio must stay positive.")

    build_cf = np.zeros(strategy.build_years)
    build_cf[0] = -tdc * strategy.capex_split[0]
    build_cf[1] = -tdc * strategy.capex_split[1]

    debt_draw = strategy.debt_ratio * tdc
    equity_draw = tdc - debt_draw
    equity_build_cf = np.zeros(strategy.build_years)
    equity_build_cf[0] = -equity_draw * strategy.capex_split[0]
    equity_build_cf[1] = -equity_draw * strategy.capex_split[1]

    annual_debt_service = annuity_payment(debt_draw, debt_cost, strategy.debt_tenor_years)
    debt_balance = debt_draw

    op_cf_project: List[float] = []
    op_cf_equity: List[float] = []
    annual_rows = []
    min_dscr = float("inf")

    for op_year in range(1, strategy.op_years + 1):
        occ = strategy.occupancy_ramp[min(op_year - 1, len(strategy.occupancy_ramp) - 1)]
        occ = clamp(occ + scenario.occupancy_shift, 0.0, 1.0)

        revenue = it_kw * rent * 12.0 * occ * ((1.0 + strategy.lease_escalation) ** (op_year - 1))
        opex = opex_year1 * ((1.0 + strategy.opex_escalation) ** (op_year - 1))
        noi = revenue - opex

        interest = debt_balance * debt_cost
        principal = max(0.0, annual_debt_service - interest)
        principal = min(principal, debt_balance)
        debt_service = interest + principal
        debt_balance = max(0.0, debt_balance - principal)

        dscr = noi / debt_service if debt_service > 0 else np.inf
        min_dscr = min(min_dscr, dscr)

        project_cf = noi
        equity_cf = noi - debt_service

        terminal_value = 0.0
        if strategy.sale_year_index is not None:
            if op_year == strategy.sale_year_index:
                next_revenue = it_kw * rent * 12.0 * occ * ((1.0 + strategy.lease_escalation) ** op_year)
                next_opex = opex_year1 * ((1.0 + strategy.opex_escalation) ** op_year)
                next_noi = next_revenue - next_opex
                terminal_value = blended_terminal_value(next_noi, exit_yield, tdc, strategy)
        elif op_year == strategy.op_years:
            next_occ = strategy.occupancy_ramp[min(op_year, len(strategy.occupancy_ramp) - 1)]
            next_occ = clamp(next_occ + scenario.occupancy_shift, 0.0, 1.0)
            next_revenue = it_kw * rent * 12.0 * next_occ * ((1.0 + strategy.lease_escalation) ** op_year)
            next_opex = opex_year1 * ((1.0 + strategy.opex_escalation) ** op_year)
            next_noi = next_revenue - next_opex
            terminal_value = blended_terminal_value(next_noi, exit_yield, tdc, strategy)

        if terminal_value > 0:
            project_cf += terminal_value
            equity_cf += max(0.0, terminal_value - debt_balance)

        op_cf_project.append(project_cf)
        op_cf_equity.append(equity_cf)
        annual_rows.append(
            {
                "op_year": op_year,
                "occupancy": occ,
                "revenue": revenue,
                "opex": opex,
                "noi": noi,
                "interest": interest,
                "principal": principal,
                "debt_service": debt_service,
                "debt_balance_end": debt_balance,
                "project_cf": project_cf,
                "equity_cf": equity_cf,
                "terminal_value_added": terminal_value,
                "dscr": dscr,
            }
        )

        if strategy.sale_year_index is not None and op_year == strategy.sale_year_index:
            break

    cf_project = np.concatenate([build_cf, np.array(op_cf_project)])
    cf_equity = np.concatenate([equity_build_cf, np.array(op_cf_equity)])

    npv_project = npf_npv(global_assumptions.discount_rate_project, cf_project)
    irr_project = npf_irr(cf_project)
    irr_equity = npf_irr(cf_equity)

    return {
        "cf_project": cf_project,
        "cf_equity": cf_equity,
        "npv_project": npv_project,
        "irr_project": irr_project,
        "irr_equity": irr_equity,
        "min_dscr": min_dscr,
        "annual_table": pd.DataFrame(annual_rows),
        "tdc": tdc,
        "rent": rent,
        "exit_yield": exit_yield,
    }


def npf_npv(rate: float, values: np.ndarray) -> float:
    """Dogtags: #valuation #npv.

    Compute net present value for a stream of periodic cash flows.
    """
    return float(sum(v / ((1 + rate) ** i) for i, v in enumerate(values)))


def npf_irr(values: np.ndarray) -> float:
    """Dogtags: #valuation #irr #numerical-method.

    Estimate the internal rate of return with a bisection solver that avoids
    adding an external financial-math dependency.
    """
    # Safe IRR using numpy_financial-like bisection without extra dependency
    if np.all(values >= 0) or np.all(values <= 0):
        return float("nan")

    def f(r: float) -> float:
        return sum(v / ((1 + r) ** i) for i, v in enumerate(values))

    lo, hi = -0.99, 5.0
    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo * fhi > 0 and tries < 25:
        hi *= 2
        fhi = f(hi)
        tries += 1
    if flo * fhi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < 1e-8:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return mid


def summarize_strategy(strategy: Strategy, ga: GlobalAssumptions, scenario: Scenario) -> Dict[str, float]:
    """Dogtags: #summary #kpis #reporting.

    Convert full cash-flow output into a compact set of decision-ready KPIs.
    """
    res = build_cashflows(strategy, ga, scenario)
    annual = res["annual_table"]
    stabilized_noi = float(annual["noi"].iloc[-1])
    terminal_value = float(annual["terminal_value_added"].iloc[-1])

    # Break-even rent with last occupancy as reference
    occ_ref = max(0.01, float(annual["occupancy"].iloc[-1]))
    rent_be = find_break_even_rent(strategy, ga, scenario)
    occ_be = find_break_even_occupancy(strategy, ga, scenario)

    return {
        "Strategy": strategy.name,
        "Scenario": scenario.name,
        "TDC_EUR": res["tdc"],
        "Rent_EUR_per_kW_month": res["rent"],
        "Exit_Yield": res["exit_yield"],
        "Project_IRR": res["irr_project"],
        "Equity_IRR": res["irr_equity"],
        "Project_NPV_9pct": res["npv_project"],
        "Min_DSCR": res["min_dscr"],
        "Stabilized_NOI": stabilized_noi,
        "Terminal_or_Sale_Value": terminal_value,
        "Break_even_Rent_EUR_per_kW_month": rent_be,
        "Break_even_Stabilized_Occupancy": occ_be,
    }


def find_break_even_rent(strategy: Strategy, ga: GlobalAssumptions, scenario: Scenario) -> float:
    """Dogtags: #break-even #rent #binary-search.

    Estimate the monthly rent per kW required for project NPV to reach zero.
    """
    lo, hi = 1.0, 300.0
    for _ in range(120):
        mid = (lo + hi) / 2
        temp = Strategy(**{**strategy.__dict__, "rent_eur_per_kw_month": mid})
        npv = build_cashflows(temp, ga, scenario)["npv_project"]
        if npv >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def find_break_even_occupancy(strategy: Strategy, ga: GlobalAssumptions, scenario: Scenario) -> float:
    """Dogtags: #break-even #occupancy #binary-search.

    Estimate the stabilized occupancy needed for project NPV to reach zero.
    """
    lo, hi = 0.05, 0.99
    for _ in range(120):
        mid = (lo + hi) / 2
        temp_ramp = [min(mid, x) for x in strategy.occupancy_ramp]
        temp = Strategy(**{**strategy.__dict__, "occupancy_ramp": temp_ramp})
        npv = build_cashflows(temp, ga, scenario)["npv_project"]
        if npv >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def monte_carlo(strategy: Strategy, ga: GlobalAssumptions) -> pd.DataFrame:
    """Dogtags: #simulation #risk #monte-carlo.

    Run randomized underwriting cases around the frozen base assumptions.
    """
    rng = np.random.default_rng(ga.random_seed)
    rows = []
    for _ in range(ga.iterations):
        # Triangular distributions around the frozen answer assumptions.
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
        else:  # Build-to-exit is more fragile on timing and exit yield
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
        res = build_cashflows(strategy, ga, scenario)
        rows.append(
            {
                "npv": res["npv_project"],
                "project_irr": res["irr_project"],
                "equity_irr": res["irr_equity"],
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
    """Dogtags: #simulation #risk-summary #percentiles.

    Summarize Monte Carlo results into percentiles and failure probabilities.
    """
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
    """Dogtags: #sensitivity #stress-test #npv-delta.

    Test one input shock at a time and measure the change in project NPV.
    """
    base = build_cashflows(strategy, ga, base_scenario)["npv_project"]
    tests = {
        "rent_-10%": Scenario("rent_-10", rent_multiplier=0.90),
        "rent_+10%": Scenario("rent_+10", rent_multiplier=1.10),
        "capex_+10%": Scenario("capex_+10", capex_multiplier=1.10),
        "capex_-5%": Scenario("capex_-5", capex_multiplier=0.95),
        "exit_yield_+100bps": Scenario("exit_+100", exit_yield_shift=0.01),
        "exit_yield_-50bps": Scenario("exit_-50", exit_yield_shift=-0.005),
        "debt_+100bps": Scenario("debt_+100", debt_cost_shift=0.01),
        "occupancy_-5pts": Scenario("occ_-5", occupancy_shift=-0.05),
    }
    rows = []
    for name, sc in tests.items():
        res = build_cashflows(strategy, ga, sc)
        rows.append({"variable": name, "npv_delta": res["npv_project"] - base})
    return pd.DataFrame(rows).sort_values("npv_delta")


def score_option(
    strategy: Strategy,
    base_summary: Dict[str, float],
    downside_summary: Dict[str, float],
    upside_summary: Dict[str, float],
    mc_summary: Dict[str, float],
) -> Dict[str, float]:
    """Dogtags: #decision-matrix #scoring #ranking.

    Blend model outputs with qualitative judgments into a weighted option score.
    """
    # Scores are 1-5, combining model output and market-judgment assumptions.
    risk_adjusted_return = 5 if mc_summary["Prob_EqIRR_lt_Hurdle"] < 0.05 else 4 if mc_summary["Prob_EqIRR_lt_Hurdle"] < 0.15 else 3 if mc_summary["Prob_EqIRR_lt_Hurdle"] < 0.30 else 2
    downside_resilience = 5 if downside_summary["Project_NPV_9pct"] > 0 else 3 if downside_summary["Project_NPV_9pct"] > -5_000_000 else 2
    absolute_upside = 5 if upside_summary["Equity_IRR"] > 0.30 else 4 if upside_summary["Equity_IRR"] > 0.22 else 3
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

    weighted = sum(weights[k] * scores[k] for k in weights) / 5.0
    out = {"Strategy": strategy.name, "Weighted_Score_100": weighted}
    out.update({f"score_{k}": v for k, v in scores.items()})
    return out


def make_strategies() -> Dict[str, Strategy]:
    """Dogtags: #factory #strategy-setup.

    Build the predefined set of development strategies used by the model.
    """
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
    """Dogtags: #factory #scenario-setup.

    Build the standard Base, Downside, and Upside scenario definitions.
    """
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
    """Dogtags: #risk-register #qualitative-risk.

    Create a tabular view of the main execution and market risks plus mitigants.
    """
    return pd.DataFrame(
        [
            ["Lease-up delay", "Turnkey", "High capex needs faster absorption in Ioannina", "Anchor-first marketing, phased release"],
            ["Pricing miss", "Turnkey", "Higher rent hurdle to justify fit-out", "Only spec-fit with tenant visibility"],
            ["Exit-yield widening", "Build-to-exit", "Sale thesis depends on buyer pricing", "Underwrite hold as fallback"],
            ["Bankability / leverage", "Powered Shell", "Speculative shell may need lower leverage", "Model 50/50 and sponsor support"],
            ["Scope mismatch", "Turnkey", "Wrong fit-out design can trap capex", "Keep modular base design"],
            ["Counterparty weakness", "Build-to-exit", "Secondary-city occupier pool may be weaker", "Guarantees, deposits, covenants"],
            ["Power/schedule slip", "All", "July 2027 target still has execution risk", "Float, utility milestones, staged commitments"],
            ["Thin buyer pool", "Build-to-exit", "Ioannina is not a proven prime exit market", "Do not force a sale on a fixed date"],
            ["Under-monetization", "Powered Shell", "Too conservative if strong anchor arrives", "Pre-engineer turnkey conversion"],
        ],
        columns=["Risk", "Most_Exposed_Option", "Why_It_Matters", "Mitigation"],
    )


def make_swot() -> Dict[str, Dict[str, List[str]]]:
    """Dogtags: #swot #strategy-review #qualitative-analysis.

    Return SWOT-style observations for each strategy option.
    """
    return {
        "Turnkey": {
            "Strengths": [
                "Higher absolute NOI after lease-up",
                "Stronger lender story with a long contracted lease",
                "Good fit for sovereign / public / regulated workloads",
            ],
            "Weaknesses": [
                "Highest capex at risk",
                "Tenant-specific design mismatch risk",
                "Harder to reverse once committed",
            ],
            "Opportunities": [
                "Anchor-led BTS",
                "Public-sector resilience or compliance-heavy users",
                "Premium pricing if speed + customization are valued",
            ],
            "Threats": [
                "Speculative over-fit-out",
                "Delayed lease-up",
                "Exit-yield widening",
            ],
        },
        "Powered Shell": {
            "Strengths": [
                "Lowest capital at risk",
                "Best optionality",
                "Strong fit for a secondary-market city",
            ],
            "Weaknesses": [
                "Lower immediate NOI",
                "Some tenants prefer more delivered scope",
                "Weaker speculative bankability than leased turnkey",
            ],
            "Opportunities": [
                "Regional operator / MSP / carrier-neutral demand",
                "Tenant-controlled fit-out",
                "Future turnkey conversion after anchor signing",
            ],
            "Threats": [
                "Under-monetization if a strong anchor appears",
                "Tenant coordination delays",
                "Slower early revenue ramp",
            ],
        },
        "Build-to-exit": {
            "Strengths": [
                "Highest headline IRR in the base case",
                "Shorter capital hold period",
                "Good for capital recycling strategy",
            ],
            "Weaknesses": [
                "Very sensitive to sale timing",
                "More exposed to exit-yield widening",
                "Less forgiving if lease-up slips",
            ],
            "Opportunities": [
                "Could work well if a credible tenant/operator is signed",
                "Appeal to infrastructure / RE buyers once de-risked",
            ],
            "Threats": [
                "Ioannina is not a proven prime exit market",
                "Fixed sale horizon can force bad timing",
                "Buyer pool may be thinner than primary hubs",
            ],
        },
    }


def print_table(title: str, df: pd.DataFrame) -> None:
    """Dogtags: #display #console-output.

    Print a DataFrame using wider formatting for terminal review.
    """
    print("\n" + title)
    print("=" * len(title))
    with pd.option_context("display.max_columns", None, "display.width", 180, "display.float_format", lambda x: f"{x:,.2f}"):
        print(df)


def run_model() -> Dict[str, pd.DataFrame]:
    """Dogtags: #orchestration #entrypoint #reporting.

    Run the full underwriting workflow and return the main output tables.
    """
    ga = GlobalAssumptions()
    strategies = make_strategies()
    scenarios = make_scenarios()

    base_rows = []
    decision_rows = []
    mc_rows = []
    sensitivity_outputs: Dict[str, pd.DataFrame] = {}

    for strategy in strategies.values():
        base = summarize_strategy(strategy, ga, scenarios["Base"])
        downside = summarize_strategy(strategy, ga, scenarios["Downside"])
        upside = summarize_strategy(strategy, ga, scenarios["Upside"])
        base_rows.extend([base, downside, upside])

        mc_df = monte_carlo(strategy, ga)
        mc_summary = summarize_mc(mc_df, ga.hurdle_equity_irr)
        mc_rows.append({"Strategy": strategy.name, **mc_summary})

        decision_rows.append(score_option(strategy, base, downside, upside, mc_summary))
        sensitivity_outputs[strategy.name] = one_way_sensitivity(strategy, ga, scenarios["Base"])

    scenario_df = pd.DataFrame(base_rows)
    mc_df = pd.DataFrame(mc_rows)
    decision_df = pd.DataFrame(decision_rows).sort_values("Weighted_Score_100", ascending=False)

    return {
        "scenario_summary": scenario_df,
        "mc_summary": mc_df,
        "decision_matrix": decision_df,
        "risk_register": make_risk_register(),
        "sensitivity_Turnkey": sensitivity_outputs["Turnkey"],
        "sensitivity_Powered Shell": sensitivity_outputs["Powered Shell"],
        "sensitivity_Build-to-exit": sensitivity_outputs["Build-to-exit"],
    }


if __name__ == "__main__":
    outputs = run_model()
    for name, df in outputs.items():
        print_table(name, df)

    best = outputs["decision_matrix"].iloc[0]
    print("\nFrozen recommendation reproduced by the model:")
    print(f"Best option now: {best['Strategy']} (weighted score {best['Weighted_Score_100']:.1f}/100)")

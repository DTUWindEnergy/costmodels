"""Financial utilities for evaluating energy projects.

This module collects helper functions for common financial calculations such as
net present value (NPV), internal rate of return (IRR) and levelized cost of
energy (LCO).  Most of the computations rely on JAX in order to allow
differentiable and vectorized execution where possible.
"""

import pickle
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import partial

import jax
import jax.numpy as np
from jax import tree_util
from jax.scipy.optimize import minimize


def _irr(cashflows):
    res = np.roots(cashflows[::-1], strip_zeros=False)
    mask = (res.imag == 0) & (res.real > 0)

    def true_fn(args):
        res, mask = args
        rates = 1.0 / res.real - 1.0
        valid_rates = np.where(mask, np.abs(rates), np.inf)
        best_idx = np.argmin(valid_rates)
        return rates[best_idx]

    def false_fn(_):
        return np.nan

    return jax.lax.cond(np.any(mask), true_fn, false_fn, (res, mask))


def _npv(rate, cashflows):
    """
    Calculates the Net Present Value (NPV) of a series of cashflows using JAX.

    The formula used is: NPV = sum_{i=0 to n-1} [ cashflows[i] / (1 + rate)^i ]
    where cashflows[0] is the cash flow at t=0.

    Args:
        rate: The discount rate (scalar).
        cashflows: A JAX array or array-like structure of cash flow values.
                   The first value (cashflows[0]) is assumed to be at time t=0.

    Returns:
        The NPV as a JAX scalar.
    """
    cashflows_arr = np.asarray(cashflows)
    periods = np.arange(cashflows_arr.shape[0], dtype=cashflows_arr.dtype)
    discount_factors = (1 + np.asarray(rate, dtype=cashflows_arr.dtype)) ** periods
    discounted_cashflows = cashflows_arr / discount_factors
    npv = np.sum(discounted_cashflows)
    return npv


def _annual_revenue(technologies, product_prices, ny):
    annual_revenue = np.zeros(ny)
    for t in technologies:
        t0 = t.t0
        lifetime = t.lifetime
        penalty = np.zeros_like(t.production)  # TODO: or t.penalty
        annual_revenue = annual_revenue.at[t0 : lifetime + t0].add(
            np.sum(
                np.asarray(
                    np.split(
                        np.asarray(t.production) * np.asarray(product_prices[t.product])
                        - penalty,
                        lifetime,
                    )
                ),
                axis=1,
            )
        )
    return annual_revenue


def _annual_production(technologies, ny):
    annual_energy_production = np.zeros(ny)
    for t in technologies:
        t0 = t.t0
        lifetime = t.lifetime
        non_rev = t.non_revenue_production
        annual_energy_production = annual_energy_production.at[t0 : lifetime + t0].add(
            np.sum(
                np.asarray(np.split(np.asarray(t.production) + non_rev, lifetime)),
                axis=1,
            )
        )
    return annual_energy_production


def _wacc(capexs, waccs, shared_capex):
    """This function returns the weighted average cost of capital after tax, using solar, wind, and battery
    WACC. First the shared costs WACC is computed by taking the mean of the WACCs across all technologies.
    Then the WACC after tax is calculated by taking the weighted sum by the corresponding CAPEX.

    Parameters
    ----------
    capexs : CAPEX for each technology
    shared_capex : CAPEX of the shared cost e.g. electrical costs
    waccs : After tax WACC for each technology

    Returns
    -------
    WACC_after_tax : WACC after tax
    """

    # Weighted average cost of capital
    WACC_after_tax = (
        np.sum(np.asarray(capexs) * np.asarray(waccs))
        + shared_capex * np.mean(np.asarray(waccs))
    ) / (np.sum(np.asarray(capexs)) + shared_capex)
    return WACC_after_tax


def _inflation_index(yr, inflation):
    """Compute inflation index via linear interpolation in JAX."""
    # to JAX arrays
    years = np.asarray(yr)
    x = np.asarray(inflation.year)
    y = np.asarray(inflation.rate)
    # find right bin for each query year
    idx = np.searchsorted(x, years, side="right")
    idx = np.clip(idx, 1, x.shape[0] - 1)
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    # linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    infl = y0 + slope * (years - x0)
    # cumulative product and normalization at reference year
    infl_idx = np.cumprod(1.0 + infl)
    ref_mask = years == inflation.year_ref
    ref_idx = np.argmax(ref_mask)  # first match
    return infl_idx / infl_idx[ref_idx]


def _capex_phasing(
    capex,
    phasing_yr,
    phasing_capex,
    discount_rate,
    inflation_index,
):
    """This function calulates the equivalent net present value CAPEX given a early paying "phasing" approach.

    Parameters
    ----------
    CAPEX : CAPEX
    phasing_yr : Yearly early paying of CAPEX curve. x-axis, time in years.
    phasing_CAPEX : Yearly early paying of CAPEX curve. Shares will be normalized to sum the CAPEX.
    discount_rate : Discount rate for present value calculation
    inflation_index : Inflation index time series at the phasing_yr years. Accounts for inflation.

    Returns
    -------
    CAPEX_eq : Present value equivalent CAPEX
    """

    phasing_capex = inflation_index * capex * phasing_capex / np.sum(phasing_capex)
    capex_eq = np.sum(
        np.asarray(
            [
                phasing_capex[ii] / (1 + discount_rate) ** yr
                for ii, yr in enumerate(phasing_yr)
            ]
        )
    )

    return capex_eq


def _break_even_price(
    product,
    CAPEX,
    OPEX,
    tax_rate,
    discount_rate,
    depreciation_yr,
    depreciation,
    DEVEX,
    inflation_index,
    technologies,
    generations,
    product_prices,
    ny,
):
    product_prices_temp = product_prices.copy()

    def fun(price):
        product_prices_temp[product] = (
            np.ones_like(product_prices_temp[product]) * price
        )
        revenues = _annual_revenue(technologies, generations, product_prices_temp, ny)
        cashflow = _cashflow(
            net_revenue_t=revenues,
            investment_cost=CAPEX,
            maintenance_cost_per_year=OPEX,
            tax_rate=tax_rate,
            discount_rate=discount_rate,
            depreciation_yr=depreciation_yr,
            depreciation=depreciation,
            development_cost=DEVEX,
            inflation_index=inflation_index,
        )
        NPV = _npv(discount_rate, cashflow)
        return NPV**2

    out = minimize(fun=fun, x0=np.asarray([50.0]), method="BFGS", tol=1e-10)
    return out.x[0]


def _cashflow(
    net_revenue_t,
    investment_cost,
    maintenance_cost_per_year,
    tax_rate,
    depreciation,
    development_cost,
    inflation_index,
):
    """A function to estimate the yearly cashflow using the net revenue time series, and the yearly OPEX costs.
    It then calculates the NPV and IRR using the yearly cashlow, the CAPEX, the WACC after tax, and the tax rate.

    Parameters
    ----------
    Net_revenue_t : Net revenue time series
    investment_cost : Capital costs
    maintenance_cost_per_year : yearly operation and maintenance costs
    tax_rate : tax rate
    discount_rate : Discount rate
    depreciation_yr : Depreciation curve (x-axis) time in years
    depreciation : Depreciation curve at the given times
    development_cost : DEVEX
    inflation_index : Yearly Inflation index time-sereis

    Returns
    -------
    NPV : Net present value
    IRR : Internal rate of return
    """

    yr = np.arange(
        len(net_revenue_t) + 1
    )  # extra year to start at 0 and end at end of lifetime.
    depre = np.interp(
        np.asarray(yr), np.asarray(depreciation.year), np.asarray(depreciation.rate)
    )

    # EBITDA: earnings before interest and taxes in nominal prices
    EBITDA = (net_revenue_t - maintenance_cost_per_year) * inflation_index[1:]

    # EBIT taxable income
    depreciation_on_each_year = np.diff(investment_cost * depre)
    EBIT = EBITDA - depreciation_on_each_year

    # Taxes
    Taxes = EBIT * tax_rate

    Net_income = EBITDA - Taxes
    Cashflow = np.insert(Net_income, 0, -investment_cost - development_cost)
    return Cashflow


def _phased_capex(technologies, shared_capex, inflation, phasing_yr, global_t_neg):
    """Return discounted CAPEX and discount factor for a set of technologies.

    The function aggregates CAPEX phasing from each technology and computes the
    equivalent present value CAPEX considering inflation and a weighted average
    cost of capital.
    """

    capexs = [t.CAPEX for t in technologies]
    waccs = [t.WACC for t in technologies]

    phasing_capex = np.zeros_like(phasing_yr, dtype=float)
    for t in technologies:
        for y, c in zip(t.phasing_yr, t.phasing_capex):
            phasing_capex = phasing_capex.at[y + t.t0 - global_t_neg].add(c * t.CAPEX)

    discount_rate = _wacc(capexs, waccs, shared_capex)
    inflation_index_phasing = _inflation_index(yr=phasing_yr, inflation=inflation)
    capex_eq = _capex_phasing(
        capex=np.sum(np.asarray(capexs)) + shared_capex,
        phasing_yr=phasing_yr,
        phasing_capex=phasing_capex,
        discount_rate=discount_rate,
        inflation_index=inflation_index_phasing,
    )

    return capex_eq, discount_rate


def _annual_costs(technologies, ny):
    """Compute annual OPEX, consumption costs and production."""

    annual_operational_cost = np.zeros(ny)
    annual_consumption_cost = np.zeros(ny)

    for t in technologies:
        lifetime = t.lifetime
        t0 = t.t0
        annual_operational_cost = annual_operational_cost.at[t0 : lifetime + t0].set(
            annual_operational_cost[t0 : lifetime + t0]
            + np.broadcast_to(t.OPEX, lifetime)
        )

        consumption = t.consumption
        if np.size(consumption) > lifetime:
            c = np.sum(np.split(consumption, lifetime), axis=1)
        elif np.size(consumption) == lifetime:
            c = consumption
        else:
            c = np.broadcast_to(consumption, lifetime)
        annual_consumption_cost = annual_consumption_cost.at[t0 : lifetime + t0].add(c)

    annual_energy_production = _annual_production(technologies, ny)

    return annual_operational_cost, annual_consumption_cost, annual_energy_production


def _compute_lco(
    annual_operational_cost,
    annual_consumption_cost,
    annual_energy_production,
    capex_eq,
    discount_rate,
    iy,
):
    """Calculate the levelized cost of output from yearly costs and production."""

    level_costs = (
        np.sum(
            (annual_operational_cost + annual_consumption_cost)
            / (1 + discount_rate) ** iy
        )
        + capex_eq
    )
    level_aep = np.sum(annual_energy_production / (1 + discount_rate) ** iy)

    lco = jax.lax.cond(
        level_aep > 0,
        lambda _: level_costs / level_aep,
        lambda _: 1e6,
        operand=None,
    )
    return lco


def _product_specific_finance(
    technologies,
    shared_capex,
    inflation,
    t0s,
    lifetimes,
    global_t_neg,
    ny,
    iy,
    phasing_yr,
):
    """Calculate levelized costs for a subset of technologies."""
    capex_eq, hpp_discount_factor = _phased_capex(
        technologies, shared_capex, inflation, phasing_yr, global_t_neg
    )

    annual_operational_cost, annual_consumption_cost, annual_energy_production = (
        _annual_costs(technologies, ny)
    )

    LCO = _compute_lco(
        annual_operational_cost,
        annual_consumption_cost,
        annual_energy_production,
        capex_eq,
        hpp_discount_factor,
        iy,
    )
    return {
        "LCO": LCO,
        "CAPEX_eq": capex_eq,
        "annual_operational_cost": annual_operational_cost,
        "hpp_discount_factor": hpp_discount_factor,
    }


class Product(Enum):
    SPOT_ELECTRICITY = 0
    HYDROGEN = 1

    def __lt__(self, other):
        return self.value > other.value


_jax_ptree_child = partial(field, metadata={"child": True})


@tree_util.register_pytree_node_class
@dataclass
class Technology:
    # dynamic
    CAPEX: float = _jax_ptree_child()
    OPEX: float = _jax_ptree_child()
    production: list = _jax_ptree_child()

    # static
    name: str
    lifetime: int
    t0: int
    WACC: float
    phasing_yr: list
    phasing_capex: list
    product: Product
    non_revenue_production: list = None
    penalty: float = None
    consumption: float = 0.0

    def tree_flatten(self):
        children, aux = [], []
        for f in fields(self):
            (children if f.metadata.get("child") else aux).append(getattr(self, f.name))
        return tuple(children), tuple(aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        init_kwargs = {}
        child_it = iter(children)
        aux_it = iter(aux)
        for f in fields(cls):
            if f.metadata.get("child"):
                init_kwargs[f.name] = next(child_it)
            else:
                init_kwargs[f.name] = next(aux_it)
        return cls(**init_kwargs)


@dataclass
class Inflation:
    rate: tuple
    year: tuple
    year_ref: int

    def __hash__(self):
        vals = tuple(getattr(self, f.name) for f in fields(self))
        return hash(pickle.dumps(vals, protocol=pickle.HIGHEST_PROTOCOL))


@dataclass
class Depreciation:
    year: tuple
    rate: tuple

    def __hash__(self):
        vals = tuple(getattr(self, f.name) for f in fields(self))
        return hash(pickle.dumps(vals, protocol=pickle.HIGHEST_PROTOCOL))


@dataclass
class LCO:
    name: str
    costs: tuple[str]
    accounts_for_shared: bool = True

    def __hash__(self):
        vals = tuple(getattr(self, f.name) for f in fields(self))
        return hash(pickle.dumps(vals, protocol=pickle.HIGHEST_PROTOCOL))


def finances(
    technologies: list[Technology],
    product_prices: dict,
    shared_capex: float,
    inflation: Inflation,
    tax_rate: float,
    depreciation: Depreciation,
    devex: float,
    lcos: tuple[LCO] = None,
):
    """Compute overall project finances for a set of technologies.

    Parameters
    ----------
    technologies : list[Technology]
        Technologies taking part in the project.
    product_prices : dict
        Mapping from :class:`Product` to sale price time series.
    shared_capex : float
        Capital expenditure shared among all technologies.
    inflation : Inflation
        Inflation rates used to compute the price index.
    tax_rate : float
        Corporate tax rate.
    depreciation : Depreciation
        Depreciation schedule for the assets.
    devex : float
        Development expenditure.
    lcos : tuple[LCO], optional
        Definitions of levelized costs to evaluate.

    Returns
    -------
    dict
        Dictionary with keys like ``NPV`` and ``IRR`` along with levelized
        costs for each entry in ``lcos``.
    """
    techs = [k.name for k in technologies]
    lcos = lcos or [LCO(name="LCOE", costs=techs, accounts_for_shared=True)]
    t0s = [v.t0 for v in technologies]
    lifetimes = [v.lifetime for v in technologies]
    global_t0 = min(t0s)
    global_t1 = max([_lt + _t0 for _lt, _t0 in zip(lifetimes, t0s)])
    global_t_neg = min([v.t0 + min(v.phasing_yr) for v in technologies])
    ny = global_t1 - global_t0
    iy = np.arange(ny) + 1
    phasing_yr = np.arange(global_t1 - global_t_neg) + global_t_neg

    lcos_res = {}  # for each prpduct calculate the levelized costs
    for _, lco in enumerate(lcos):
        technologies_i = [t for t in technologies if t.name in lco.costs]
        shared_capex_i = shared_capex if lco.accounts_for_shared else 0
        res = _product_specific_finance(
            technologies_i,
            shared_capex_i,
            inflation,
            t0s,
            lifetimes,
            global_t_neg,
            ny,
            iy,
            phasing_yr,
        )
        lcos_res[lco.name] = res["LCO"]
    res = _product_specific_finance(
        technologies,
        shared_capex,
        inflation,
        t0s,
        lifetimes,
        global_t_neg,
        ny,
        iy,
        phasing_yr,
    )

    CAPEX_eq = res["CAPEX_eq"]
    annual_operational_cost = res["annual_operational_cost"]
    hpp_discount_factor = res["hpp_discount_factor"]
    cashflows = np.zeros(ny)
    inflation_index = _inflation_index(  # It includes t=0, to compute the reference
        yr=np.arange(len(cashflows) + 1), inflation=inflation
    )
    annual_revenue = _annual_revenue(technologies, product_prices, ny)
    cashflow = _cashflow(
        net_revenue_t=annual_revenue,
        investment_cost=CAPEX_eq,
        maintenance_cost_per_year=annual_operational_cost,
        tax_rate=tax_rate,
        depreciation=depreciation,
        development_cost=devex,
        inflation_index=inflation_index,
    )
    IRR = _irr(cashflow)
    NPV = _npv(hpp_discount_factor, cashflow)

    # break_even_prices = {} TODO: !!!
    # for product, _ in product_prices.items():
    #     break_even_prices[product] = calculate_break_even_price(
    #         product,
    #         CAPEX_eq,
    #         annual_operational_cost,
    #         tax_rate,
    #         hpp_discount_factor,
    #         depreciation_yr,
    #         depreciation,
    #         devex,
    #         inflation_index,
    #         technologies,
    #         generations,
    #         product_prices,
    #         ny,
    #     )
    out = {
        "NPV": NPV,
        "IRR": IRR,
        "CAPEX": CAPEX_eq,
        "OPEX": annual_operational_cost,
        "cashflow": cashflow,
        "break_even_prices": None,
    }
    out.update(lcos_res)
    return out

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
            Net_revenue_t=revenues,
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
    Net_revenue_t,
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
        len(Net_revenue_t) + 1
    )  # extra year to start at 0 and end at end of lifetime.
    depre = np.interp(
        np.asarray(yr), np.asarray(depreciation.year), np.asarray(depreciation.rate)
    )

    # EBITDA: earnings before interest and taxes in nominal prices
    EBITDA = (Net_revenue_t - maintenance_cost_per_year) * inflation_index[1:]

    # EBIT taxable income
    depreciation_on_each_year = np.diff(investment_cost * depre)
    EBIT = EBITDA - depreciation_on_each_year

    # Taxes
    Taxes = EBIT * tax_rate

    Net_income = EBITDA - Taxes
    Cashflow = np.insert(Net_income, 0, -investment_cost - development_cost)
    return Cashflow


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
    capexs = [t.CAPEX for t in technologies]
    waccs = [t.WACC for t in technologies]
    opexs = [t.OPEX for t in technologies]
    consumptions = [t.consumption for t in technologies]
    phasing_capex = np.zeros_like(phasing_yr)
    for t in technologies:
        for y, c in zip(t.phasing_yr, t.phasing_capex):
            phasing_capex = phasing_capex.at[y + t.t0 - global_t_neg].add(c * t.CAPEX)
    # Discount rate
    hpp_discount_factor = _wacc(capexs, waccs, shared_capex)
    # Apply CAPEX phasing using the inflation index for all years before the start of the project (t=0).
    inflation_index_phasing = _inflation_index(yr=phasing_yr, inflation=inflation)
    CAPEX_eq = _capex_phasing(
        capex=np.sum(np.asarray(capexs)) + shared_capex,
        phasing_yr=phasing_yr,
        phasing_capex=phasing_capex,
        discount_rate=hpp_discount_factor,
        inflation_index=inflation_index_phasing,
    )
    annual_operational_cost = np.zeros(ny)
    annual_consumption_cost = np.zeros(ny)
    for opex, lifetime, t0, consumption in zip(opexs, lifetimes, t0s, consumptions):
        annual_operational_cost = annual_operational_cost.at[t0 : lifetime + t0].set(
            annual_operational_cost[t0 : lifetime + t0]
            + np.broadcast_to(opex, lifetime)
        )  # np.ones(lifetime) * opex
        if np.size(consumption) > lifetime:
            c = np.sum(np.split(consumption, lifetime), axis=1)
        elif np.size(consumption) == lifetime:
            c = consumption
        else:
            c = np.broadcast_to(consumption, lifetime)
        annual_consumption_cost = annual_consumption_cost.at[t0 : lifetime + t0].add(c)
    annual_energy_production = _annual_production(technologies, ny)

    level_costs = (
        np.sum(
            (annual_operational_cost + annual_consumption_cost)
            / (1 + hpp_discount_factor) ** iy
        )
        + CAPEX_eq
    )
    level_AEP = np.sum(annual_energy_production / (1 + hpp_discount_factor) ** iy)

    LCO = jax.lax.cond(
        level_AEP > 0,
        lambda _: level_costs / level_AEP,  # in Euro/MWh
        lambda _: 1e6,
        operand=None,
    )
    return {
        "LCO": LCO,
        "CAPEX_eq": CAPEX_eq,
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


# @partial(
#     jax.jit,
#     static_argnames=[
#         "shared_capex",
#         "inflation",
#         "tax_rate",
#         "depreciation",
#         "DEVEX",
#         "lcos",
#     ],
# )  # TODO: remove the JIT; should be on top level function
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
        Net_revenue_t=annual_revenue,
        investment_cost=CAPEX_eq,
        maintenance_cost_per_year=annual_operational_cost,
        tax_rate=tax_rate,
        depreciation=depreciation,
        development_cost=devex,
        inflation_index=inflation_index,
    )
    IRR = _irr(cashflow)  # Quant(calculate_irr_jax(cashflow) * 100, "%")
    NPV = _npv(
        hpp_discount_factor, cashflow
    )  # Quant(calculate_npv_jax(hpp_discount_factor, cashflow), cashflows.units)

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


if __name__ == "__main__":
    import os
    import timeit
    from pathlib import Path

    import numpy
    import pandas as pd

    CAPEX_wind = 2.41170504e08
    CAPEX_solar = 66125000
    CAPEX_bess = 9882866.10284274
    CAPEX_shared = 61122845.07042254
    OPEX_wind = 4262488.80495959
    OPEX_solar = 1331250
    OPEX_bess = 0
    OPEX_shared = 0
    ts_inputs = pd.read_csv(
        Path(os.path.dirname(__file__)) / Path("../../tests/data/finance_inputs.csv"),
        index_col=0,
        sep=";",
    )
    technologies = {
        "wind": {
            "CAPEX": CAPEX_wind,
            "OPEX": OPEX_wind,
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.06,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
        },
        "solar": {
            "CAPEX": CAPEX_solar,
            "OPEX": OPEX_solar,
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.06,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
        },
        "batt": {
            "CAPEX": CAPEX_bess,
            "OPEX": OPEX_bess,
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.06,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
        },
    }
    product_prices = {Product.SPOT_ELECTRICITY: np.asarray(ts_inputs["price_t"])}

    production_sample = ts_inputs.get("p_wind", None)
    zeros = np.zeros_like(production_sample)

    # Originally the solar has battery production added here...
    # {
    #     "name": "solar_power",
    #     "technology": "solar",
    #     "production": ts_inputs["p_solar"] + ts_inputs["p_batt"],
    #     "product": "spot_electricity",
    # },
    technologies = [
        Technology(
            name=k,
            CAPEX=v["CAPEX"],
            OPEX=v["OPEX"],
            lifetime=v["lifetime"],
            t0=v["t0"],
            WACC=v["WACC"],
            phasing_yr=v["phasing_yr"],
            phasing_capex=v["phasing_capex"],
            production=np.asarray(ts_inputs.get(f"p_{k}", zeros)),
            non_revenue_production=np.zeros(len(ts_inputs.get(f"p_{k}", zeros))),
            product=Product.SPOT_ELECTRICITY,
        )
        for k, v in technologies.items()
    ]

    inflation = Inflation(
        year=[-3, 0, 1, 25],
        rate=[0.10, 0.10, 0.06, 0.06],
        year_ref=0,  # inflation index is computed with respect to this year
    )

    depreciation = Depreciation(
        year=[0, 25],
        rate=[0, 1],
    )

    tax_rate = 0.22
    DEVEX = 0
    shared_capex = CAPEX_shared

    res, grad = jax.value_and_grad(
        lambda *args: finances(*args)["IRR"], argnums=0, allow_int=True
    )(
        technologies,
        product_prices,
        shared_capex,
        inflation,
        tax_rate,
        depreciation,
        DEVEX,
    )
    print(res)
    print(grad)

    res = finances(
        technologies,
        product_prices,
        shared_capex,
        inflation,
        tax_rate,
        depreciation,
        DEVEX,
    )
    print(res)

    print(
        timeit.timeit(
            lambda: finances(
                technologies,
                product_prices,
                shared_capex,
                inflation,
                tax_rate,
                depreciation,
                DEVEX,
            ),
            number=1,
        ),
        "s",
    )

    ref = {
        "cashflow": np.array(
            [
                -3.71423011e08,
                2.24513680e07,
                2.33846642e07,
                2.43226626e07,
                2.53007504e07,
                2.63201665e07,
                2.73823554e07,
                2.84887682e07,
                2.96408093e07,
                3.08397247e07,
                3.20871903e07,
                3.33846531e07,
                3.47335186e07,
                3.61353925e07,
                3.75917414e07,
                3.91034056e07,
                4.06717494e07,
                4.22984174e07,
                4.39854907e07,
                4.57343056e07,
                4.75457839e07,
                4.94213055e07,
                5.13608503e07,
                5.33661798e07,
                5.54385615e07,
                5.76183758e07,
            ]
        ),
        "NPV": 55399626.7,
        "IRR": 7.26525611 / 100,
        "CAPEX": 371423011.2610241,
        "OPEX": np.array(
            [
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
            ]
        ),
        "break_even_prices": {"spot_electricity": 29.123712413268382},
        "LCOE": 43.39325844791887,
    }
    for ref_key, ref_value in ref.items():
        if isinstance(ref_value, dict):
            continue  # skip break_even_prices
            rsv, rfv = rsv["spot_electricity"], rfv["spot_electricity"]
        numpy.testing.assert_allclose(ref_value, res[ref_key])
        print("Passed")

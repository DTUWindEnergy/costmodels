import warnings

import numpy as np
import numpy_financial as npf

from costmodels.units import Quant


def calc_cashflows(
    capexs: list[float],
    opexs: list[float],
    lifetimes: list[int],
    t0s: list[int],
    prod: np.ndarray | float,  # representative year energy production prod(time)
    eprice: np.ndarray | float,  # representative year energy price epice(time)
):
    global_t0 = min(t0s)
    global_t1 = max([_lt + _t0 for _lt, _t0 in zip(lifetimes, t0s)])
    cashflows = Quant(np.zeros(global_t1 - global_t0), "EUR")
    for capex, opex, lifetime, t0 in zip(capexs, opexs, lifetimes, t0s):
        annual_revenue = np.sum(prod * eprice)
        annual_cashflow = annual_revenue - opex
        p_cashflow = Quant(np.zeros(lifetime), "EUR")
        p_cashflow[0] = -capex.to("EUR")
        p_cashflow += annual_cashflow.to("EUR")
        cashflows[t0 - global_t0 : lifetime + t0 - global_t0] += p_cashflow
    return cashflows


_NAN_RETURN_WARN = (
    "Cashflows contain NaN values. Returning NaN for $var. "
    "The input data is likely missing values like AEP or OPEX."
)


def calc_lcoe(cashflows: Quant, aep_net: Quant) -> Quant:
    if np.isnan(cashflows.m).any():
        warnings.warn(_NAN_RETURN_WARN.replace("$var", "LCOE"))
        return Quant(np.nan, "%")
    return (np.sum(cashflows) / aep_net).to("EUR/MWh")


def calc_irr(cashflows: Quant) -> Quant:
    if np.isnan(cashflows.m).any():
        warnings.warn(_NAN_RETURN_WARN.replace("$var", "IRR"))
        return Quant(np.nan, "%")
    return Quant(npf.irr(cashflows.m) * 100, "%")


def calc_npv(discount: Quant, cashflows: Quant):
    if np.isnan(cashflows.m).any():
        warnings.warn(_NAN_RETURN_WARN.replace("$var", "NPV"))
        return Quant(np.nan, "MEUR")
    return Quant(npf.npv(discount.to_base_units().m, cashflows.m), cashflows.units)

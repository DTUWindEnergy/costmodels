from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from costmodels.base import CostModel
from costmodels.units import Quant


def _runx(idict):
    capex = (
        (idict["panel_cost"] + idict["hardware_installation_cost"])
        * idict["dc_ac_ratio"]
        + idict["inverter_cost"]
    ) * idict["solar_capacity"]
    opex = idict["fixed_onm_cost"] * idict["solar_capacity"] * idict["dc_ac_ratio"]

    return {
        "capex": capex,
        "opex": opex,
    }


runjacobian: Callable = jax.jit(jax.jacrev(_runx))


def _input_dict_to_magnitudes(idict):
    """Convert all Quant values in the input dictionary to their magnitudes."""
    return {
        key: (
            jnp.array(value.magnitude, dtype=jnp.float32)
            if isinstance(value, Quant)
            else value
        )
        for key, value in idict.items()
    }


class PVCostModel(CostModel):

    @property
    def _cm_input_def(self):
        return {
            "solar_capacity": Quant(np.nan, "MW"),
            "dc_ac_ratio": 1.5,
            "panel_cost": Quant(1.1e5, "EUR/MW"),
            "hardware_installation_cost": Quant(1e5, "EUR/MW"),
            "inverter_cost": Quant(2e4, "EUR/MW"),
            "fixed_onm_cost": Quant(4.5e3, "EUR/MW"),
        }

    def __validate_input(self):
        for key, value in self._cm_input.items():
            if not hasattr(value, "m"):
                continue
            if np.isnan(value.m).any():
                raise ValueError(f"Value of {key} is not defined")

    def _run(self):
        self.__validate_input()

        capex = (
            (self.panel_cost + self.hardware_installation_cost) * self.dc_ac_ratio
            + self.inverter_cost
        ) * self.solar_capacity
        opex = self.fixed_onm_cost * self.solar_capacity * self.dc_ac_ratio

        return {
            "capex": capex,
            "opex": opex,
        }

    def _format_output(self, output):
        """Format the output of the cost model."""
        return {
            "capex": Quant(output["capex"], "EUR"),
            "opex": Quant(output["opex"], "EUR"),
        }

    def run(self, **kwargs):
        self._set_input(**kwargs)
        raw_input = _input_dict_to_magnitudes(self._cm_input)
        return self._format_output(_runx(raw_input)), runjacobian(raw_input)


def pprint_dict(d):
    """Pretty print a dictionary."""
    for key, value in d.items():
        if isinstance(value, Quant):
            print(f"{key}: {value.magnitude} {value.units}")
        else:
            print(f"{key}: {value}")


from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["lifetimes", "t0s"])
def calc_cashflows(
    capexs: jnp.ndarray,
    opexs: jnp.ndarray,
    lifetimes: jnp.ndarray,
    t0s: jnp.ndarray,
    prod: jnp.ndarray,
    eprice: jnp.ndarray,
):
    """
    Calculates cashflows over time for multiple components using JAX.

    Args:
        capexs: Array of capital expenditures for each component.
        opexs: Array of operational expenditures for each component.
        lifetimes: Array of lifetimes (in years) for each component.
        t0s: Array of start years for each component.
        prod: Representative year energy production (e.g., hourly for a year).
        eprice: Representative year energy price (e.g., hourly for a year).

    Returns:
        A JAX array representing the total cashflow for each year.
    """
    # capexs = jnp.asarray(capexs)
    # opexs = jnp.asarray(opexs)
    # lifetimes = jnp.asarray(lifetimes)
    # t0s = jnp.asarray(t0s)
    # prod = jnp.asarray(prod)
    # eprice = jnp.asarray(eprice)

    # # Handle potential scalar inputs by treating them as 1-element arrays
    # if capexs.ndim == 0:
    #     capexs = capexs.reshape(1)
    # if opexs.ndim == 0:
    #     opexs = opexs.reshape(1)
    # if lifetimes.ndim == 0:
    #     lifetimes = lifetimes.reshape(1)
    # if t0s.ndim == 0:
    #     t0s = t0s.reshape(1)

    num_components = capexs.shape[0]
    if not (
        opexs.shape[0] == num_components
        and len(lifetimes) == num_components
        and len(t0s) == num_components
    ):
        raise ValueError(
            "Input arrays (capexs, opexs, lifetimes, t0s) must have the same length."
        )

    # Determine the global time range
    global_t0 = min(t0s)
    global_t1 = max(t0s + lifetimes)
    total_years = global_t1 - global_t0

    # Initialize the cashflows array
    cashflows = jnp.zeros(total_years)  # Match dtype
    global_t0 = min(t0s)
    num_components = capexs.shape[0]
    annual_revenue = jnp.sum(prod * eprice)

    for i in range(num_components):
        capex = capexs[i]
        opex = opexs[i]
        lifetime = lifetimes[i]
        t0 = t0s[i]

        annual_cashflow = annual_revenue - opex

        # Create the cashflow vector for this specific component
        # p_cashflow[0] = -capex + annual_cashflow
        # p_cashflow[1:] = annual_cashflow
        p_cashflow_component = jnp.full(lifetime, annual_cashflow)
        # Use .add() for the capex adjustment relative to the filled value
        p_cashflow_component = p_cashflow_component.at[0].add(-capex)

        # Calculate the indices for this component in the global cashflow array
        start_idx = t0 - global_t0
        indices = jnp.arange(start_idx, start_idx + lifetime)

        # Update the overall cashflows using indexed addition (immutable update)
        # Ensure indices are valid; JAX handles out-of-bounds indices in .at[] by default (no-op)
        # but explicit clipping might be desired depending on logic.
        cashflows = cashflows.at[indices].add(p_cashflow_component)

    return cashflows


@partial(jax.jit)
def calculate_irr_jax(cashflows):
    res = jnp.roots(cashflows[::-1], strip_zeros=False)
    mask = (res.imag == 0) & (res.real > 0)

    def true_fn(args):
        res, mask = args
        rates = 1.0 / res.real - 1.0
        valid_rates = jnp.where(mask, jnp.abs(rates), jnp.inf)
        best_idx = jnp.argmin(valid_rates)
        return rates[best_idx]

    def false_fn(_):
        return jnp.nan

    return jax.lax.cond(jnp.any(mask), true_fn, false_fn, (res, mask))


if __name__ == "__main__":
    solar_capacity = Quant(100, "MW")
    pv_cm = PVCostModel(solar_capacity=solar_capacity)
    print("=" * 5 + " Input " + "=" * 5)
    pprint_dict(pv_cm._cm_input)
    cmo, _ = pv_cm.run()
    print("=" * 5 + " Output " + "=" * 5)
    print(cmo)

    # Prepare inputs as JAX arrays
    capex_val = jnp.array([cmo["capex"].m], dtype=jnp.float32)
    opex_val = jnp.array([cmo["opex"].m], dtype=jnp.float32)
    lifetime_val = (25,)
    t0_val = (0,)
    prod_val = jnp.array(np.ones(8760)) * 1e3  # Example production
    eprice_val = jnp.array(np.ones(8760))  # Example price

    import time

    dcash_dprod_jac = jax.jit(
        jax.jacrev(calc_cashflows, argnums=[-2]), static_argnums=[2, 3]
    )
    dirr_dcash_jac = jax.jit(jax.jacrev(calculate_irr_jax))

    s = time.time()
    cashflows = calc_cashflows(
        capexs=capex_val,
        opexs=opex_val,
        lifetimes=lifetime_val,
        t0s=t0_val,
        prod=prod_val,
        eprice=eprice_val,
    )

    dcash_dprod = dcash_dprod_jac(
        capex_val,
        opex_val,
        lifetime_val,
        t0_val,
        prod_val,
        eprice_val,
    )
    dcash_dprod = dcash_dprod[0]
    irr = calculate_irr_jax(cashflows)
    dirr_dcash = dirr_dcash_jac(cashflows)
    dirr_dprod = dirr_dcash @ dcash_dprod
    print(dirr_dprod.block_until_ready())
    print(f"Cashflow+Jac time: {time.time() - s:.2f} seconds")
    exit(0)

    # class method jacobian
    import jax
    import jax.numpy as jnp

    class MyClass:
        @staticmethod
        def afunc(params):
            return jnp.sin(params["a"] + params["b"] ** 2)

    MyClass.afunc_jacobian = jax.jacrev(MyClass.afunc)

    params = {"a": 1.0, "b": 2.0}
    jac = MyClass.afunc_jacobian(params)
    print(jac)

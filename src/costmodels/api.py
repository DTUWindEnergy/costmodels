from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# from functools import partial
from numbers import Number

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

# import chex
import pint

from costmodels.units import Quant


@tree_util.register_pytree_node_class
@dataclass
class CostModelOutput:
    capex: float
    opex: float

    def tree_flatten(self):
        # dynamic values propagated by jax AD
        children = (self.capex, self.opex)
        # no static data needed
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, _, children):
        # reconstruct the object from children
        return cls(*children)


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


class CostModel(ABC):
    """Base class for all the cost models."""

    @property
    @abstractmethod
    def _cm_input_def(self) -> dict:  # pragma: no cover
        """Definition of cost model input."""
        ...

    @staticmethod
    @abstractmethod
    def _run(x: dict):  # pragma: no cover
        """Internal function to run the cost model."""
        ...

    def __init__(self, **kwargs):
        # make sure if developer overrides the input
        # definition as a method it still works...
        self._cm_input = (
            self._cm_input_def
            if isinstance(self._cm_input_def, dict)
            else self._cm_input_def()
        )
        self._set_input(**kwargs)

    def _set_input(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self._set_input_key(key, value)

    def _set_input_key(self, key: str, value) -> None:
        assert (
            key in self._cm_input.keys()
        ), f"Invalid input '{key}' key not found in the input definition."

        if isinstance(self._cm_input[key], bool):
            assert isinstance(
                value, type(self._cm_input[key])
            ), f"Invalid type for '{key}', must be {type(self._cm_input[key])}"
            self._cm_input[key] = value
        elif isinstance(self._cm_input[key], Enum):
            if not isinstance(value, Enum):
                value = self._cm_input[key].__class__(value)
            self._cm_input[key] = value
        elif isinstance(self._cm_input[key], (Quant, Number, jnp.number)):
            is_quant_expected = isinstance(self._cm_input[key], Quant)
            if is_quant_expected:
                # the default value is a Quantity and
                # we try to assign units to provided number
                units = self._cm_input[key].units
                try:
                    quant = (
                        value.to(units)
                        if isinstance(value, Quant)
                        else Quant(value, units)
                    )
                except pint.errors.DimensionalityError:
                    raise ValueError(
                        f"Invalid unit for '{key}'; "
                        f"Expected [{units}] and got [{value.units}]."
                    )
            else:
                # keep a unitless number; input
                # specification does not expect a unit
                quant = value
            self._cm_input[key] = quant
        else:
            raise ValueError(
                f"Invalid type for '{key}'. Only numeric values, "
                f"pint.Quantity or Enum are allowed."
            )

    def run(self, **kwargs):
        self._set_input(**kwargs)
        cmo = self._run(_input_dict_to_magnitudes(self._cm_input))

        def _check_and_warn_for_nans_callback(capex_val, opex_val, inputs_val):
            import warnings  # fmt:skip

            import numpy as np  # fmt:skip

            if np.any(np.isnan(capex_val)) or np.any(np.isnan(opex_val)):
                warnings.warn(
                    f"NaNs detected in CostModelOutput (capex or opex). "
                    f"Effective inputs (magnitudes) to _run method: {str(inputs_val)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None  # Callbacks for side-effects should return None

        jax.debug.callback(
            _check_and_warn_for_nans_callback,
            cmo.capex,
            cmo.opex,
            _input_dict_to_magnitudes(self._cm_input),
        )

        return cmo


class ICostModel(CostModel):
    """Example implementation of cost model."""

    @property
    def _cm_input_def(self):
        return {
            "a": Quant(2, "m"),
            "b": 2,
            "dv": Quant(jnp.nan, "m"),
        }

    @staticmethod
    # @jax.jit
    def _run(x: dict):
        capex = jnp.abs(jnp.sin(x["dv"] ** 2 / x["b"] + x["a"] * jnp.cos(x["dv"])))
        opex = jnp.abs(jnp.cos(x["dv"] ** 2 / x["a"] + x["b"] * jnp.sin(x["dv"])))
        cmo = CostModelOutput(capex=capex, opex=opex)
        # chex.assert_tree_all_finite(cmo)
        # chex.block_until_chexify_assertions_complete()
        return cmo


# @partial(jax.jit, static_argnames=["lifetimes", "t0s"])
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


# @partial(jax.jit)
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
    import numpy as np  # fmt:skip

    cm = ICostModel(a=Quant(3, "m"))
    x0 = {"dv": 1.0}

    # @jax.jit
    def obj(x):
        out = cm.run(**x)
        cashflows = calc_cashflows(
            capexs=jnp.array([out.capex]),
            opexs=jnp.array([out.opex]),
            lifetimes=(25,),
            t0s=(0,),
            prod=jnp.array(np.ones(8760)) / 8760,
            eprice=jnp.array(np.ones(8760)),
        )
        irr = calculate_irr_jax(cashflows)
        return irr

    vg_func = jax.jit(jax.value_and_grad(obj))

    def vg_func_print(x):
        value, grad = vg_func(x)
        value.block_until_ready()
        grad["dv"].block_until_ready()
        return value, grad

    from timeit import timeit

    # warmup
    _ = vg_func_print(x0)

    print(
        timeit(
            lambda: vg_func_print(x0),
            number=100,
        )
    )

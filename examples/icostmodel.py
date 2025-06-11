"""Example of extending :class:`~costmodels.api.CostModel`.

This script implements ``ICostModel`` using JAX operations and demonstrates
basic differentiation of model run.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from costmodels.api import CostModel, CostModelOutput
from costmodels.units import Quant


class ICostModel(CostModel):
    """Minimal example cost model."""

    @property
    def _cm_input_def(self) -> dict:
        return {"a": Quant(2, "m"), "b": 2, "dv": Quant(jnp.nan, "m")}

    @staticmethod
    def _run(x: dict) -> CostModelOutput:
        capex = jnp.abs(jnp.sin(x["dv"] ** 2 / x["b"] + x["a"] * jnp.cos(x["dv"])))
        opex = jnp.abs(jnp.cos(x["dv"] ** 2 / x["a"] + x["b"] * jnp.sin(x["dv"])))
        return CostModelOutput(capex=capex, opex=opex)


if __name__ == "__main__":
    cm = ICostModel(a=Quant(3, "m"))
    x0 = {"dv": 1.0}

    def objective(x):
        out = cm.run(**x)
        return (out.opex + out.capex) ** 2

    value, grad = jax.jit(jax.value_and_grad(objective))(x0)
    print(f"Objective value: {value}")
    print(f"d(objective)/dv: {grad['dv']}")

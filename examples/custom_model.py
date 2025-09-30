"""Minimal example demonstrating the :mod:`costmodels` interface.

This script defines a trivial cost model using the dataclass based API and
computes the gradient of its output with JAX.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from costmodels.cmodel import CostInput, CostModel, CostOutput


class CustomCostInput(CostInput):
    design_variable: float
    a: float = 2.0
    b: float = 2.0


class CustomCostModel(CostModel):
    """Minimal example cost model."""

    _inputs_cls = CustomCostInput

    def _run(self, inputs: CustomCostInput) -> CostOutput:
        capex = jnp.abs(
            jnp.sin(
                inputs.design_variable**2 / inputs.b
                + inputs.a * jnp.cos(inputs.design_variable)
            )
        )
        opex = jnp.abs(
            jnp.cos(
                inputs.design_variable**2 / inputs.a
                + inputs.b * jnp.sin(inputs.design_variable)
            )
        )
        return CostOutput(capex=capex, opex=opex)


if __name__ == "__main__":
    cm = CustomCostModel(a=3.0)

    @jax.jit
    def objective(dv: float) -> jnp.ndarray:
        out = cm.run(design_variable=dv)
        return (out.opex + out.capex) ** 2

    value, grad = jax.value_and_grad(objective)(1.0)
    print(f"Objective value: {value}")
    print(f"d(objective)/dv: {grad}")

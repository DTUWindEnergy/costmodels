from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from costmodels._interface import CostModel, CostOutput, cost_input_dataclass


@cost_input_dataclass
class ExampleCostModelInputs:
    a: float = 2.1
    b: int = 2
    flag: bool = True
    dv: float = jnp.nan


class ExampleCostModel(CostModel):
    _inputs_cls = ExampleCostModelInputs

    def _run(self, inputs: ExampleCostModelInputs) -> Dict[str, Any]:
        capex = (
            jnp.abs(jnp.sin(inputs.dv**2 / inputs.b + inputs.a * jnp.cos(inputs.dv)))
            if inputs.flag
            else 0.0
        )
        opex = jnp.abs(jnp.cos(inputs.dv**2 / inputs.a + inputs.b * jnp.sin(inputs.dv)))
        return {"capex": capex, "opex": opex}


def test_example_cost_model():
    cm = ExampleCostModel(a=2.1, b=3.3, flag=True, dv=0.0)
    out = cm.run(dv=1.0)
    assert isinstance(out, CostOutput)
    assert out.capex >= 0
    assert out.opex >= 0

    val, grad = jax.value_and_grad(lambda x: cm.run(dv=x).capex)(1.0)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad)

    val, grad = jax.value_and_grad(lambda x: cm.run(dv=x, flag=False).capex)(1.0)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad)

    # JIT support feature is prosponed until need arises !!!
    # check that jit works on value_and_grad
    # jit_fn = jax.jit(lambda x: cm.run(dv=x).capex + cm.run(dv=x).opex)
    # jit_val, jit_grad = jax.value_and_grad(jit_fn)(3.0)
    # assert jnp.isfinite(jit_val)
    # assert jnp.isfinite(jit_grad)


def test_model_does_not_run_with_nan_values_in_inputs():
    cm = ExampleCostModel(a=2.1, b=3.3, flag=True, dv=jnp.nan)
    with pytest.raises(ValueError):
        cm.run()
    with pytest.raises(ValueError):
        cm.run(dv=jnp.nan)
    with pytest.raises(ValueError):
        cm.run(dv=np.nan)

    val_grad_func = jax.value_and_grad(lambda x: cm.run(dv=x).capex)
    val_grad_func(jnp.nan)  # jax traced values will not be checked for NaN
    val_grad_func = jax.value_and_grad(lambda x: cm.run(dv=x, a=np.nan).capex)
    with pytest.raises(ValueError):
        val_grad_func(1.0)  # should raise ValueError due to NaN in inputs in a

    cm.run(dv=1.0)


def test_array_input_with_shape_one():
    cm = ExampleCostModel(a=2.1, b=3.3, flag=True, dv=jnp.array([1.0]))
    out = cm.run()
    # should always be scalar values
    assert jnp.isscalar(out.capex) and jnp.isscalar(out.opex)

    # value and grad should work with array inputs
    val, grad = jax.value_and_grad(lambda x: cm.run(dv=x).capex)(jnp.array([1.0]))
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad)

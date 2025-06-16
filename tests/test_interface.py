from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp

from costmodels.interface import CostModel, CostOutput


@dataclass
class ExampleCostModelInputs:
    a: float = 2.1
    b: int = 2
    flag: bool = True
    dv: float = 0.0


class ExampleCostModel(CostModel):
    _inputs_cls = ExampleCostModelInputs

    def _run(self, inputs: ExampleCostModelInputs) -> Dict[str, Any]:
        if inputs.flag:
            capex = abs(
                jnp.sin(inputs.dv**2 / inputs.b + inputs.a * jnp.cos(inputs.dv))
            )
        else:
            capex = 0.0
        opex = abs(jnp.cos(inputs.dv**2 / inputs.a + inputs.b * jnp.sin(inputs.dv)))
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

    # check that jit works on value_and_grad
    jit_fn = jax.jit(lambda x: cm.run(dv=x).capex + cm.run(dv=x).opex)
    jit_val, jit_grad = jax.value_and_grad(jit_fn)(3.0)
    assert jnp.isfinite(jit_val)
    assert jnp.isfinite(jit_grad)

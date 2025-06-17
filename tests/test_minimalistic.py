import jax
import jax.numpy as jnp

from costmodels._interface import CostOutput
from costmodels.models import MinimalisticCostModel


def test_minimalistic_cost_model():
    mcm = MinimalisticCostModel()

    area = mcm.base_inputs.Area

    cmo = mcm.run(lifetime=20)
    assert isinstance(cmo, CostOutput)
    assert cmo.capex > 0

    area /= 2
    assert area < 65 * 10**6
    cm_output_small_area = mcm.run(Area=area)

    assert cm_output_small_area.capex < cmo.capex
    print(f"CAPEX: {cmo}")

    grad_depth = jax.grad(lambda x: mcm.run(depth=x).capex)(mcm.base_inputs.depth)
    grad_area = jax.grad(lambda x: mcm.run(Area=x).capex)(float(mcm.base_inputs.Area))
    assert jnp.isfinite(grad_depth)
    assert jnp.isfinite(grad_area)

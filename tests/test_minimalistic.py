import jax
import jax.numpy as jnp

from costmodels.cmodel import CostOutput
from costmodels.models import MinimalisticCostModel


def test_minimalistic_cost_model():
    mcm = MinimalisticCostModel()

    area = mcm._inputs_cls.Area

    cmo = mcm.run(lifetime=20)
    assert isinstance(cmo, CostOutput)
    assert cmo.capex > 0

    area /= 2
    assert area < 65 * 10**6
    cm_output_small_area = mcm.run(Area=area)

    assert cm_output_small_area.capex < cmo.capex

    grad_depth = jax.grad(lambda x: mcm.run(depth=x).capex)(mcm._inputs_cls.depth)
    grad_area = jax.grad(lambda x: mcm.run(Area=x).capex)(float(mcm._inputs_cls.Area))
    assert jnp.isfinite(grad_depth)
    assert jnp.isfinite(grad_area)

import jax.numpy as jnp

from costmodels._interface import CostModel, CostOutput, cost_input_dataclass
from costmodels.finance import Product, Technology
from costmodels.project import Project


@cost_input_dataclass
class DummyInputs:
    dv: float = jnp.nan


class DummyCM(CostModel):
    _inputs_cls = DummyInputs

    def _run(self, inputs: DummyInputs) -> CostOutput:
        return CostOutput(capex=jnp.abs(inputs.dv) * 1e6, opex=1.0)


cm = DummyCM()
costs = cm.run(dv=100.0)

tech = Technology(
    name="demo",
    lifetime=20,
    capex=costs.capex,
    opex=costs.opex,
    product=Product.SPOT_ELECTRICITY,
)

proj = Project(
    technologies=[tech],
    product_prices={Product.SPOT_ELECTRICITY: jnp.array([50.0])},
)

npv = proj.npv(productions={tech.name: jnp.array([1.0] * 20)})
grad_prod, grad_cm = proj.npv_grad(productions={tech.name: jnp.array([1.0] * 20)})
print(f"Net Present Value: {npv}")
print(f"dNPV/dproduction: {grad_prod[tech.name]}")
_ = grad_cm  # empty because didn't pass any cost model args

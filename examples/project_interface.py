import jax.numpy as jnp

from costmodels._interface import CostModel, CostOutput, cost_input_dataclass
from costmodels.finance import Depreciation, Inflation, Product, Technology
from costmodels.project import Project


@cost_input_dataclass
class DummyInputs:
    dv: float = jnp.nan


class DummyCM(CostModel):
    _inputs_cls = DummyInputs

    def _run(self, inputs: DummyInputs) -> CostOutput:
        return CostOutput(capex=jnp.abs(inputs.dv) * 1e6, opex=1.0)


TECH_NAME = "demo"
cm = DummyCM()
costs = cm.run(dv=100.0)

tech = Technology(
    name=TECH_NAME,
    CAPEX=costs.capex,
    OPEX=costs.opex,
    lifetime=20,
    t0=0,
    WACC=0.0,
    phasing_yr=[0],
    phasing_capex=[1],
    production=jnp.array([0.0] * 20),
    non_revenue_production=jnp.array([0.0] * 20),
    product=Product.SPOT_ELECTRICITY,
)

proj = Project(
    technologies=[tech],
    product_prices={Product.SPOT_ELECTRICITY: jnp.array([50.0])},
    inflation=Inflation(rate=[0.0, 0.0], year=[0, 20], year_ref=0),
    depreciation=Depreciation(year=[0, 20], rate=[0, 1]),
)

npv, grad = proj.npv_and_grad_production({TECH_NAME: jnp.array([1.0] * 20)})
print(f"Net Present Value: {npv}")
print(f"dNPV/dproduction: {grad}")

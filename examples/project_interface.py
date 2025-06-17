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
        return CostOutput(capex=jnp.abs(inputs.dv) * 1e6, opex=0.0)


cm = DummyCM()
tech = Technology(
    name="demo",
    CAPEX=cm.run(dv=1.0).capex,
    OPEX=0.0,
    lifetime=1,
    t0=0,
    WACC=0.05,
    phasing_yr=[0],
    phasing_capex=[1],
    production=jnp.array([100.0]),
    non_revenue_production=jnp.array([0.0]),
    product=Product.SPOT_ELECTRICITY,
)

proj = Project(
    technologies=[tech],
    product_prices={Product.SPOT_ELECTRICITY: jnp.array([50.0])},
    inflation=Inflation(rate=[0.0], year=[0], year_ref=0),
    depreciation=Depreciation(year=[0, 1], rate=[0, 1]),
)

# Compute NPV and its gradient with respect to production
npv, grad = proj.npv_and_grad_production({"demo": jnp.array([100.0])})
print(f"Net Present Value: {npv}")
print(f"dNPV/dproduction: {grad}")

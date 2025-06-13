import jax.numpy as jnp

from costmodels.api import CostModel, CostModelOutput
from costmodels.finance import Depreciation, Inflation, Product, Technology
from costmodels.project import Project
from costmodels.units import Quant


class DummyCM(CostModel):
    @property
    def _cm_input_def(self):
        return {"dv": Quant(jnp.nan, "m")}

    @staticmethod
    def _run(x):
        return CostModelOutput(capex=jnp.abs(x["dv"]) * 1e6, opex=0.0)


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

npv, grad = proj.npv_and_grad_production("demo")
print(f"Net Present Value: {npv}")
print(f"dNPV/dproduction: {grad}")

import jax.numpy as jnp
import numpy as np

from costmodels.cmodel import CostModel, CostOutput, cost_input_dataclass
from costmodels.finance import Product, Technology
from costmodels.project import Project

np.random.seed(0)


@cost_input_dataclass
class DummyInputs:
    design_variable0: float
    design_variable1: float = jnp.array([1.0])


class DummyCM(CostModel):
    _inputs_cls = DummyInputs

    def _run(self, inputs: DummyInputs) -> CostOutput:
        return CostOutput(
            capex=jnp.abs(inputs.design_variable0) * 1e6,
            opex=jnp.abs(jnp.sum(inputs.design_variable1)) * 1e2,
        )


LIFETIME = 5  # years
tech = Technology(
    name="demo",
    lifetime=LIFETIME,
    product=Product.SPOT_ELECTRICITY,
    cost_model=DummyCM(),
    opex=1.0,  # note here if opex is set, it won't be overridden by the cost model call
    capex=None,  # same with capex, if None, it will be set by the cost model
)

proj = Project(
    technologies=[tech],
    product_prices={Product.SPOT_ELECTRICITY: np.array([50.0])},
)

npv_val, __aux = proj.npv(
    productions={tech.name: np.array([np.random.randn()] * LIFETIME)},
    cost_model_args={
        tech.name: {
            "design_variable0": 10.0,
            "design_variable1": np.array([np.random.randn()] * 3),
        }
    },
    finance_args={"shared_capex": 1e5},
    return_aux=True,
)

npv_grad = proj.npv_grad(
    productions={tech.name: np.array([np.random.randn()] * LIFETIME)},
    cost_model_args={
        tech.name: {
            "design_variable0": 10.0,
            "design_variable1": np.array([np.random.randn()] * 3),
        }
    },
    finance_args={"shared_capex": 1e5},
)

print(f"NPV: {npv_val}")
print(f"dNPV/dProduction: {npv_grad[0][tech.name]}")
print(f"dNPV/dDesignVariable0: {npv_grad[1][tech.name]['design_variable0']}")
print(f"dNPV/dDesignVariable1: {npv_grad[1][tech.name]['design_variable1']}")

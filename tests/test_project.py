import jax.numpy as jnp
import numpy as np

from costmodels.finance import Depreciation, Inflation, Product, Technology
from costmodels.project import Project


def test_npv_and_grad_production_dict():
    tech1 = Technology(
        name="wind",
        CAPEX=10.0,
        OPEX=0.0,
        lifetime=1,
        t0=0,
        WACC=0.0,
        phasing_yr=[0],
        phasing_capex=[1],
        production=jnp.array([0.0]),
        non_revenue_production=jnp.array([0.0]),
        product=Product.SPOT_ELECTRICITY,
    )
    tech2 = Technology(
        name="solar",
        CAPEX=10.0,
        OPEX=0.0,
        lifetime=1,
        t0=0,
        WACC=0.0,
        phasing_yr=[0],
        phasing_capex=[1],
        production=jnp.array([0.0]),
        non_revenue_production=jnp.array([0.0]),
        product=Product.SPOT_ELECTRICITY,
    )
    proj = Project(
        technologies=[tech1, tech2],
        product_prices={Product.SPOT_ELECTRICITY: jnp.array([50.0])},
        inflation=Inflation(rate=[0.0, 0.0], year=[0, 1], year_ref=0),
        depreciation=Depreciation(year=[0, 1], rate=[0, 1]),
    )

    productions = {"wind": jnp.array([1.0]), "solar": jnp.array([2.0])}
    npv, grad = proj.npv_and_grad_production(productions)

    assert np.isclose(npv, 50.0 * (1.0 + 2.0) - 20.0)
    assert np.allclose(grad["wind"], jnp.array([50.0]))
    assert np.allclose(grad["solar"], jnp.array([50.0]))

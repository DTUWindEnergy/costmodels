from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest

from costmodels.finance import Depreciation, Inflation, Product, Technology
from costmodels.project import Project


@pytest.fixture
def project():
    tech1 = Technology(
        name="wind",
        capex=10.0,
        opex=1.0,
        lifetime=1,
        t0=0,
        wacc=0.0,
        phasing_yr=[0],
        phasing_capex=[1],
        production=[1.0],
        non_revenue_production=[0.0],
        product=Product.SPOT_ELECTRICITY,
    )
    tech2 = Technology(
        name="solar",
        capex=20.0,
        opex=2.0,
        lifetime=1,
        t0=0,
        wacc=0.0,
        phasing_yr=[0],
        phasing_capex=[1],
        production=[2.0],
        non_revenue_production=[0.0],
        product=Product.SPOT_ELECTRICITY,
    )
    proj = Project(
        technologies=[tech1, tech2],
        product_prices={Product.SPOT_ELECTRICITY: [50.0]},
        inflation=Inflation(rate=[0.0, 0.0], year=[0, 1], year_ref=0),
        depreciation=Depreciation(year=[0, 1], rate=[0, 1]),
    )
    return proj


def test_npv(project):
    cost_model_args = {}
    productions = {"wind": jnp.array([1.0]), "solar": jnp.array([2.0])}
    npv, __aux = project.npv(cost_model_args, productions, return_aux=True)

    expected_npv = 50.0 * (1.0 + 2.0) - (10.0 + 20.0) - (1.0 + 2.0)
    assert np.isclose(npv, expected_npv)


def test_project_inputs_as_ints_or_lists(project):
    productions = {"wind": 1, "solar": 2}
    npv, __aux = project.npv(productions=productions, return_aux=True)
    expected_npv = 50.0 * (1.0 + 2.0) - (10.0 + 20.0) - (1.0 + 2.0)
    assert np.isclose(npv, expected_npv)

    productions = {"wind": [1], "solar": [2]}
    npv, __aux = project.npv(productions=productions, return_aux=True)
    expected_npv = 50.0 * (1.0 + 2.0) - (10.0 + 20.0) - (1.0 + 2.0)
    assert np.isclose(npv, expected_npv)


@pytest.fixture
def mock_project_and_cm():
    # Mock cost model
    mock_cost_model = Mock()

    def mock_run(**kwargs):
        param1 = kwargs.get("param1", 0)
        return Mock(
            capex=(10.0 + param1 * 0.05) / 1e6, opex=(2.0 + param1 * 0.01) / 1e6
        )

    mock_cost_model.run.side_effect = mock_run

    tech1 = Technology(
        name="wind",
        lifetime=1,
        t0=0,
        wacc=0.0,
        phasing_yr=[0],
        phasing_capex=[1],
        production=jnp.array([1.0]),
        non_revenue_production=jnp.array([0.0]),
        product=Product.SPOT_ELECTRICITY,
        cost_model=mock_cost_model,
    )
    tech2 = Technology(
        name="solar",
        capex=20.0,
        opex=2.0,
        lifetime=1,
        t0=0,
        wacc=0.0,
        phasing_yr=[0],
        phasing_capex=[1],
        production=jnp.array([2.0]),
        non_revenue_production=jnp.array([0.0]),
        product=Product.SPOT_ELECTRICITY,
    )
    proj = Project(
        technologies=[tech1, tech2],
        product_prices={Product.SPOT_ELECTRICITY: jnp.array([50.0])},
        inflation=Inflation(rate=[0.0, 0.0], year=[0, 1], year_ref=0),
        depreciation=Depreciation(year=[0, 1], rate=[0, 1]),
    )
    return proj, mock_cost_model


def test_npv_with_cost_model(mock_project_and_cm):
    proj, mock_cost_model = mock_project_and_cm
    cost_model_args = {"wind": {"param1": 100.0}}
    productions = {"wind": jnp.array([1.0]), "solar": jnp.array([2.0])}
    npv, __aux = proj.npv(productions, cost_model_args, return_aux=True)

    # Expected NPV calculation
    expected_npv = 50.0 * (1.0 + 2.0) - (15.0 + 20.0) - (3.0 + 2.0)
    assert np.isclose(npv, expected_npv)

    # Verify the cost model was called
    mock_cost_model.run.assert_called_once()


def test_npv_grad_with_cost_model(mock_project_and_cm):
    proj, mock_cost_model = mock_project_and_cm

    cost_model_args = {"wind": {"param1": 100.0}}
    productions = {"wind": jnp.array([1.0]), "solar": jnp.array([2.0])}
    prod_grad, cm_grad = proj.npv_grad(productions, cost_model_args)

    # Verify gradients
    assert np.allclose(prod_grad["wind"], jnp.array([50.0]))
    assert np.allclose(prod_grad["solar"], jnp.array([50.0]))
    # due to mocking, the cost model gradient should be zero
    assert np.allclose(cm_grad["wind"]["param1"], jnp.array([-0.06]))
    # Verify the cost model was called with the correct arguments
    mock_cost_model.run.assert_called_once()

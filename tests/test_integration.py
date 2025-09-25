import numpy as np
from py_wake import BastankhahGaussian
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.examples.data.iea37 import IEA37_WindTurbines

from costmodels.finance import Product, Technology
from costmodels.models import DTUOffshoreCostModel
from costmodels.project import Project


def test_integration_of_project_with_dtu_offshore_cost_model():
    n_wt = 30
    x_min = 0
    x_max = 6000
    y_min = -10000
    y_max = 0
    LIFETIME = 25  # years
    el_price = 50  # fixed ppa price Euro per MWh

    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    initial = np.asarray(
        [
            rng1.random(n_wt) * (x_max - x_min) + x_min,
            rng2.random(n_wt) * (y_max - y_min) + y_min,
        ]
    ).T
    x_init = initial[:, 0]
    y_init = initial[:, 1]

    windTurbines = IEA37_WindTurbines()
    site = Hornsrev1Site()
    wfm = BastankhahGaussian(site, windTurbines)

    simres = wfm(x_init, y_init)
    aep_ref = simres.aep().values.sum()
    RP_MW = windTurbines.power(20) * 1e-6
    CF_ref = aep_ref * 1e3 / (RP_MW * 24 * 365 * n_wt)

    # test with capacity factor not available!
    cost_model = DTUOffshoreCostModel(
        rated_power=windTurbines.power(20) / 1e6,
        rotor_speed=10.0,
        rotor_diameter=windTurbines.diameter(),
        hub_height=windTurbines.hub_height(),
        lifetime=LIFETIME,
        # capacity_factor=CF_ref,
        nwt=n_wt,
        profit=0,
    )

    wind_plant = Technology(
        name="wind",
        lifetime=LIFETIME,
        product=Product.SPOT_ELECTRICITY,
        opex=12600 * n_wt * RP_MW + 1.35 * aep_ref * 1000,  # Euro
        wacc=0.06,
        cost_model=cost_model,
    )

    project = Project(
        technologies=[wind_plant],
        product_prices={Product.SPOT_ELECTRICITY: el_price},
    )

    def economic_func(aep, water_depth, cabling_cost, **kwargs):
        aep_over_lifetime = aep * np.ones(LIFETIME) * 10**3
        npv, aux = project.npv(
            productions={wind_plant.name: aep_over_lifetime},
            cost_model_args={
                wind_plant.name: {"water_depth": water_depth, "aep": aep_over_lifetime}
            },
            finance_args={"shared_capex": cabling_cost},
            return_aux=True,
        )
        return npv, {
            "LCOE": aux["LCOE"][0],
            "IRR": aux["IRR"],
            "CAPEX": aux["CAPEX"],
            "OPEX": np.mean(aux["OPEX"]),
        }

    economic_func(1e3, 10.0, 10.0)

    # test with capacity factor and no AEP passed to the cost model
    cost_model = DTUOffshoreCostModel(
        rated_power=windTurbines.power(20) / 1e6,
        rotor_speed=10.0,
        rotor_diameter=windTurbines.diameter(),
        hub_height=windTurbines.hub_height(),
        lifetime=LIFETIME,
        capacity_factor=CF_ref,
        nwt=n_wt,
        profit=0,
    )

    wind_plant = Technology(
        name="wind",
        lifetime=LIFETIME,
        product=Product.SPOT_ELECTRICITY,
        opex=12600 * n_wt * RP_MW + 1.35 * aep_ref * 1000,  # Euro
        wacc=0.06,
        cost_model=cost_model,
    )

    project = Project(
        technologies=[wind_plant],
        product_prices={Product.SPOT_ELECTRICITY: el_price},
    )

    def economic_func(aep, water_depth, cabling_cost, **kwargs):
        aep_over_lifetime = aep * np.ones(LIFETIME) * 10**3
        npv, aux = project.npv(
            productions={wind_plant.name: aep_over_lifetime},
            cost_model_args={wind_plant.name: {"water_depth": water_depth}},
            finance_args={"shared_capex": cabling_cost},
            return_aux=True,
        )
        return npv, {
            "LCOE": aux["LCOE"][0],
            "IRR": aux["IRR"],
            "CAPEX": aux["CAPEX"],
            "OPEX": np.mean(aux["OPEX"]),
        }

    economic_func(1e3, 10.0, 10.0)

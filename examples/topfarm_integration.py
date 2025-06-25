import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from py_wake import BastankhahGaussian
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from scipy.interpolate import RegularGridInterpolator
from topfarm._topfarm import TopFarmGroup, TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp
from topfarm.utils import plot_list_recorder

from costmodels.finance import Depreciation, Inflation, Product, Technology, finances
from costmodels.models import DTUOffshoreCostModel

warnings.filterwarnings("ignore", category=UserWarning)

### Site

n_wt = 30
initial = np.asarray([np.random.random(30) * 6000, np.random.random(30) * -10000]).T
x_init = initial[:, 0]
y_init = initial[:, 1]
boundary = np.array(
    [(0, 0), (6000, 0), (6000, -10000), (0, -10000)]
)  # turbine boundaries
windTurbines = IEA37_WindTurbines()
site = Hornsrev1Site()
wfm = BastankhahGaussian(site, windTurbines)

### Bathymetry

sigma = 3000.0
mu = 0.0

x_peak_1 = 1000
y_peak_1 = -1000
x_peak_2 = 4000
y_peak_2 = -8000
x1, y1 = np.meshgrid(
    np.linspace(0 - x_peak_1, 6000 - x_peak_1, 100),
    np.linspace(-10000 - y_peak_1, 0 - y_peak_1, 100),
)
d1 = np.sqrt(x1 * x1 + y1 * y1)
g1 = np.exp(-((d1 - mu) ** 2 / (2.0 * sigma**2)))
x2, y2 = np.meshgrid(
    np.linspace(0 - x_peak_2, 6000 - x_peak_2, 100),
    np.linspace(-10000 - y_peak_2, 0 - y_peak_2, 100),
)
d2 = np.sqrt(x2 * x2 + y2 * y2)
g2 = np.exp(-((d2 - mu) ** 2 / (2.0 * sigma**2)))
g = 5 * g1 - 8 * g2 - 30

if 1:
    plt.imshow(g, extent=(-1000, 7000, -11000, 1000), origin="lower", cmap="viridis")
    plt.colorbar()
    plt.title("2D Gaussian Function")
    plt.show()

x = np.linspace(-1000, 7000, 100)
y = np.linspace(-11000, 1000, 100)
f = RegularGridInterpolator((x, y), g)

### Economy

LIFETIME = 25
cost_model = DTUOffshoreCostModel(
    # water_depth=30.0, # DYNAMIC
    # aep=1.0e9, # DYNAMIC
    rated_power=windTurbines.power(20) / 1e6,
    rotor_speed=10.0,
    rotor_diameter=windTurbines.diameter(),
    hub_height=windTurbines.hub_height(),
    profit=0.01,
    capacity_factor=0.4,
    decline_factor=0.01,
    nwt=n_wt,
    wacc=0.07,
    devex=0.0,
    abex=0.0,
    electrical_cost=0.0,
    lifetime=LIFETIME,
    inflation=0.02,
    opex=0.02,
    eprice=0.1,
    foundation_option=1,
)

out = cost_model.run(aep=1.0e9, water_depth=20.0)
print(out)

wind_technology = Technology(
    name="wind",
    CAPEX=out.capex,
    # TODO: technology assumes annual opex and model gives out over lifetime
    OPEX=out.opex / LIFETIME,
    lifetime=LIFETIME,
    t0=0,
    WACC=0.0,
    phasing_yr=[0],
    phasing_capex=[1],
    production=np.array([1.0] * LIFETIME),
    non_revenue_production=np.array([0.0] * LIFETIME),
    product=Product.SPOT_ELECTRICITY,
)

product_prices = {Product.SPOT_ELECTRICITY: np.array([0.1])}  # EUR/kWh
inflation = Inflation(rate=[0.0, 0.0], year=[0, LIFETIME], year_ref=0)
depreciation = Depreciation(year=[0, LIFETIME], rate=[0, 1])


# Economy
def npv_func(AEP, water_depth, **kwargs):
    cost_model_output = cost_model.run(aep=AEP, water_depth=water_depth)
    wind_technology.CAPEX = cost_model_output.capex
    wind_technology.OPEX = cost_model_output.opex / LIFETIME
    wind_technology.production = jnp.array([AEP] * LIFETIME).squeeze()

    wind_finances = finances(
        technologies=[wind_technology],
        product_prices=product_prices,
        shared_capex=0.0,
        inflation=inflation,
        tax_rate=0.0,
        depreciation=depreciation,
        devex=0.0,
    )

    return wind_finances["NPV"], {
        "irr": wind_finances["IRR"],
        "OPEX": wind_finances["OPEX"],
        "CAPEX": wind_finances["CAPEX"],
    }


def topfarm_npv_func(AEP, water_depth, **kwargs):
    npv_value, _ = npv_func(AEP, water_depth)
    return np.asarray(npv_value)


npv_grad_func = jax.grad(npv_func, argnums=(0, 1), has_aux=True)

# print(npv_func(1e9, [20.0] * n_wt))  # Test the npv_func
# print(npv_grad_func(1e9, jnp.array([20.0] * n_wt)))  # Test the gradient function


def topfarm_npv_grad_func(AEP, water_depth, **kwargs):
    gradients, _ = npv_grad_func(AEP, water_depth)
    aep_grad, water_depth_grad = gradients
    return np.asarray(aep_grad), np.asarray(water_depth_grad)


# print(
#     topfarm_npv_grad_func(1e9, jnp.array([20.0] * n_wt))
# )  # Test the TopFarm gradient function


# Water Depth
def water_depth_func(x, y, **kwargs):
    xnew, ynew = np.meshgrid(x, y)
    points = np.array([xnew.flatten(), ynew.flatten()]).T
    return -np.diag(f(points).reshape(n_wt, n_wt).T)


# Water Depth
water_depth_component = CostModelComponent(
    input_keys=[("x", x_init), ("y", y_init)],
    n_wt=n_wt,
    cost_function=water_depth_func,
    objective=False,
    output_keys=[("water_depth", np.zeros(n_wt))],
)

# Economy
npv_comp = CostModelComponent(
    input_keys=[
        ("AEP", 0),
        ("water_depth", 30 * np.ones(n_wt)),
    ],
    n_wt=n_wt,
    cost_function=topfarm_npv_func,
    cost_gradient_function=topfarm_npv_grad_func,
    objective=True,
    maximize=True,
    output_keys=[("npv", 0)],
)

# AEP
aep_comp = PyWakeAEPCostModelComponent(wfm, n_wt, objective=False)

cost_comp = TopFarmGroup(
    [
        PyWakeAEPCostModelComponent(wfm, n_wt, objective=False),
        water_depth_component,
        npv_comp,
    ]
)

tf = TopFarmProblem(
    design_vars=dict(zip("xy", initial.T)),
    cost_comp=cost_comp,
    constraints=[XYBoundaryConstraint(boundary), SpacingConstraint(500)],
    driver=EasyScipyOptimizeDriver(maxiter=100),
    plot_comp=XYPlotComp(),
)


### Optimize

cost, _, recorder = tf.optimize()

### Plot

plot_list_recorder(recorder)

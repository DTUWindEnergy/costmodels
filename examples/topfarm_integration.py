import warnings

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

from costmodels.finance import Product, Technology
from costmodels.models import DTUOffshoreCostModel
from costmodels.project import Project

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
    lifetime=LIFETIME,
    capacity_factor=0.4,
    nwt=n_wt,
)

out = cost_model.run(aep=1.0e9, water_depth=20.0)
print(out)

wind_technology = Technology(
    name="wind",
    lifetime=LIFETIME,
    product=Product.SPOT_ELECTRICITY,
    cost_model=cost_model,
)

wind_farm_project = Project(
    technologies=[wind_technology],
    product_prices={Product.SPOT_ELECTRICITY: np.random.uniform(0, 6, LIFETIME)},
)


# Economy
def npv_func(AEP, water_depth, **kwargs):
    return np.asarray(
        wind_farm_project.npv(
            productions={
                wind_technology.name: AEP,
            },
            cost_model_args={
                wind_technology.name: {
                    "water_depth": water_depth,
                    "aep": AEP,
                }
            },
        )
    )


def npv_grad_func(AEP, water_depth, **kwargs):
    grads = wind_farm_project.npv_grad(
        productions={
            wind_technology.name: AEP,
        },
        cost_model_args={
            wind_technology.name: {
                "water_depth": water_depth,
                "aep": AEP,
            }
        },
    )
    prod_grad = grads[0][wind_technology.name]
    water_depth_grad = grads[1][wind_technology.name]["water_depth"]
    return np.asarray(prod_grad), np.asarray(water_depth_grad)


# Water Depth
def water_depth_func(x, y, **kwargs):
    xnew, ynew = np.meshgrid(x, y)
    points = np.array([xnew.flatten(), ynew.flatten()]).T
    return 10 * np.diag(f(points).reshape(n_wt, n_wt).T)


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
    cost_function=npv_func,
    cost_gradient_function=npv_grad_func,
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
    driver=EasyScipyOptimizeDriver(maxiter=10),  # too little; just for demo purposes
    plot_comp=XYPlotComp(),
)


### Optimize

cost, _, recorder = tf.optimize()

### Plot

plot_list_recorder(recorder)

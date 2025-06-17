# %% [markdown]

# TODO: update to new interface

# # Usage of Cost Models package
#
# This notebook will go through the currently implemented models to show how they work; This should act as a simple user guide. For more advanced examples like optimization please look at the other notebooks in this directory.
#
# All models have their input defined in the property called `_cm_input_def`, one can inspect the source code and see what inputs are available, their default values and units. Alternatively, you can instantiate a model and call `print_input` method to get an overview of the same information.
#
# Each model constructor can take in inputs in the argument of instantiation. For example, `MyModel(input1=10.0, input2=30.0)` and these are meant to be used as static inputs like energy asset specification.
#
# Moreover, models are meant to be executed with a common `run` method: `MyModel().run(aep=1000)`. In this case the input arguments are dynamic values, that are changing in between model calls. Like in optimization, if one is designing a wind farm layout, the AEP would be different at each iteration depending on the layout.

# %% [markdown]
# ### Units
#
# Units are managed with `pint` package; Documentation can be found https://pint.readthedocs.io/en/stable/.

# %%
import pint

length = pint.Quantity(1, "km")
width = pint.Quantity(100, "m")
length, width

# %%
area = (length * width).to_reduced_units()
area

# %% [markdown]
# Trying to operate on non-matching units will result in dimensionality error. Try uncommenting the cell below and run it:

# %%
# length + pint.Quantity(1, "kg")

# %% [markdown]
# One can obtain a magnitude of a quantity by accessing an attribute called `m` or `magnitude`; Change the units calling method `to("unit string")` and access the units object with attribute `units`.

# %%
area.m, area.magnitude, area.to("km^2").magnitude, area.units

# %% [markdown]
# ### PV Cost Model

import numpy as np

# %%
from costmodels import PVCostModel
from costmodels.units import Quant

pv_cm = PVCostModel()
pv_cm.print_input()

# %%
pv_cm = PVCostModel(  # static inputs
    panel_cost=Quant(1e5, "EUR/MW"),
    hardware_installation_cost=Quant(1e5, "EUR/MW"),
    inverter_cost=Quant(1e5, "EUR/MW"),
    fixed_onm_cost=Quant(1e5, "EUR/MW"),
    dc_ac_ratio=1.2,
)

# dynamic inputs
output_capacity_10mw = pv_cm.run(solar_capacity=Quant(10, "MW"))
output_capacity_42mw = pv_cm.run(solar_capacity=Quant(42, "MW"))

print("10 MW PV system cost: ", output_capacity_10mw)
print("42 MW PV system cost: ", output_capacity_42mw)

# %% [markdown]
# ### DTU Offshore Cost Model

# %%
from costmodels import DTUOffshoreCostModel

# notice no units are specified;
# they are implicitly defined in the class;
# Be careful with calling models in this way
# make sure you are using the correct units!
dtu_offshore_cm = DTUOffshoreCostModel(
    **{
        "rated_power": 5.111111111111111,
        "rotor_diameter": 80,
        "rotor_speed": 9.444444444444445,
        "hub_height": 200.111486515663536,
        "profit": 1.0,
        "capacity_factor": 33,
        "decline_factor": 2.0,
        "nwt": 10,
        "lifetime": 30,
        "wacc": 7.2,
        "inflation": 8,
        "opex": 3.0,
        "devex": 11.11111111111111,
        "abex": 5.555555555555555,
        "water_depth": 33.33333333333333,
        "electrical_cost": 0.0,
        "foundation_option": 1,
        "eprice": 0.2,
        "inflation": 8,
    }
)
cm_output = dtu_offshore_cm.run()
for k in list(cm_output.keys()):
    v = cm_output[k]
    if not isinstance(v, np.ndarray) and not isinstance(v.m, np.ndarray):
        print(k, v)
    else:
        print(k, f"Mean value: {v.mean()}")

# %%
dtu_offshore_cm.print_input()

# %% [markdown]
# ### Minimalistic Cost Model

# %%
from costmodels import MinimalisticCostModel

mcm = MinimalisticCostModel()
cmo = mcm.run()
cmo

# %% [markdown]
# ### NREL Cost Model

# %%
from costmodels import NRELCostModel

nrel_cm = NRELCostModel(
    machine_rating=Quant(5.0, "MW"),
    rotor_diameter=126.0,
    tower_length=90.0,
    blade_number=3,
    blade_has_carbon=False,
    max_tip_speed=80.0,
    max_efficiency=90,
    main_bearing_number=2,
    crane=True,
    eprice=0.2,
    inflation=2,
    nwt=20,
    lifetime=20,
    opex=Quant(20.0, "EUR/kW"),
)
nrel_cm.run(
    aep=Quant(20.0, "GWh"),
)

# %%


# %%

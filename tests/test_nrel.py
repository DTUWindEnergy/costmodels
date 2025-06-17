# om = pytest.importorskip("openmdao.api", minversion="0")
import openmdao.api as om
import pytest

from costmodels.models import NRELCostModel, NRELTurbineClass
from costmodels.models.external.nrel_csm_mass_2015 import nrel_csm_2015


def test_nrel():
    prob = om.Problem(reports=False)
    prob.model = nrel_csm_2015()
    prob.setup()

    prob["machine_rating"] = 5000.0
    prob["rotor_diameter"] = 126.0
    prob["turbine_class"] = 2
    prob["tower_length"] = 90.0
    prob["blade_number"] = 3
    prob["blade_has_carbon"] = False
    prob["max_tip_speed"] = 80.0
    prob["max_efficiency"] = 0.90
    prob["main_bearing_number"] = 2
    prob["crane"] = True

    prob.run_model()

    NWT = 10
    nrel_cm = NRELCostModel(
        machine_rating=5000.0,
        rotor_diameter=126.0,
        turbine_class=NRELTurbineClass.CLASS_II,
        tower_length=90.0,
        blade_number=3,
        blade_has_carbon=False,
        max_tip_speed=80.0,
        max_efficiency=90,
        main_bearing_number=2,
        crane=True,
        nwt=NWT,
        opex=20.0,
    )

    res = nrel_cm.run(aep=10.0)
    assert (res.capex / NWT) == prob.model._outputs["turbine_cost"][0]

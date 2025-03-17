import openmdao.api as om

from costmodels.external.nrel_csm_mass_2015 import nrel_csm_2015
from costmodels.nrel import NRELCM, TurbineClass
from costmodels.units import Quant


def test_nrel():
    # OpenMDAO Problem instance
    prob = om.Problem(reports=False)
    prob.model = nrel_csm_2015()
    prob.setup()

    # Initialize variables for NREL CSM
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

    # Evaluate the model
    prob.run_model()

    NWT = 10
    nrel_cm = NRELCM(
        machine_rating=Quant(5.0, "MW"),
        rotor_diameter=126.0,
        turbine_class=TurbineClass.II,
        tower_length=90.0,
        blade_number=3,
        blade_has_carbon=False,
        max_tip_speed=80.0,
        max_efficiency=90,
        main_bearing_number=2,
        crane=True,
        eprice=0.2,
        inflation=2,
        nwt=NWT,
        lifetime=20,
        opex=Quant(20.0, "MEUR/kW"),
    )

    nrel_cmo: NRELCM.Output = nrel_cm.run()
    assert (nrel_cmo["capex"].to("EUR").m / NWT) == prob.model._outputs["turbine_cost"][
        0
    ]

    # grads = nrel_cm.grad(nrel_cm_input, "capex", ("rotor_diameter",))
    # assert "rotor_diameter" in grads.keys()

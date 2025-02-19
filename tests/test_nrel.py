import openmdao.api as om

from .assets.nrel_csm_mass_2015 import nrel_csm_2015


def test_nrel():
    assert False

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

    # Print all intermediate inputs and outputs to the screen
    prob.model.list_inputs(units=True)
    prob.model.list_outputs(units=True)

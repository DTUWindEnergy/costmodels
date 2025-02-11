from costmodels import (
    DTUOffshoreCM,
    DTUOffshoreCMInput,
    DTUOffshoreCMOutput,
)


def test_dtu_offshore():
    cm = DTUOffshoreCM()

    cm_input = DTUOffshoreCMInput(
        rated_power=3.111111111111111,
        rotor_diameter=0.060314403509210746,
        rotor_speed=9.444444444444445,
        hub_height=20.111486515663536,
        profit=0.01,
        capacity_factor=0.3333333333333333,
        decline_factor=-0.02,
        nwt=290,
        project_lifetime=27,
        wacc=0.07222222222222223,
        inflation=0.08,
        opex=30.0,
        devex=11.11111111111111,
        abex=5.555555555555555,
        water_depth=33.33333333333333,
        electrical_cost=0.0,
        foundation_option=0,
        eprice=(0.2, "EUR/kWh"),
    )
    cm_output = cm.run(cm_input)

    assert isinstance(cm_output, DTUOffshoreCMOutput)

    cm_input.AEP = 1.0
    cm_output_aep = cm.run(cm_input)

    assert cm_output_aep.aep_net != cm_output.aep_net

from costmodels import DTUOffshoreCM
from costmodels.units import Quant


def test_dtu_offshore():
    cm = DTUOffshoreCM()

    cm_input = cm.Input(
        rated_power=Quant(5.111111111111111, "MW"),
        rotor_diameter=0.060314403509210746,
        rotor_speed=9.444444444444445,
        hub_height=20.111486515663536,
        profit=Quant(1, "%"),
        capacity_factor=Quant(33.333, "%"),
        decline_factor=Quant(2, "%"),
        nwt=290,
        lifetime=27,
        wacc=0.07222222222222223,
        inflation=0.08,
        opex=0.0,
        devex=11.11111111111111,
        abex=5.555555555555555,
        water_depth=33.33333333333333,
        electrical_cost=0.0,
        foundation_option=0,
        eprice=Quant(0.2, "EUR/kWh"),
        aep=5 * 1e6 * 8760,
    )
    cm_output = cm.run(cm_input)
    assert isinstance(cm_output, cm.Output)

    cm_input.aep = 1.0
    cm_output_aep = cm.run(cm_input)

    assert cm_output_aep.aep_net != cm_output.aep_net
    print(f"{cm_output.irr=}")
    print(f"{cm_output.npv=}")

    grads = cm.grad(cm_input, "lcoe", ("water_depth",))
    assert "water_depth" in grads.keys()

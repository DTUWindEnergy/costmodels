from costmodels import DTUOffshoreCostModel
from costmodels.units import Quant


def test_dtu_offshore():
    cm = DTUOffshoreCostModel(
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
        inflation=Quant(8, "%"),
        opex=0.0,
        devex=11.11111111111111,
        abex=5.555555555555555,
        water_depth=33.33333333333333,
        electrical_cost=0.0,
        foundation_option=0,
        eprice=Quant(0.2, "EUR/kWh"),
        aep=[5 * 1e6 * 8760, 5 * 1e6 * 8760],
    )
    cm_output = cm.run()

    cm_output_aep = cm.run(aep=1.0)

    assert cm_output_aep["aep_net"] != cm_output["aep_net"]

    grads = cm.grad("npv", ("eprice",))
    assert "eprice" in grads.keys()

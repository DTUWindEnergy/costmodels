from costmodels import (
    MinimalisticCM,
    MinimalisticCMInput,
    MinimalisticCMOutput,
)


def test_minimalistic_cost_model():
    mcm = MinimalisticCM()

    cmi = MinimalisticCMInput(eprice=(0.2, "EUR/kWh"))

    cmo = mcm.run(cmi)

    assert isinstance(cmo, MinimalisticCMOutput)
    assert cmo.npv > 0

    cmi.Area /= 2
    assert cmi.Area < 65 * 10**6
    cm_output_small_area = mcm.run(cmi)

    assert cm_output_small_area.lcoe > cmo.lcoe

    print(f"CAPEX: {cmo}")

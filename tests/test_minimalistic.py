from costmodels import MinimalisticCM, MinimalisticCMInput, MinimalisticCMOutput
from costmodels.units import Quant


def test_minimalistic_cost_model():
    mcm = MinimalisticCM()

    cmi = MinimalisticCMInput(
        eprice=Quant(0.2, "EUR/kWh"),
        inflation=Quant(8, "%"),
    )

    cmo = mcm.run(cmi)

    assert isinstance(cmo, MinimalisticCMOutput)
    assert cmo.npv.magnitude > 0

    cmi.Area /= 2
    assert cmi.Area < 65 * 10**6
    cm_output_small_area = mcm.run(cmi)

    assert cm_output_small_area.lcoe > cmo.lcoe

    print(f"CAPEX: {cmo}")

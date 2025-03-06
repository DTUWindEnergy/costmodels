from costmodels import MinimalisticCM
from costmodels.units import Quant


def test_minimalistic_cost_model():
    mcm = MinimalisticCM()

    cmi = MinimalisticCM.Input(
        eprice=Quant(0.2, "EUR/kWh"),
        inflation=Quant(8, "%"),
        lifetime=20,
    )

    cmo = mcm.run(cmi)

    assert isinstance(cmo, MinimalisticCM.Output)
    assert cmo.npv.magnitude > 0

    cmi.Area /= 2
    assert cmi.Area < 65 * 10**6
    cm_output_small_area = mcm.run(cmi)

    assert cm_output_small_area.lcoe > cmo.lcoe

    print(f"CAPEX: {cmo}")

    grad = mcm.grad(cmi, "lcoe", ("depth", "Area"))
    assert "depth" in grad.keys()
    assert "Area" in grad.keys()

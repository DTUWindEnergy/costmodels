from costmodels.models import MinimalisticCostModel


def test_minimalistic_cost_model():
    mcm = MinimalisticCostModel()

    area = mcm.Area

    cmo = mcm.run(lifetime=20)
    assert cmo["capex"] > 0

    area /= 2
    assert area < 65 * 10**6
    cm_output_small_area = mcm.run(Area=area)

    assert cm_output_small_area["capex"] < cmo["capex"]
    print(f"CAPEX: {cmo}")

    grad = mcm.grad("capex", ("depth", "Area"))
    assert "depth" in grad.keys()
    assert "Area" in grad.keys()

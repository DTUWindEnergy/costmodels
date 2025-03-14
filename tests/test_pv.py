import pytest

from costmodels.pv import PVCostModel
from costmodels.units import Quant


def test_run_pv_model():
    solar_capacity = Quant(150, "MW")

    # good run
    pv_cm = PVCostModel()
    output = pv_cm.run(solar_capacity=solar_capacity)
    print(f"CAPEX = {output["capex"]:.2f}")
    print(f"OPEX = {output["opex"]:.3f}")

    # wrong units
    pv_cm = PVCostModel()
    with pytest.raises(ValueError):
        output = pv_cm.run(solar_capacity=solar_capacity, panel_cost=Quant(1.1e5, "m"))

    # missing required input
    pv_cm = PVCostModel()
    with pytest.raises(ValueError):
        output = pv_cm.run()

import pytest

from costmodels.models.pv import PVCostModel
from costmodels.units import Quant


def test_run_pv_model():
    solar_capacity = Quant(150, "MW")

    # good run
    pv_cm = PVCostModel()
    _ = pv_cm.run(solar_capacity=solar_capacity)

    # wrong units
    pv_cm = PVCostModel()
    with pytest.raises(ValueError):
        _ = pv_cm.run(solar_capacity=solar_capacity, panel_cost=Quant(1.1e5, "m"))

    # missing required input
    pv_cm = PVCostModel()
    with pytest.raises(ValueError):
        _ = pv_cm.run()

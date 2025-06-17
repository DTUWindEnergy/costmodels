import pytest

from costmodels.models.pv import PVCostModel


def test_run_pv_model():
    solar_capacity = 150

    # good run
    pv_cm = PVCostModel()
    res = pv_cm.run(solar_capacity=solar_capacity)
    assert res.capex > 0
    assert res.opex >= 0

    # missing required input
    pv_cm = PVCostModel()
    with pytest.raises(ValueError):
        _ = pv_cm.run()

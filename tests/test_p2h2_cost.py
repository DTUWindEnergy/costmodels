import numpy as np

from costmodels.p2h2_cost import PowerToHydrogenCostModel
from costmodels.units import Quant


def test_run_power_to_hydrogen_model():
    electrolyzer_capacity = Quant(800, "MW")
    hydrogen_storage_capacity = Quant(5000, "kg")
    mean_hydrogen_offtake = Quant(2343, "kg")
    PTHCM = PowerToHydrogenCostModel()
    res = PTHCM.run(
        electrolyzer_capacity=electrolyzer_capacity,
        hydrogen_storage_capacity=hydrogen_storage_capacity,
        mean_hydrogen_offtake=mean_hydrogen_offtake,
    )
    np.testing.assert_allclose(res["capex"].to_base_units().magnitude, 641500000)
    np.testing.assert_allclose(
        res["opex"].to_base_units().magnitude, 0.4427647207645701
    )

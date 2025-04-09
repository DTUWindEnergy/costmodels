import numpy as np

from costmodels.battery_cost import BatteryCostModel
from costmodels.units import Quant


def test_run_battery_model():
    battery_power = Quant(27, "MW")
    battery_energy = Quant(108, "MWh")
    state_of_health = np.hstack(
        [-1.7e-6 * np.arange(1.8e5) + 1, -2.5e-6 * np.arange(25 * 365 * 24 - 1.8e5) + 1]
    ).ravel()
    BCM = BatteryCostModel()
    res = BCM.run(
        battery_power=battery_power,
        battery_energy=battery_energy,
        state_of_health=state_of_health,
    )
    np.testing.assert_allclose(
        res["capex"].to_base_units().magnitude, 10162827.279138453
    )
    np.testing.assert_allclose(res["opex"].to_base_units().magnitude, 0)

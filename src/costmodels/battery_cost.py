import numpy as np

from costmodels.base import CostModel
from costmodels.units import Quant


class BatteryCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {
            "battery_power": Quant(np.nan, "MW"),
            "battery_energy": Quant(np.nan, "MWh"),
            "state_of_health": Quant(np.nan, ""),
            "battery_energy_cost": Quant(62000, "EUR/MWh"),
            "battery_power_cost": Quant(16000, "EUR/MW"),
            "battery_BOP_installation_commissioning_cost": Quant(80000, "EUR/MW"),
            "battery_control_system_cost": Quant(2250, "EUR/MW"),
            "battery_energy_onm_cost": Quant(0, "EUR/MWh"),
            "plant_lifetime": Quant(25, "year"),
            "dispatch_intervals_per_hour": Quant(1, "1/h"),
            "battery_price_reduction_per_year": Quant(0.1, ""),
        }

    def _run(self) -> dict:
        lifetime_dispatch_intervals = (
            self.plant_lifetime * 365 * 24 * self.dispatch_intervals_per_hour
        )
        age = (
            np.arange(lifetime_dispatch_intervals.magnitude)
            / (lifetime_dispatch_intervals / self.plant_lifetime).magnitude
        )

        b_E = self.battery_energy
        b_P = self.battery_power
        state_of_health = self.state_of_health
        battery_price_reduction_per_year = self.battery_price_reduction_per_year

        battery_energy_cost = self.battery_energy_cost
        battery_power_cost = self.battery_power_cost
        battery_BOP_installation_commissioning_cost = (
            self.battery_BOP_installation_commissioning_cost
        )
        battery_control_system_cost = self.battery_control_system_cost
        battery_energy_onm_cost = self.battery_energy_onm_cost

        ii_battery_change = np.where(
            (state_of_health > 0.99) & (np.append(1, np.diff(state_of_health)) > 0)
        )[0]
        year_new_battery = np.unique(np.floor(age[ii_battery_change]))

        factor = 1.0 - battery_price_reduction_per_year
        N_beq = np.sum([factor**iy for iy in year_new_battery])

        CAPEX = (
            N_beq * (battery_energy_cost * b_E)
            + (
                battery_power_cost
                + battery_BOP_installation_commissioning_cost
                + battery_control_system_cost
            )
            * b_P
        )

        OPEX = battery_energy_onm_cost * b_E

        return {
            "capex": CAPEX.to("MEUR"),
            "opex": OPEX.to("MEUR"),
        }


if __name__ == "__main__":
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
    print(res)

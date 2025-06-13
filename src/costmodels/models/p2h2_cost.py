import numpy as np

from costmodels.base import CostModel
from costmodels.units import Quant


class PowerToHydrogenCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {
            "electrolyzer_capacity": Quant(np.nan, "MW"),
            "hydrogen_storage_capacity": Quant(np.nan, "kg"),
            "mean_hydrogen_offtake": Quant(np.nan, "kg"),
            "electrolyzer_capex_cost": Quant(800000, "EUR/MW"),
            "electrolyzer_opex_cost": Quant(16000, "EUR/MW/year"),
            "electrolyzer_power_electronics_cost": Quant(0, "EUR/MW"),
            "water_cost": Quant(4, "EUR/m**3"),
            "water_treatment_cost": Quant(2, "EUR/m**3"),
            "water_consumption": Quant(9.4, "l/kg"),
            "storage_capex_cost": Quant(300, "EUR/kg"),
            "storage_opex_cost": Quant(3, "EUR/kg/year"),
            "transportation_cost": Quant(5, "EUR/kg/km"),
            "transportation_distance": Quant(0, "km"),
            "plant_lifetime": Quant(25, "year"),
            "dispatch_intervals_per_hour": Quant(1, "1/h"),
        }

    def _run(self) -> dict:
        yearly_intervals = (
            Quant(365, "day/year")
            * Quant(24, "hour/day")
            * self.dispatch_intervals_per_hour
        )
        lifetime_dispatch_intervals = self.plant_lifetime * yearly_intervals
        electrolyzer_capacity = self.electrolyzer_capacity
        hydrogen_storage_capacity = self.hydrogen_storage_capacity
        mean_hydrogen_offtake = self.mean_hydrogen_offtake

        electrolyzer_capex_cost = self.electrolyzer_capex_cost
        electrolyzer_opex_cost = self.electrolyzer_opex_cost
        electrolyzer_power_electronics_cost = self.electrolyzer_power_electronics_cost
        water_cost = self.water_cost
        water_treatment_cost = self.water_treatment_cost
        water_consumption = self.water_consumption
        storage_capex_cost = self.storage_capex_cost
        storage_opex_cost = self.storage_opex_cost
        transportation_cost = self.transportation_cost
        transportation_distance = self.transportation_distance

        CAPEX = (
            electrolyzer_capacity
            * (electrolyzer_capex_cost + electrolyzer_power_electronics_cost)
            + storage_capex_cost * hydrogen_storage_capacity
            + (
                mean_hydrogen_offtake
                * lifetime_dispatch_intervals
                * transportation_cost
                * transportation_distance
            )
        )
        water_consumption_cost = (
            mean_hydrogen_offtake
            * yearly_intervals
            * water_consumption
            * (water_cost + water_treatment_cost)
        )  # annual mean water consumption to produce hydrogen over an year
        OPEX = (
            electrolyzer_capacity * electrolyzer_opex_cost
            + storage_opex_cost * hydrogen_storage_capacity
            + water_consumption_cost
        )

        return {
            "capex": CAPEX.to("MEUR"),
            "opex": OPEX.to("MEUR/year"),
        }


if __name__ == "__main__":
    electrolyzer_capacity = Quant(800, "MW")
    hydrogen_storage_capacity = Quant(5000, "kg")
    mean_hydrogen_offtake = Quant(2343, "kg")
    PTHCM = PowerToHydrogenCostModel()
    res = PTHCM.run(
        electrolyzer_capacity=electrolyzer_capacity,
        hydrogen_storage_capacity=hydrogen_storage_capacity,
        mean_hydrogen_offtake=mean_hydrogen_offtake,
    )
    print(res)

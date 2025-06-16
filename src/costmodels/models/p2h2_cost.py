import numpy as np

from costmodels.base import CostModel


class PowerToHydrogenCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {
            "electrolyzer_capacity": np.nan,
            "hydrogen_storage_capacity": np.nan,
            "mean_hydrogen_offtake": np.nan,
            "electrolyzer_capex_cost": 800000,
            "electrolyzer_opex_cost": 16000,
            "electrolyzer_power_electronics_cost": 0,
            "water_cost": 4,
            "water_treatment_cost": 2,
            "water_consumption": 9.4e-3,
            "storage_capex_cost": 300,
            "storage_opex_cost": 3,
            "transportation_cost": 5,
            "transportation_distance": 0,
            "plant_lifetime": 25,
            "dispatch_intervals_per_hour": 1,
        }

    def _run(self) -> dict:
        yearly_intervals = 365 * 24 * self.dispatch_intervals_per_hour
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
            "capex": CAPEX / 1e6,
            "opex": OPEX / 1e6,
        }


if __name__ == "__main__":
    electrolyzer_capacity = 800
    hydrogen_storage_capacity = 5000
    mean_hydrogen_offtake = 2343
    PTHCM = PowerToHydrogenCostModel()
    res = PTHCM.run(
        electrolyzer_capacity=electrolyzer_capacity,
        hydrogen_storage_capacity=hydrogen_storage_capacity,
        mean_hydrogen_offtake=mean_hydrogen_offtake,
    )
    print(res)

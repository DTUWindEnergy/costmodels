import numpy as np

from costmodels.base import CostModel


class PVCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {
            "solar_capacity": np.nan,  # MW
            "dc_ac_ratio": 1.5,
            "panel_cost": 1.1e5,  # EUR/MW
            "hardware_installation_cost": 1e5,  # EUR/MW
            "inverter_cost": 2e4,  # EUR/MW
            "fixed_onm_cost": 4.5e3,  #  "EUR/MW
        }

    def __validate_input(self):
        for key, value in self._cm_input.items():
            if isinstance(value, (int, float)) and np.isnan(value):
                raise ValueError(f"Value of {key} is not defined")

    def _run(self):
        self.__validate_input()

        capex = (
            (self.panel_cost + self.hardware_installation_cost) * self.dc_ac_ratio
            + self.inverter_cost
        ) * self.solar_capacity
        opex = self.fixed_onm_cost * self.solar_capacity * self.dc_ac_ratio

        return {
            "capex": capex,
            "opex": opex,
        }

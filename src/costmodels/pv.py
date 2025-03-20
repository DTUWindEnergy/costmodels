import numpy as np

from costmodels.base import CostModel
from costmodels.units import Quant


class PVCostModel(CostModel):

    @property
    def _cm_input_def(self):
        return {
            "solar_capacity": Quant(np.nan, "MW"),
            "dc_ac_ratio": 1.5,
            "panel_cost": Quant(1.1e5, "EUR/MW"),
            "hardware_installation_cost": Quant(1e5, "EUR/MW"),
            "inverter_cost": Quant(2e4, "EUR/MW"),
            "fixed_onm_cost": Quant(4.5e3, "EUR/MW"),
        }

    def __validate_input(self):
        for key, value in self._cm_input.items():
            if not hasattr(value, "m"):
                continue
            if np.isnan(value.m).any():
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

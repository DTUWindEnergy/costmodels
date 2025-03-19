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


if __name__ == "__main__":  # pragma: no cover
    solar_capacity = Quant(150, "MW")
    pv_cm = PVCostModel(
        panel_cost=1.1e5,
        hardware_installation_cost=1e5,
        inverter_cost=2e4,
        fixed_onm_cost=4.5e3,
        dc_ac_ratio=1.5,
    )
    pv_cm.print_input()

    output = pv_cm.run(solar_capacity=solar_capacity)
    print("Output:")
    print(f"  CAPEX = {output["capex"]:.2f}")
    print(f"  OPEX = {output["opex"]:.3f}")

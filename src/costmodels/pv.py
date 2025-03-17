import numpy as np
import pint

from costmodels.units import Quant


class PVCostModel:
    def __init__(self, **kwargs):
        self._cm_input = {
            "solar_capacity": Quant(np.nan, "MW"),
            "dc_ac_ratio": Quant(1.5, "dimensionless"),
            "panel_cost": Quant(1.1e5, "EUR/MW"),
            "hardware_installation_cost": Quant(1e5, "EUR/MW"),
            "inverter_cost": Quant(2e4, "EUR/MW"),
            "fixed_onm_cost": Quant(4.5e3, "EUR/MW"),
        }
        self.__set_input(**kwargs)

    def __getattr__(self, name):
        if name in super().__getattribute__("_cm_input"):
            return self._cm_input[name]
        return super().__getattribute__(name)

    def __set_input(self, **kwargs):
        for key, value in kwargs.items():
            units = self._cm_input[key].units
            try:
                quant = (
                    value.to(units) if isinstance(value, Quant) else Quant(value, units)
                )
            except pint.errors.DimensionalityError:
                raise ValueError(
                    f"Invalid unit for '{key}'; Expected [{units}] and got [{value.units}]."
                )
            self._cm_input[key] = quant

    def __validate_input(self):
        for key, value in self._cm_input.items():
            if np.isnan(value.m).any():
                raise ValueError(f"Value of {key} is not defined")

    def run(self, **kwargs):
        self.__set_input(**kwargs)
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


if __name__ == "__main__":
    solar_capacity = Quant(150, "MW")
    pv_cm = PVCostModel(
        panel_cost=1.1e5,
        hardware_installation_cost=1e5,
        inverter_cost=2e4,
        fixed_onm_cost=4.5e3,
    )
    output = pv_cm.run(solar_capacity=solar_capacity)
    del pv_cm
    print(f"CAPEX = {output["capex"]:.2f}")
    print(f"OPEX = {output["opex"]:.3f}")

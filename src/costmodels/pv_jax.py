from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from costmodels.base import CostModel
from costmodels.units import Quant


def _runx(idict):
    capex = (
        (idict["panel_cost"] + idict["hardware_installation_cost"])
        * idict["dc_ac_ratio"]
        + idict["inverter_cost"]
    ) * sum(idict["solar_capacity"])
    opex = idict["fixed_onm_cost"] * sum(idict["solar_capacity"]) * idict["dc_ac_ratio"]

    return {
        "capex": capex,
        "opex": opex,
    }


runjacobian: Callable = jax.jit(jax.jacrev(_runx))


def _input_dict_to_magnitudes(idict):
    """Convert all Quant values in the input dictionary to their magnitudes."""
    return {
        key: (
            jnp.array(value.magnitude, dtype=jnp.float32)
            if isinstance(value, Quant)
            else value
        )
        for key, value in idict.items()
    }


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

    def run(self, **kwargs):
        self._set_input(**kwargs)
        return _runx(self._cm_input), runjacobian(
            _input_dict_to_magnitudes(self._cm_input)
        )


def pprint_dict(d):
    """Pretty print a dictionary."""
    for key, value in d.items():
        if isinstance(value, Quant):
            print(f"{key}: {value.magnitude} {value.units}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    solar_capacity = Quant(jnp.array([150, 100]), "MW")
    pv_cm = PVCostModel()
    pprint_dict(pv_cm._cm_input)
    cm_output = pv_cm.run(solar_capacity=solar_capacity)
    print(cm_output[1])
    exit(0)

    # class method jacobian
    import jax
    import jax.numpy as jnp

    class MyClass:
        @staticmethod
        def afunc(params):
            return jnp.sin(params["a"] + params["b"] ** 2)

    MyClass.afunc_jacobian = jax.jacrev(MyClass.afunc)

    params = {"a": 1.0, "b": 2.0}
    jac = MyClass.afunc_jacobian(params)
    print(jac)

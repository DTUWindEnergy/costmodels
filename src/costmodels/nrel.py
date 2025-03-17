from enum import Enum
from numbers import Number

import numpy as np
import pint

from costmodels.units import Quant


class TurbineClass(Enum):
    O = 0
    I = 1
    II = 2


class NRELCM:

    def __init__(self, **kwargs):
        self._cm_input = {
            "nwt": Quant(np.nan, "count"),
            "machine_rating": Quant(np.nan, "W"),
            "rotor_diameter": Quant(np.nan, "m"),
            "turbine_class": TurbineClass.II,
            "tower_length": Quant(np.nan, "m"),
            "blade_number": Quant(np.nan, "count"),
            "blade_has_carbon": False,
            "max_tip_speed": Quant(np.nan, "m/s"),
            "max_efficiency": Quant(np.nan, "%"),
            "main_bearing_number": Quant(np.nan, "count"),
            "crane": False,
            "eprice": Quant(0.2, "EUR/kWh"),
            "inflation": Quant(2, "%"),
            "lifetime": Quant(20, "count"),
            "opex": Quant(0.0, "EUR/kW"),  # TODO: per year?
        }
        self.__set_input(**kwargs)

        from openmdao.api import Problem  # fmt:skip isort:skip
        from costmodels.external.nrel_csm_mass_2015 import (  # fmt:skip isort:skip
            nrel_csm_2015,
        )

        self.org_impl = nrel_csm_2015()
        self.prob = Problem(reports=False)
        self.prob.model = nrel_csm_2015()
        self.prob.setup()
        super().__init__()

    def __getattr__(self, name):
        if name in super().__getattribute__("_cm_input"):
            return self._cm_input[name]
        return super().__getattribute__(name)

    def __set_input(self, **kwargs):
        for key, value in kwargs.items():
            assert key in self._cm_input.keys(), f"Invalid input '{key}'"

            if isinstance(self._cm_input[key], (Enum, bool)):
                assert isinstance(
                    value, type(self._cm_input[key])
                ), f"Invalid type for '{key}'"
                self._cm_input[key] = value
            elif isinstance(self._cm_input[key], (Quant, Number, np.number)):
                units = self._cm_input[key].units
                try:
                    quant = (
                        value.to(units)
                        if isinstance(value, Quant)
                        else Quant(value, units)
                    )
                except pint.errors.DimensionalityError:
                    raise ValueError(
                        f"Invalid unit for '{key}'; Expected [{units}] and got [{value.units}]."
                    )
                self._cm_input[key] = quant
            else:
                raise ValueError(f"Invalid type for '{key}'")

    def run(self, **kwargs):
        self.__set_input(**kwargs)

        self.prob["machine_rating"] = self.machine_rating.to("kW").m
        self.prob["rotor_diameter"] = self.rotor_diameter.m
        self.prob["turbine_class"] = self.turbine_class.value
        self.prob["tower_length"] = self.tower_length.m
        self.prob["blade_number"] = self.blade_number.m
        self.prob["blade_has_carbon"] = self.blade_has_carbon
        self.prob["max_tip_speed"] = self.max_tip_speed.m
        self.prob["max_efficiency"] = self.max_efficiency.to_base_units().m
        self.prob["main_bearing_number"] = self.main_bearing_number.m
        self.prob["crane"] = self.crane

        self.prob.run_model()

        wtc = self.prob.model._outputs["turbine_cost"][0]
        capex = Quant(wtc, "EUR") * self.nwt
        opex_total = self.opex * self.machine_rating
        # cashflows = self.cashflows(self, capex, opex_total, self.aep, self.lifetime)

        return {
            "capex": capex,
            "opex": opex_total.to_base_units(),
            # "lcoe": self.lceo(capex, opex_total, self.aep, self.lifetime),
            # "npv": self.npv(self.inflation.to_base_units().m, cashflows),
            # "irr": self.irr(cashflows),
        }

    def _list_inputs(self):
        return self.prob.model.list_inputs(units=True)

    def _list_outputs(self):
        return self.prob.model.list_outputs(units=True)


if __name__ == "__main__":

    model = NRELCM(
        eprice=0.2,
        inflation=2,
        nwt=10,
        rotor_diameter=Quant(126.0, "m"),
        turbine_class=TurbineClass.II,
        tower_length=Quant(90.0, "m"),
        blade_number=Quant(3, "count"),
        blade_has_carbon=False,
        max_tip_speed=Quant(80.0, "m/s"),
        max_efficiency=Quant(90, "%"),
        main_bearing_number=Quant(2, "count"),
        crane=True,
        lifetime=20,
    )
    cmo = model.run(
        machine_rating=Quant(5000.0, "kW"),
    )
    print(cmo["opex"])

    # TODO;
    # grads = model.grad(cmi, "capex", ("machine_rating",))
    # assert "machine_rating" in grads.keys()
    # print(grads)

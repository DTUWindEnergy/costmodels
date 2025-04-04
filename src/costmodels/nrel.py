from enum import Enum

import numpy as np

from costmodels.base import CostModel
from costmodels.units import Quant


class TurbineClass(Enum):
    O = 0
    I = 1
    II = 2


class NRELCostModel(CostModel):

    @property
    def _cm_input_def(self):
        return {
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
            "opex": Quant(np.nan, "EUR/kW"),
            "aep": Quant(np.nan, "MWh"),
        }

    def __init__(self, **kwargs):
        from openmdao.api import Problem  # fmt:skip isort:skip
        from costmodels.external.nrel_csm_mass_2015 import (  # fmt:skip isort:skip
            nrel_csm_2015,
        )

        self.org_impl = nrel_csm_2015()
        self.prob = Problem(reports=False)
        self.prob.model = nrel_csm_2015()
        self.prob.setup()
        super().__init__(**kwargs)

    def _run(self):
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
        cashflows = self.cashflows(
            self.eprice, self.inflation, capex, opex_total, self.aep, self.lifetime
        )

        return {
            "capex": capex.to("EUR"),
            "opex": opex_total.to_reduced_units(),
            "lcoe": self.lcoe(cashflows, self.aep * self.lifetime),
            "npv": self.npv(self.inflation, cashflows),
            "irr": self.irr(cashflows),
        }

    def _list_inputs(self):
        return self.prob.model.list_inputs(units=True)

    def _list_outputs(self):
        return self.prob.model.list_outputs(units=True)

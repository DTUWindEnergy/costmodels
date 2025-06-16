from enum import Enum

import numpy as np

from costmodels.base import CostModel


class NRELTurbineClass(Enum):
    O = 0
    I = 1
    II = 2


class NRELCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {
            "nwt": np.nan,
            "machine_rating": np.nan,
            "rotor_diameter": np.nan,
            "turbine_class": NRELTurbineClass.II,
            "tower_length": np.nan,
            "blade_number": np.nan,
            "blade_has_carbon": False,
            "max_tip_speed": np.nan,
            "max_efficiency": np.nan,
            "main_bearing_number": np.nan,
            "crane": False,
            "eprice": 0.2,
            "inflation": 2,
            "lifetime": 20,
            "opex": np.nan,
            "aep": np.nan,
        }

    def __init__(self, **kwargs):
        # check if openmdao is installed
        try:
            import openmdao  # noqa: F401
        except ImportError:  # pragma: no cover
            raise ImportError(
                "openmdao is not installed. Please install it to use the NREL cost model."
            )
        from openmdao.api import Problem  # fmt:skip isort:skip
        from .external.nrel_csm_mass_2015 import (  # fmt:skip isort:skip
            nrel_csm_2015,
        )

        self.org_impl = nrel_csm_2015()
        self.prob = Problem(reports=False)
        self.prob.model = nrel_csm_2015()
        self.prob.setup()
        super().__init__(**kwargs)

    def _run(self):
        self.prob["machine_rating"] = self.machine_rating
        self.prob["rotor_diameter"] = self.rotor_diameter
        self.prob["turbine_class"] = self.turbine_class.value
        self.prob["tower_length"] = self.tower_length
        self.prob["blade_number"] = self.blade_number
        self.prob["blade_has_carbon"] = self.blade_has_carbon
        self.prob["max_tip_speed"] = self.max_tip_speed
        self.prob["max_efficiency"] = self.max_efficiency / 100
        self.prob["main_bearing_number"] = self.main_bearing_number
        self.prob["crane"] = self.crane

        self.prob.run_model()

        wtc = self.prob.model._outputs["turbine_cost"][0]
        capex = wtc * self.nwt
        opex_total = self.opex * self.machine_rating
        # cashflows = compute_cashflows(
        #     self.eprice, self.inflation, capex, opex_total, self.aep, self.lifetime
        # )

        return {
            "capex": capex,
            "opex": opex_total,
            # "lcoe": self.lcoe(cashflows, self.aep * self.lifetime.m),
            # "npv": self.npv(self.inflation, cashflows),
            # "irr": self.irr(cashflows),
            # "cashflows": cashflows,
        }

    def _list_inputs(self):
        return self.prob.model.list_inputs(units=True)

    def _list_outputs(self):
        return self.prob.model.list_outputs(units=True)

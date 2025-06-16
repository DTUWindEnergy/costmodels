from enum import Enum

import numpy as np

from costmodels.base import CostModel


class NRELTurbineClass(Enum):
    CLASS_O = 0
    CLASS_I = 1
    CLASS_II = 2


class NRELCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {
            "nwt": np.nan,  # count
            "machine_rating": np.nan,  # W
            "rotor_diameter": np.nan,  # m
            "turbine_class": NRELTurbineClass.CLASS_O,
            "tower_length": np.nan,  # m
            "blade_number": np.nan,  # count
            "blade_has_carbon": False,
            "max_tip_speed": np.nan,  # m/s
            "max_efficiency": np.nan,  # %
            "main_bearing_number": np.nan,  # count
            "crane": False,
            "opex": np.nan,  # EUR/kW
            "aep": np.nan,  # MWh
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

        return {
            "capex": capex,
            "opex": opex_total,
        }

    def _list_inputs(self):
        return self.prob.model.list_inputs(units=True)

    def _list_outputs(self):
        return self.prob.model.list_outputs(units=True)

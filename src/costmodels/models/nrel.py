from enum import Enum

from costmodels._interface import CostModel, CostOutput, cost_input_dataclass


class NRELTurbineClass(Enum):
    CLASS_O = 0
    CLASS_I = 1
    CLASS_II = 2


@cost_input_dataclass
class NRELCostInput:
    nwt: float
    machine_rating: float
    rotor_diameter: float
    tower_length: float
    blade_number: float
    max_tip_speed: float
    max_efficiency: float
    main_bearing_number: float
    opex: float
    aep: float
    crane: bool = False
    turbine_class: NRELTurbineClass = NRELTurbineClass.CLASS_O
    blade_has_carbon: bool = False


class NRELCostModel(CostModel):
    _inputs_cls = NRELCostInput

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

    def _run(self, inputs: NRELCostInput) -> CostOutput:
        self.prob["machine_rating"] = inputs.machine_rating
        self.prob["rotor_diameter"] = inputs.rotor_diameter
        self.prob["turbine_class"] = inputs.turbine_class.value
        self.prob["tower_length"] = inputs.tower_length
        self.prob["blade_number"] = inputs.blade_number
        self.prob["blade_has_carbon"] = inputs.blade_has_carbon
        self.prob["max_tip_speed"] = inputs.max_tip_speed
        self.prob["max_efficiency"] = inputs.max_efficiency / 100
        self.prob["main_bearing_number"] = inputs.main_bearing_number
        self.prob["crane"] = inputs.crane

        self.prob.run_model()

        wtc = self.prob.model._outputs["turbine_cost"][0]
        capex = wtc * inputs.nwt
        opex_total = inputs.opex * inputs.machine_rating

        return CostOutput(capex=capex, opex=opex_total)

    def _list_inputs(self):
        return self.prob.model.list_inputs(units=True)

    def _list_outputs(self):
        return self.prob.model.list_outputs(units=True)

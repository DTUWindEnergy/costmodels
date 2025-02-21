from enum import Enum
from typing import Annotated

from costmodels.base import CostModel, CostModelInput, CostModelOutput
from costmodels.units import Quant, getppq


class TurbineClass(Enum):
    O = 0
    I = 1
    II = 2


class NRELCMInput(CostModelInput):

    nwt: Annotated[Quant, getppq("count")]
    machine_rating: Annotated[Quant, getppq("W")]
    rotor_diameter: Annotated[Quant, getppq("m")]
    turbine_class: TurbineClass
    tower_length: Annotated[Quant, getppq("m")]
    blade_number: Annotated[Quant, getppq("count")]
    blade_has_carbon: bool
    max_tip_speed: Annotated[Quant, getppq("m/s")]
    max_efficiency: Annotated[Quant, getppq("%")]
    main_bearing_number: Annotated[Quant, getppq("count")]
    crane: bool


class NRELCMOutput(CostModelOutput): ...


class NRELCM(CostModel):

    def __init__(self):
        from openmdao.api import Problem

        from costmodels.nrel_csm_mass_2015 import nrel_csm_2015

        self.org_impl = nrel_csm_2015()
        self.prob = Problem(reports=False)
        self.prob.model = nrel_csm_2015()
        self.prob.setup()
        super().__init__()

    def run(self, misepc: NRELCMInput) -> NRELCMOutput:
        self.prob["machine_rating"] = misepc.machine_rating.m
        self.prob["rotor_diameter"] = misepc.rotor_diameter.m
        self.prob["turbine_class"] = misepc.turbine_class.value
        self.prob["tower_length"] = misepc.tower_length.m
        self.prob["blade_number"] = misepc.blade_number.m
        self.prob["blade_has_carbon"] = False
        self.prob["max_tip_speed"] = misepc.max_tip_speed.m
        self.prob["max_efficiency"] = misepc.max_efficiency.m
        self.prob["main_bearing_number"] = misepc.main_bearing_number.m
        self.prob["crane"] = misepc.crane
        self.prob.run_model()

        self.prob.model.list_inputs(units=True)
        self.prob.model.list_outputs(units=True)

        wtc = self.prob.model._outputs["turbine_cost"]

        capex = Quant(wtc, "MEUR") * misepc.nwt

        return NRELCMOutput(
            capex=capex,
            opex=Quant(0.0, "MEUR"),
            lcoe=Quant(0.0, "EUR/MWh"),
            npv=Quant(0.0, "MEUR"),
            irr=Quant(0.0, "%"),
        )


if __name__ == "__main__":

    cmi = NRELCMInput(
        eprice=0.2,
        inflation=2,
        nwt=10,
        machine_rating=Quant(5000.0, "kW"),
        rotor_diameter=Quant(126.0, "m"),
        turbine_class=TurbineClass.II,
        tower_length=Quant(90.0, "m"),
        blade_number=Quant(3, "count"),
        blade_has_carbon=False,
        max_tip_speed=Quant(80.0, "m/s"),
        max_efficiency=Quant(0.90, "%"),
        main_bearing_number=Quant(2, "count"),
        crane=True,
    )

    model = NRELCM()
    model.run(cmi)

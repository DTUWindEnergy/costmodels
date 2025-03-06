from enum import Enum
from typing import Annotated

from costmodels.base import CostModel
from costmodels.units import Quant, getppq


class TurbineClass(Enum):
    O = 0
    I = 1
    II = 2


class NRELCM(CostModel):

    class Input(CostModel.Input):

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

    class Output(CostModel.Output): ...

    def __init__(self):
        from openmdao.api import Problem

        from costmodels.nrel_csm_mass_2015 import nrel_csm_2015

        self.org_impl = nrel_csm_2015()
        self.prob = Problem(reports=False)
        self.prob.model = nrel_csm_2015()
        self.prob.setup()
        super().__init__()

    def run(self, mispec: Input) -> Output:
        self.prob["machine_rating"] = mispec.machine_rating.to("kW").m
        self.prob["rotor_diameter"] = mispec.rotor_diameter.m
        self.prob["turbine_class"] = mispec.turbine_class.value
        self.prob["tower_length"] = mispec.tower_length.m
        self.prob["blade_number"] = mispec.blade_number.m
        self.prob["blade_has_carbon"] = False
        self.prob["max_tip_speed"] = mispec.max_tip_speed.m
        self.prob["max_efficiency"] = mispec.max_efficiency.m
        self.prob["main_bearing_number"] = mispec.main_bearing_number.m
        self.prob["crane"] = mispec.crane

        self.prob.run_model()

        wtc = self.prob.model._outputs["turbine_cost"][0]
        capex = Quant(wtc, "EUR") * mispec.nwt
        opex_total = mispec.opex * mispec.machine_rating
        cashflows = self.cashflows(
            mispec, capex, opex_total, mispec.aep, mispec.lifetime
        )

        return self.Output(
            capex=capex,
            opex=opex_total,
            lcoe=self.lceo(capex, opex_total, mispec.aep, mispec.lifetime),
            npv=self.npv(mispec.inflation.to_base_units().m, cashflows),
            irr=self.irr(cashflows),
        )

    def _list_inputs(self):
        return self.prob.model.list_inputs(units=True)

    def _list_outputs(self):
        return self.prob.model.list_outputs(units=True)


if __name__ == "__main__":

    cmi = NRELCM.Input(
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
        lifetime=20,
    )

    model = NRELCM()
    cmo = model.run(cmi)
    print(cmo)

    grads = model.grad(cmi, "capex", ("machine_rating",))
    assert "machine_rating" in grads.keys()
    print(grads)

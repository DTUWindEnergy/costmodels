raise RuntimeError("Experimental code, do not use.")

from enum import Enum
from typing import Annotated

import jax
from pydantic import Field
from pydantic_pint import PydanticPintQuantity, PydanticPintValue

from costmodels.base import CostModel, CostModelInput, CostModelOutput
from costmodels.units import Quant


class GradMethod(Enum):
    JAX = 0
    CD = 1


class GradModel(CostModel):

    class I(CostModelInput):
        nwt: Annotated[
            Quant,
            PydanticPintQuantity("count"),
            Field(gt=PydanticPintValue(0, "count")),
        ]
        turbine_cost: Annotated[
            Quant,
            PydanticPintQuantity("EUR"),
            Field(gt=PydanticPintValue(0, "EUR")),
        ]

    class O(CostModelOutput): ...

    def run(self, i: I) -> O:
        return GradModel.O(
            capex=self.capex(i.nwt, i.turbine_cost),
            opex=0,
            lcoe=0,
            npv=0,
            irr=0,
        )

    @staticmethod
    def capex(nwt, turbine_cost):
        return nwt * turbine_cost

    @staticmethod
    def __zero_out_input(i: I) -> I:
        i = i.model_copy()
        for field_name in i.model_fields:
            current_value = getattr(i, field_name)
            if isinstance(current_value, Quant):
                zero_value = Quant(0, current_value.units)
                setattr(i, field_name, zero_value)
            else:
                setattr(i, field_name, 0)
        return i

    def dcapex(self, i: I, method=GradMethod.JAX) -> I:
        if method == GradMethod.JAX:
            # in reality NWT is constant
            if not hasattr(self, "capex_jac_func"):
                self.capex_jac_func = jax.jit(jax.jacrev(self.capex, argnums=[0, 1]))
            capex_jac = self.capex_jac_func(float(i.nwt.m), i.turbine_cost.m)
            dcapex_nwt = capex_jac[0]
            dcapex_turbine_cost = capex_jac[1]
            i = self.__zero_out_input(i)
            i.nwt = Quant(dcapex_nwt, "count")
            i.turbine_cost = Quant(dcapex_turbine_cost, "EUR")
            return i

        elif method == GradMethod.CD:
            import numpy as np

            nwt_val = float(i.nwt.m)
            turbine_cost_val = float(i.turbine_cost.m)

            eps = 1e-6
            # Calculate derivative with respect to nwt
            h_nwt = max(eps * nwt_val, eps)
            nwt_points = np.array([nwt_val - h_nwt, nwt_val, nwt_val + h_nwt])
            capex_nwt_vals = np.array(
                [self.capex(n, turbine_cost_val) for n in nwt_points]
            )
            dcapex_nwt = np.gradient(capex_nwt_vals, nwt_points)[1]

            # Calculate derivative with respect to turbine_cost
            h_tc = max(eps * turbine_cost_val, eps)
            tc_points = np.array(
                [turbine_cost_val - h_tc, turbine_cost_val, turbine_cost_val + h_tc]
            )
            capex_tc_vals = np.array([self.capex(nwt_val, tc) for tc in tc_points])
            dcapex_turbine_cost = np.gradient(capex_tc_vals, tc_points)[1]

            i = self.__zero_out_input(i)
            i.nwt = Quant(dcapex_nwt, "count")
            i.turbine_cost = Quant(dcapex_turbine_cost, "EUR")
            return i

        else:
            raise ValueError("Invalid method")


if __name__ == "__main__":

    gcm = GradModel()

    cmi = GradModel.I(
        nwt=Quant(100, "count"),
        turbine_cost=Quant(1e6, "EUR"),
        eprice=Quant(0.1, "EUR/kWh"),
        inflation=Quant(2, "%"),
    )

    cmo = gcm.run(cmi)
    print(cmo)

    dcmi = gcm.dcapex(cmi, method=GradMethod.CD)
    print(dcmi)

    import timeit

    # benchmark jax vs CD
    print(
        timeit.timeit(
            lambda: gcm.dcapex(cmi, method=GradMethod.JAX),
            number=100,
        )
    )
    print(
        timeit.timeit(
            lambda: gcm.dcapex(cmi, method=GradMethod.CD),
            number=100,
        )
    )

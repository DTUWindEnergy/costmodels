from abc import ABC, abstractmethod
from typing import Annotated, Callable

import numpy as np
import numpy_financial as npf
from pydantic import AfterValidator
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field
from pydantic_pint import PydanticPintQuantity

from costmodels.units import IsValidPercent, Quant, getppq
from costmodels.utils import np2scalar


def _gtt(cmp: float) -> Callable:
    def _gtt(value: Quant) -> Quant:  # fmt:skip
        if value < Quant(cmp, value.units):
            raise ValueError(f"must be greater than {cmp}")
        return value
    return _gtt


class _StrReprInOut:
    def __new__(cls, *_, **__):
        if cls is _StrReprInOut:
            raise TypeError("StrRepr cannot be instantiated directly")
        return super().__new__(cls)

    def __str__(self: PydanticBaseModel):
        data = self.model_dump()
        header = f"{self.__class__.__name__}:"

        def __print_val(val):
            from numbers import Number

            from numpy import number

            if hasattr(val, "magnitude"):
                return round(float(val.magnitude), 3)
            elif isinstance(val, (Number, number)):
                return round(float(val), 3)
            return val

        return (
            header
            + "\n"
            + "\n".join(
                f"{key}: {__print_val(value)} {value.units if hasattr(value, 'units') else ''}"
                for key, value in data.items()
            )
        )


class CostModel(ABC):
    """Base class for all the cost models."""

    class Input(_StrReprInOut, ABC, PydanticBaseModel):
        """Base class for all the cost model inputs."""

        eprice: Annotated[Quant, getppq("EUR/kWh")]
        inflation: Annotated[Quant, getppq("%"), IsValidPercent]
        lifetime: int = Field(gt=0)
        aep: Annotated[Quant, PydanticPintQuantity("MWh", strict=False)] = Quant(
            0.0, "MWh"
        )
        opex: Annotated[Quant, PydanticPintQuantity("EUR/kW", strict=False)] = Quant(
            0.0, "MEUR/kW"
        )

    class Output(_StrReprInOut, ABC, PydanticBaseModel):
        """Base class for all the cost model outputs."""

        # Capital expenditure
        capex: Annotated[Quant, getppq("MEUR"), AfterValidator(_gtt(0))]
        # Operational expenditure
        opex: Annotated[Quant, getppq("MEUR"), AfterValidator(_gtt(0))]
        # Levelized cost of electricity
        lcoe: Annotated[Quant, getppq("EUR/MWh"), AfterValidator(_gtt(0))]
        # Net present value
        npv: Annotated[Quant, getppq("MEUR")]
        # Internal rate of return
        irr: Annotated[Quant, getppq("%")]

        model_config = ConfigDict(frozen=True)

    def __init__(self):  # pragma: no cover
        pass

    @abstractmethod
    def run(self, mispec: Input) -> Output:  # pragma: no cover
        """Abstract method to run the cost model.

        Parameters
        ----------
        mispec : CostModel.Input
            Model input specification.
        """
        pass

    def grad(
        self, input_spec: Input, of: str, wrt: list[str], delta: float = 1e-6
    ) -> dict:
        for pname in wrt:
            assert hasattr(input_spec, pname), f"Parameter {pname} not found in input."

        wrt_params = {
            pname: pval
            for pname, pval in input_spec.model_dump().items()
            if pname in wrt
        }

        gradients = {}
        for pname, pval in wrt_params.items():
            step = max(
                abs(pval * delta),
                delta if pval == 0 else abs(pval) * 1e-6,
            )
            input_plus = input_spec.model_copy(deep=True)
            input_minus = input_spec.model_copy(deep=True)

            setattr(input_plus, pname, pval + step)
            setattr(input_minus, pname, pval - step)

            output_plus = self.run(input_plus)
            output_minus = self.run(input_minus)

            plus_val = output_plus.model_dump()[of]
            minus_val = output_minus.model_dump()[of]

            if hasattr(plus_val, "magnitude"):
                plus_val = plus_val.magnitude
                minus_val = minus_val.magnitude
            if hasattr(step, "magnitude"):
                step = step.magnitude

            gradient = np.gradient(
                [np2scalar(minus_val), np2scalar(plus_val)], 2 * step
            )[1]
            gradients[pname] = gradient

        return gradients

    @staticmethod
    def cashflows(
        mispec: Input,
        capex: Quant,
        opex: Quant,
        aep: Quant,
        lifetime: int,
    ) -> list[float]:
        annual_revenue = aep * mispec.eprice
        assert annual_revenue.check(
            "EUR"
        ), f"Annual revenue must be in EUR not in {annual_revenue.units}"
        annual_cashflow = annual_revenue - opex
        cashflows = [-capex.magnitude] + [
            (annual_cashflow * ((1 + mispec.inflation) ** (year - 1)))
            .to_base_units()
            .magnitude
            for year in range(1, lifetime + 1)
        ]
        return cashflows

    @staticmethod
    def lceo(capex: Quant, opex: Quant, aep: Quant, lifetime: int) -> Quant:
        lceo = (capex + opex * lifetime) / (aep * lifetime / 10**6)
        assert lceo.check("EUR/Wh")
        return lceo

    @staticmethod
    def irr(cashflows: list[float]) -> Quant:
        return Quant(npf.irr(cashflows) * 100, "%")

    @staticmethod
    def npv(discount, cashflows):
        return Quant(npf.npv(discount, cashflows), "MEUR")


if __name__ == "__main__":

    cmi0 = CostModel.Input(eprice=0.2, inflation=2, lifetime=20)
    print(cmi0, "\n")
    cmi1 = CostModel.Input(
        eprice=Quant(0.2, "EUR/kWh"), inflation=Quant(2, "%"), lifetime=20
    )
    assert cmi0 == cmi1
    assert cmi0.eprice == cmi1.eprice
    assert cmi0.inflation == cmi1.inflation

    cmo = CostModel.Output(
        capex=Quant(1.0, "MEUR"),
        opex=Quant(0.1, "MEUR"),
        lcoe=Quant(10.0, "EUR/MWh"),
        npv=Quant(100.0, "MEUR"),
        irr=Quant(10.0, "%"),
    )
    print(cmo)

from abc import ABC, abstractmethod
from typing import Annotated, Callable

import numpy as np
from pydantic import AfterValidator
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict

from costmodels.units import Quant, getppq
from costmodels.utils import np2scalar


def _is_valid_percentage(value: Quant) -> Quant:
    if value < Quant(0, "%") or value > Quant(100, "%"):
        raise ValueError("percentage must be between 0 and 100")
    return value


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
                f"{key}: {__print_val(value)} {value.units if hasattr(value, "units") else ''}"
                for key, value in data.items()
            )
        )


class CostModelInput(_StrReprInOut, ABC, PydanticBaseModel):
    """Base class for all the cost model inputs."""

    eprice: Annotated[Quant, getppq("EUR/kWh")]
    inflation: Annotated[Quant, getppq("%"), AfterValidator(_is_valid_percentage)]


class CostModelOutput(_StrReprInOut, ABC, PydanticBaseModel):
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


class CostModel(ABC):
    """Base class for all the cost models."""

    def __init__(self):  # pragma: no cover
        pass

    @abstractmethod
    def run(self, mispec: CostModelInput):  # pragma: no cover
        """Abstract method to run the cost model.

        Parameters
        ----------
        mispec : CostModelInput
            Model input specification.
        """
        pass

    def grad(
        self, input_spec: CostModelInput, of: str, wrt: list[str], delta: float = 1e-6
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


if __name__ == "__main__":

    cmi0 = CostModelInput(
        eprice=0.2,
        inflation=2,
    )
    print(cmi0, "\n")
    cmi1 = CostModelInput(
        eprice=Quant(0.2, "EUR/kWh"),
        inflation=Quant(2, "%"),
    )
    assert cmi0 == cmi1
    assert cmi0.eprice == cmi1.eprice
    assert cmi0.inflation == cmi1.inflation

    cmo = CostModelOutput(
        capex=Quant(1.0, "MEUR"),
        opex=Quant(0.1, "MEUR"),
        lcoe=Quant(10.0, "EUR/MWh"),
        npv=Quant(100.0, "MEUR"),
        irr=Quant(10.0, "%"),
    )
    print(cmo)

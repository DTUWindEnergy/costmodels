from abc import ABC, abstractmethod
from typing import Annotated, Callable

from pydantic import AfterValidator
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict

from costmodels.units import Quant, getppq


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

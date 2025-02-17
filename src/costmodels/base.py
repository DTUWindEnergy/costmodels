from abc import ABC, abstractmethod
from typing import Annotated, Callable

from pydantic import AfterValidator
from pydantic import BaseModel as PydanticBaseModel

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


class CostModelInput(ABC, PydanticBaseModel):
    """Base class for all the cost model inputs."""

    eprice: Annotated[Quant, getppq("EUR/kWh")]
    inflation: Annotated[Quant, getppq("%"), AfterValidator(_is_valid_percentage)]

    def __str__(self):
        data = self.model_dump()
        print(f"{self.__class__.__name__}:")
        return "\n".join(
            f"{key}: {round(float(value.magnitude), 2)} {value.units}"
            for key, value in data.items()
        )


class CostModelOutput(ABC, PydanticBaseModel):
    """Base class for all the cost model outputs."""

    capex: Annotated[Quant, getppq("MEUR"), AfterValidator(_gtt(0))]
    opex: Annotated[Quant, getppq("MEUR"), AfterValidator(_gtt(0))]
    lcoe: Annotated[Quant, getppq("EUR/MWh"), AfterValidator(_gtt(0))]
    npv: Annotated[Quant, getppq("MEUR")]
    irr: Annotated[Quant, getppq("%")]

    class Config:
        frozen = True

    def __str__(self):
        data = self.model_dump()
        print(f"{self.__class__.__name__} data:")
        return "\n".join(
            f"{key}: {round(float(value.magnitude), 2)} {value.units}"
            for key, value in data.items()
        )


class CostModel(ABC):
    """Base class for all the cost models."""

    def __init__(self):
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

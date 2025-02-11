from abc import abstractmethod, ABC
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from costmodels.ufloat import ufloat


class CostModel(ABC):
    """Base class for all the cost models."""

    def __init__(self):
        pass

    @abstractmethod
    def run(self):  # pragma: no cover
        pass


class CostModelInput(ABC, PydanticBaseModel):
    """Base class for all the cost model inputs."""

    eprice: ufloat
    inflation: float = Field(default=0.02, gt=0)  # %


# TODO: remove the commented units; now it's going to be programatically added
# TODO: could be a frozen dataclass
# from pydantic.dataclasses import dataclass
# @dataclass(frozen=True)
class CostModelOutput(ABC, PydanticBaseModel):
    """Base class for all the cost model outputs."""

    # fmt:off
    capex: ufloat    # M€
    opex: ufloat     # M€
    lcoe: ufloat     # €/MWh
    npv: ufloat      # M€
    irr: ufloat      # %
    # fmt:on

    def __str__(self):
        return (
            f"CAPEX:\t{self.capex}\n"
            f"OPEX:\t{self.opex}\n"
            f"LCoE:\t{self.lcoe}\n"
            f"NPV:\t{self.npv}\n"
            f"IRR:\t{self.irr}"
        )


if __name__ == "__main__":
    cmo = CostModelOutput(
        capex=1.0,
        opex=1.0,
        lcoe=1.0,
        npv=1.0,
        irr=1.0,
    )
    print(cmo)

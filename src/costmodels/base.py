from abc import abstractmethod, ABC
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field


class CostModel(ABC):
    """Base class for all the cost models."""

    def __init__(self):
        pass

    @abstractmethod
    def run(self):  # pragma: no cover
        pass


class CostModelInput(ABC, PydanticBaseModel):
    """Base class for all the cost model inputs."""

    eprice: float  # €/kWh
    inflation: float = Field(default=0.02, gt=0)  # %


# TODO: could be a frozen dataclass
# from pydantic.dataclasses import dataclass
# @dataclass(frozen=True)
class CostModelOutput(ABC, PydanticBaseModel):
    """Base class for all the cost model outputs."""

    # fmt:off
    capex: float    # M€
    opex: float     # M€
    lcoe: float     # €/MWh
    npv: float      # M€
    irr: float      # %
    # fmt:on

    def __str__(self):
        return (
            f"CAPEX:\t{self.capex} M€\n"
            f"OPEX:\t{self.opex} M€\n"
            f"LCoE:\t{self.lcoe} €/MWh\n"
            f"NPV:\t{self.npv} M€\n"
            f"IRR:\t{self.irr} %"
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

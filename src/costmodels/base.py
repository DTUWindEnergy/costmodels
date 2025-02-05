from abc import abstractmethod, ABC
from pydantic import BaseModel as PydanticBaseModel


class CostModel(ABC):
    """Base class for all the cost models."""

    def __init__(self):
        pass

    @abstractmethod
    def run(self):  # pragma: no cover
        pass


class CostModelInput(ABC, PydanticBaseModel):
    """Base class for all the cost model inputs."""


class CostModelOutput(ABC, PydanticBaseModel):
    """Base class for all the cost model outputs."""

    capex: float
    opex: float
    lcoe: float
    npv: float
    irr: float

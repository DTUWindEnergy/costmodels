from abc import abstractmethod, ABC
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
import numpy as np
from dataclasses import dataclass
import operator

from typing import Any

from pydantic_core import core_schema
from typing_extensions import Annotated

from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue


@dataclass
class uval:
    """Unit tagged value"""

    value: np.number | float | int
    unit: str

    def __str__(self):
        return f"{self.value} {self.unit}"

    def __op_exec(self, other, op, funit, unitmatch=False):
        if not isinstance(other, uval):
            try:
                return uval(op(self.value, other), funit)
            except Exception as e:
                raise ValueError(
                    f"Cannot {op.__name__} {type(other)} to {type(self.value)}"
                ) from e

        if unitmatch and self.unit != other.unit:
            raise ValueError(
                f"Units must match! Cannot perform {op.__name__}"
                f" between {self.unit} and {other.unit}"
            )

        return uval(op(self.value, other.value), funit)

    def __add__(self, other):
        self.__op_exec(other, operator.add, self.unit, True)

    def __sub__(self, other):
        self.__op_exec(other, operator.sub, self.unit, True)

    def __mul__(self, other):
        ounit = getattr(other, "unit", None)
        self.__op_exec(other, operator.truediv, f"{self.unit} * {ounit}")

    def __truediv__(self, other):
        ounit = getattr(other, "unit", None)
        self.__op_exec(other, operator.truediv, f"{self.unit} / {ounit}")

    @staticmethod  # TODO;
    def __simplify_unit(unit):
        raise NotImplementedError


class _ThirdPartyTypePydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        Return a CoreSchema with the following behavior:

        * Accepts floats to produce a `ThirdPartyType` with the float value as `x`
        * Accepts `ThirdPartyType` instances and returns them unchanged
        * Fails validation for any other type
        * Serializes to a float (the `x` attribute)
        """

        def validate_from_float(value: float, info) -> uval:
            print(info)
            result = uval()
            result.value = value
            return result

        # Create a schema that parses a float and then converts it to ThirdPartyType.
        from_float_schema = core_schema.chain_schema(
            [
                core_schema.float_schema(),
                core_schema.with_info_plain_validator_function(validate_from_float),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_float_schema,
            python_schema=core_schema.union_schema(
                [
                    # First check if the instance is already a ThirdPartyType
                    core_schema.is_instance_schema(uval),
                    from_float_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.x
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Expose the JSON schema for a float.
        return handler(core_schema.float_schema())


# Use Annotated to wrap ThirdPartyType with the custom Pydantic integration.
PydanticThirdPartyType = Annotated[ThirdPartyType, _ThirdPartyTypePydanticAnnotation]


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


class CostModelOutput(ABC, PydanticBaseModel):
    """Base class for all the cost model outputs."""

    # fmt:off
    capex: uval    # M€.
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

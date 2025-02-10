from pydantic import BaseModel, Field, ConfigDict, TypeAdapter, GetCoreSchemaHandler
from pydantic_core import core_schema
from typing import Any, Tuple
import numpy as np
from pydantic.dataclasses import dataclass

NO_UNIT = "-"


class ufloat(float):
    """A float with an attached unit, compatible with Pydantic validation."""

    def __new__(cls, value: float, unit: str):
        obj = super().__new__(cls, value)
        obj.unit = unit
        return obj

    def __repr__(self):
        return f"{float(self)} {self.unit}"

    def __str__(self):
        return f"{float(self)} {self.unit}"

    def as_tuple(self) -> Tuple[float, str]:
        return float(self), self.unit

    @classmethod
    def __get_pydantic_core_schema__(cls, *args, **kwargs) -> core_schema.CoreSchema:
        """Defines how Pydantic should validate this type."""
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any) -> "ufloat":
        """Validation logic for Pydantic."""
        if isinstance(value, cls):
            return value
        if isinstance(value, (int, float, np.number)):
            return cls(value, NO_UNIT)  # default unit

        raise TypeError(f"Invalid type for FloatUnit: {type(value)}")

    # TODO: all of the below !!!

    # def __add__(self, other):
    #     return ufloat(float.__add__(self, other), self.unit)

    # def __radd__(self, other):
    #     return ufloat(float.__radd__(self, other), self.unit)

    # def __sub__(self, other):
    #     return ufloat(float.__sub__(self, other), self.unit)

    # def __rsub__(self, other):
    #     return ufloat(float.__rsub__(self, other), self.unit)

    # def __mul__(self, other):
    #     return ufloat(float.__mul__(self, other), self.unit)

    # def __rmul__(self, other):
    #     return ufloat(float.__rmul__(self, other), self.unit)

    # def __truediv__(self, other):
    #     return ufloat(float.__truediv__(self, other), self.unit)

    # def __rtruediv__(self, other):
    #     return ufloat(float.__rtruediv__(self, other), self.unit)

    # def __floordiv__(self, other):
    #     return ufloat(float.__floordiv__(self, other), self.unit)

    # def __rfloordiv__(self, other):
    #     return ufloat(float.__rfloordiv__(self, other), self.unit)

    # def __mod__(self, other):
    #     return ufloat(float.__mod__(self, other), self.unit)

    # def __rmod__(self, other):
    #     return ufloat(float.__rmod__(self, other), self.unit)

    # def __divmod__(self, other):
    #     q, r = divmod(float(self), other)
    #     return (ufloat(q, self.unit), ufloat(r, self.unit))

    # def __rdivmod__(self, other):
    #     q, r = divmod(other, float(self))
    #     return (ufloat(q, self.unit), ufloat(r, self.unit))

    # def __pow__(self, other, modulo=None):
    #     if modulo is None:
    #         return ufloat(float.__pow__(self, other), self.unit)
    #     else:
    #         return ufloat(pow(float(self), other, modulo), self.unit)

    # def __rpow__(self, other):
    #     return ufloat(float.__rpow__(self, other), self.unit)

    # def __neg__(self):
    #     return ufloat(float.__neg__(self), self.unit)

    # def __pos__(self):
    #     return ufloat(float.__pos__(self), self.unit)

    # def __abs__(self):
    #     return ufloat(float.__abs__(self), self.unit)

    # def __round__(self, ndigits=None):
    #     return ufloat(round(float(self), ndigits), self.unit)


# TODO: could borrow some code for ops implementation from v0 of ufloat : )
# import operator
# class uval:
#     """Unit tagged value"""

#     unit: str
#     value: float | int = Field(..., gt=0)

#     def __init__(self, value, unit):
#         self.value = value
#         self.unit = unit

#     def __str__(self):
#         return f"{self.value} {self.unit}"

#     def __op_exec(self, other, op, funit, unitmatch=False):
#         if not isinstance(other, uval):
#             try:
#                 return uval(op(self.value, other), funit)
#             except Exception as e:
#                 raise ValueError(
#                     f"Cannot {op.__name__} {type(other)} to {type(self.value)}"
#                 ) from e

#         if unitmatch and self.unit != other.unit:
#             raise ValueError(
#                 f"Units must match! Cannot perform {op.__name__}"
#                 f" between {self.unit} and {other.unit}"
#             )

#         return uval(op(self.value, other.value), funit)

#     def __add__(self, other):
#         self.__op_exec(other, operator.add, self.unit, True)

#     def __sub__(self, other):
#         self.__op_exec(other, operator.sub, self.unit, True)

#     def __mul__(self, other):
#         ounit = getattr(other, "unit", None)
#         self.__op_exec(other, operator.truediv, f"{self.unit} * {ounit}")

#     def __truediv__(self, other):
#         ounit = getattr(other, "unit", None)
#         self.__op_exec(other, operator.truediv, f"{self.unit} / {ounit}")

#     @staticmethod
#     def __simplify_unit(unit):
#         raise NotImplementedError


if __name__ == "__main__":

    @dataclass
    class Measurement:
        length: ufloat = Field(..., gt=0)

        __pydantic_config__ = ConfigDict(validate_assignment=True)

    l = np.ones(10).sum()
    m = Measurement(length=ufloat(l, "m"))

    assert m.length == l
    assert m.length.unit == "m"
    assert m.length.as_tuple() == (10.0, "m")

    # TODO:
    # m.length += ufloat(2.0, "m")
    # assert m.length.as_tuple() == (12.0, "m")

    # TODO: assign & preserve the unit
    # m.length = 5.0
    # assert m.length.as_tuple() == (5.0, "m")

    m.length = ufloat(3.0, "m")
    assert m.length.as_tuple() == (3.0, "m"), m.length.as_tuple()

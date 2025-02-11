from pydantic import Field, ConfigDict
from pydantic_core import core_schema
from pydantic.dataclasses import dataclass
from typing import Any, Tuple
from costmodels.ureg import _ureg
import numpy as np

NO_UNIT = ""


class ufloat(float):
    """A float with an attached unit, compatible with Pydantic validation."""

    def __new__(cls, value: float, unit: str):
        obj = super().__new__(cls, value)
        try:
            _ureg.parse_units(str(unit))
        except:
            raise ValueError(f"Invalid unit: {unit}")
        obj.unit = unit
        return obj

    def __repr__(self):
        return f"{float(self)} {self.unit}"

    def __str__(self):
        return f"{float(self)} {self.unit}"

    def as_tuple(self) -> Tuple[float, str]:
        return float(self), self.unit

    def swap(self, value: float) -> "ufloat":
        return ufloat(value, self.unit)

    @classmethod
    def __get_pydantic_core_schema__(cls, *args, **kwargs) -> core_schema.CoreSchema:
        """Defines how Pydantic should validate this type."""
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any) -> "ufloat":
        """Validation logic for Pydantic."""
        if isinstance(value, cls):
            return value
        if isinstance(value, tuple):
            return cls(value[0], value[1])  # default unit

        raise TypeError(
            f"\tCannot create ufloat from: {type(value)};\n"
            f"\t\tShould be ufloat type; Or tuple of (value, unit)."
        )

    def __op_exec(self, other, op, funit, unitmatch=False):
        if not isinstance(other, ufloat):
            try:
                return ufloat(op(self, other), funit)
            except Exception as e:
                raise ValueError(
                    f"Cannot {op.__name__} {type(other)} to {type(self)}"
                ) from e

        if unitmatch and self.unit != other.unit:
            raise ValueError(
                f"Units must match! Cannot perform {op.__name__}"
                f" between {self.unit} and {other.unit}"
            )

        return ufloat(op(self, other), funit)

    def __add__(self, other):
        return self.__op_exec(other, float.__add__, self.unit, True)

    def __radd__(self, other):
        return self.__op_exec(other, float.__radd__, self.unit, True)

    def __sub__(self, other):
        return self.__op_exec(other, float.__sub__, self.unit, True)

    def __rsub__(self, other):
        return self.__op_exec(other, float.__rsub__, self.unit, True)

    def __mul__(self, other):
        runit = _ureg(self.unit) * _ureg(getattr(other, "unit", NO_UNIT))
        return self.__op_exec(other, float.__mul__, str(runit.units))

    def __rmul__(self, other):
        runit = _ureg(getattr(other, "unit", NO_UNIT)) * _ureg(self.unit)
        return self.__op_exec(other, float.__rmul__, str(runit.units))

    def __truediv__(self, other):
        runit = _ureg(self.unit) / _ureg(getattr(other, "unit", NO_UNIT))
        return self.__op_exec(other, float.__truediv__, str(runit.units))

    def __rtruediv__(self, other):
        runit = _ureg(getattr(other, "unit", NO_UNIT)) / _ureg(self.unit)
        return self.__op_exec(other, float.__rtruediv__, str(runit.units))

    # TODO: all of the below must be supported !!!

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

    # TODO: all of the above must be supported !!!


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

    m.length += ufloat(2.0, "m")
    assert m.length.as_tuple() == (12.0, "m")
    m.length += 3.0
    assert m.length.as_tuple() == (15.0, "m")
    radd = 3.0 + m.length
    assert radd.as_tuple() == (18.0, "m")

    m.length = ufloat(10.0, "m")
    m.length *= ufloat(2.0, "m")
    assert m.length.as_tuple() == (20.0, "meter ** 2"), m.length.as_tuple()
    m.length = ufloat(10.0, "m")
    m.length *= 3.0
    assert m.length.as_tuple() == (30.0, "meter"), m.length.as_tuple()
    m.length = ufloat(11.0, "m")
    rmul = 3.0 * m.length
    assert rmul.as_tuple() == (33.0, "meter"), rmul.as_tuple()

    m.length = ufloat(10.0, "m**2")
    m.length /= ufloat(2.0, "m")
    assert m.length.as_tuple() == (5.0, "meter"), m.length.as_tuple()
    m.length /= 5.0  # dimensionless division units preserved
    assert m.length.as_tuple() == (1.0, "meter"), m.length.as_tuple()
    rdiv = 10.0 / m.length
    assert rdiv.as_tuple() == (10.0, "1 / meter"), rdiv.as_tuple()
    rdiv_dimless = ufloat(7.0, "m") / m.length
    assert rdiv_dimless.as_tuple() == (7.0, "dimensionless"), rdiv_dimless.as_tuple()

    m.length = ufloat(3.0, "m")
    assert m.length.as_tuple() == (3.0, "m"), m.length.as_tuple()

    m.length = m.length.swap(5.0)
    assert m.length.as_tuple() == (5.0, "m"), m.length.as_tuple()

    m.length = (5.0, "s")
    assert m.length.as_tuple() == (5.0, "s"), m.length.as_tuple()

    import pytest

    with pytest.raises(TypeError, match="Cannot create ufloat from:"):
        m.length = 5.0
    with pytest.raises(TypeError, match="Cannot create ufloat from:"):
        m.length = np.array(5.0)
    with pytest.raises(ValueError, match="Invalid unit"):
        m.length = (np.float32(5.0), np.float32(5.0))

from pint import UnitRegistry
from pydantic_pint import PydanticPintQuantity

_UREG = UnitRegistry()
_UREG.define("EUR = [currency]")
_UREG.define("USD = [currency]")
_UREG.formatter.default_format = "~#P"
Quant = _UREG.Quantity


def getppq(units=""):
    """Get a PydanticPintQuantity object with global units registry."""
    return PydanticPintQuantity(units, ureg=_UREG, strict=False)

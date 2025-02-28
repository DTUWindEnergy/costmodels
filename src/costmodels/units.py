from pint import UnitRegistry
from pydantic_pint import PydanticPintQuantity, set_registry

_UREG = UnitRegistry()
_UREG.define("EUR = [currency]")
_UREG.define("USD = [currency]")
_UREG.formatter.default_format = "~#P"
Quant = _UREG.Quantity

set_registry(_UREG)


def getppq(units=""):
    """Get a PydanticPintQuantity object with global units registry."""
    return PydanticPintQuantity(units, strict=False)

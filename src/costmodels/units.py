from pint import UnitRegistry
from pydantic_pint import PydanticPintQuantity, set_registry

ureg = UnitRegistry()
ureg.define("EUR = [currency]")
ureg.define("USD = [currency]")
ureg.formatter.default_format = "~#P"
Quant = ureg.Quantity

set_registry(ureg)


def getppq(units=""):
    """Get a PydanticPintQuantity object with global units registry."""
    return PydanticPintQuantity(units, strict=False)

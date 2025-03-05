from pint import UnitRegistry
from pydantic import AfterValidator
from pydantic_pint import PydanticPintQuantity, set_registry

ureg = UnitRegistry()
ureg.define("EUR = [currency]")
ureg.define("USD = [currency]")
ureg.define("DKK = [currency]")
ureg.formatter.default_format = "~#P"
Quant = ureg.Quantity

set_registry(ureg)


def getppq(units=""):
    """Get a PydanticPintQuantity object with global units registry."""
    return PydanticPintQuantity(units, strict=False)


def is_valid_percentage(value: Quant) -> Quant:
    if value < Quant(0, "%") or value > Quant(100, "%"):
        raise ValueError("percentage must be between 0 and 100")
    return value


IsValidPercent = AfterValidator(is_valid_percentage)

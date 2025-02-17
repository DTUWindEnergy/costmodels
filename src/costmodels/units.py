from pint import UnitRegistry  # fmt:skip

_UREG = UnitRegistry()
_UREG.define("EUR = [currency]")
_UREG.formatter.default_format = "~#P"
Quant = _UREG.Quantity


from pydantic_pint import PydanticPintQuantity  # fmt:skip


def getppq(units=""):
    """Get a PydanticPintQuantity object with global units registry."""
    return PydanticPintQuantity(units, ureg=_UREG, strict=False)

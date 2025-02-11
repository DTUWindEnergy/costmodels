from pint import UnitRegistry  # fmt:skip
_ureg = UnitRegistry()
_ureg.define("EUR = [currency]")
_ureg.formatter.default_format = "~#P"

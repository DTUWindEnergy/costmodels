from pint import UnitRegistry

ureg = UnitRegistry()
ureg.define("EUR = [currency]")
ureg.define("USD = [currency]")
ureg.define("DKK = [currency]")
ureg.formatter.default_format = "~#P"
Quant = ureg.Quantity

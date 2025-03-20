from typing import TypeAlias

import pint

ureg = pint.UnitRegistry()
ureg.define("EUR = [currency]")
ureg.define("USD = [currency]")
ureg.define("DKK = [currency]")
ureg.formatter.default_format = "~#P"
pint.set_application_registry(ureg)
Quant: TypeAlias = pint.Quantity

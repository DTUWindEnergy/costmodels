import warnings
from abc import ABC, abstractmethod
from enum import Enum
from numbers import Number

import numpy as np
import numpy_financial as npf
import pint

from costmodels.units import Quant
from costmodels.utils import np2scalar


class CostModel(ABC):
    """Base class for all the cost models."""

    def __init__(self, **kwargs):
        self._cm_input = self._cm_input_def
        self._set_input(**kwargs)

    def __getattr__(self, name):
        # this makes sure to propagate any errors that occur in the
        # self._cm_input = self._cm_input_def assignment in constructor
        if name in ["_cm_input", "_cm_input_def"]:
            return super().__getattribute__(name)

        # treat keys of self._cm_input as attributes of the cost models
        if name in super().__getattribute__("_cm_input"):
            return self._cm_input[name]

        # default behavior
        return super().__getattribute__(name)

    @property
    @abstractmethod
    def _cm_input_def(self) -> dict:  # pragma: no cover
        """Definition of cost model input."""
        ...

    def list_input(self) -> dict:
        """List all inputs of the cost model."""
        formatted_inputs = {}
        for key, value in self._cm_input.items():
            formatted_inputs[key] = {
                "default": value,
                "type": type(value),
                "unit": value.units if isinstance(value, Quant) else None,
            }
        return formatted_inputs

    def print_input(self) -> None:
        """Show all inputs of the cost model."""
        print(f"{self.__class__.__name__} inputs:")
        for k, v in self.list_input().items():
            print(f"  {k}")
            default = v["default"].m if hasattr(v["default"], "m") else v["default"]
            print(f"\tDefault: {default}")
            print(f"\tType: {v['type']}")
            print(f"\tUnit: {v['unit'] if v['unit'] else 'N/A'}")

    def _set_input(self, **kwargs) -> None:
        for key, value in kwargs.items():
            assert key in self._cm_input.keys(), f"Invalid input '{key}'"

            if isinstance(self._cm_input[key], bool):
                assert isinstance(
                    value, type(self._cm_input[key])
                ), f"Invalid type for '{key}', must be {type(self._cm_input[key])}"
                self._cm_input[key] = value
            elif isinstance(self._cm_input[key], Enum):
                if not isinstance(value, Enum):
                    value = self._cm_input[key].__class__(value)
                self._cm_input[key] = value
            elif isinstance(self._cm_input[key], (Quant, Number, np.number)):
                is_quant_expected = isinstance(self._cm_input[key], Quant)
                if is_quant_expected:
                    # the default value is a Quantity and we try to assign units to provided number
                    units = self._cm_input[key].units
                    try:
                        quant = (
                            value.to(units)
                            if isinstance(value, Quant)
                            else Quant(value, units)
                        )
                    except pint.errors.DimensionalityError:
                        raise ValueError(
                            f"Invalid unit for '{key}'; Expected [{units}] and got [{value.units}]."
                        )
                else:
                    # keep a unitless number; input specification does not expect a unit
                    quant = value
                self._cm_input[key] = quant
            else:
                raise ValueError(f"Invalid type for '{key}'")

    def run(self, **kwargs) -> dict:
        self._set_input(**kwargs)
        return self._run()

    @abstractmethod
    def _run(self) -> dict:  # pragma: no cover
        """Abstract method to run the cost model."""
        pass

    def grad(self, of: str, wrt: list[str] | tuple[str], delta: float = 1e-6) -> dict:
        """Compute the gradient of the cost model output with respect to the input parameters."""
        gradients = {}
        for pname in wrt:
            assert hasattr(
                self, pname
            ), f"Parameter {pname} not found in cost model input."
            pval = getattr(self, pname)

            step = max(
                abs(pval * delta),
                delta if pval == 0 else abs(pval) * 1e-6,
            )

            output_plus = self.run(**{pname: pval + step})
            output_minus = self.run(**{pname: pval - step})
            # reset the input value to the original value
            self._set_input(**{pname: pval})

            try:
                plus_val = output_plus[of]
                minus_val = output_minus[of]
            except KeyError:
                raise KeyError(
                    f"Output '{of}' not found in the cost model. Available outputs: {output_plus.keys()}"
                )

            if hasattr(plus_val, "magnitude"):
                plus_val = plus_val.magnitude
                minus_val = minus_val.magnitude
            if hasattr(step, "magnitude"):
                step = step.magnitude

            gradient = np.gradient(
                [np2scalar(minus_val), np2scalar(plus_val)], 2 * step
            )[1]
            gradients[pname] = gradient

        return gradients

    @staticmethod
    def cashflows(
        epice: Quant,
        inflation: Quant,
        capex: Quant,
        opex: Quant,
        aep: Quant,
        lifetime: int | Quant,
    ) -> Quant:
        annual_revenue = aep * epice
        annual_cashflow = annual_revenue - opex
        lifetime = lifetime.m if isinstance(lifetime, Quant) else lifetime
        if not annual_cashflow.check(capex.units):
            raise ValueError(
                f"annual_cashflow units {annual_cashflow.units} do not match capex units {capex.units}"
            )
        cashflows = [-capex.to_base_units().m] + [
            (annual_cashflow * ((1 + inflation) ** (year - 1))).to_base_units().m
            for year in range(1, lifetime + 1)
        ]
        qcashflows = Quant(cashflows, annual_cashflow.units)
        qcashflows.ito_reduced_units()
        return qcashflows

    @staticmethod
    def lceo(cashflows: Quant, aep: Quant) -> Quant:
        return Quant(0.0, "EUR/MWh")  # TODO:

    NAN_RETURN_WARN = (
        "Cashflows contain NaN values. Returning NaN for $var. "
        "The input data is likely missing values like AEP or OPEX."
    )

    @staticmethod
    def irr(cashflows: Quant) -> Quant:
        if np.isnan(cashflows.m).any():
            warnings.warn(CostModel.NAN_RETURN_WARN.replace("$var", "IRR"))
            return Quant(np.nan, "%")
        return Quant(npf.irr(cashflows.m) * 100, "%")

    @staticmethod
    def npv(discount: Quant, cashflows: Quant):
        if np.isnan(cashflows.m).any():
            warnings.warn(CostModel.NAN_RETURN_WARN.replace("$var", "NPV"))
            return Quant(np.nan, "MEUR")
        return Quant(npf.npv(discount.to_base_units().m, cashflows.m), cashflows.units)

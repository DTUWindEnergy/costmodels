"""Base classes for the new cost model API.

The upcoming API requires cost models to subclass :class:`CostModel` and
implement a static, side-effect free :meth:`CostModel._run` method that
operates solely on ``np.ndarray`` inputs. Existing models still follow the old
interface but will be migrated over time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np
import pint
from autograd import value_and_grad
from autograd.numpy.numpy_boxes import ArrayBox

from costmodels.units import Quant


@dataclass
class CostModelOutput:
    capex: float
    opex: float


def _input_dict_to_magnitudes(idict: dict) -> dict[str, np.ndarray]:
    """Convert all :class:`~costmodels.units.Quant` values to ``np.ndarray``.

    Parameters
    ----------
    idict:
        Input dictionary where values may be ``pint.Quantity`` objects.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with the same keys but with all numeric values converted to
        ``np.ndarray`` of ``dtype=np.floatX``.
    """
    processed_dict = {}
    for key, value_in_dict in idict.items():
        if isinstance(value_in_dict, Quant):
            mag = value_in_dict.magnitude
        else:
            mag = value_in_dict

        if isinstance(mag, ArrayBox):
            processed_dict[key] = mag
        else:
            processed_dict[key] = anp.array(mag)
    return processed_dict


class CostModel(ABC):
    """Base class for all the cost models."""

    @property
    @abstractmethod
    def _cm_input_def(self) -> dict:  # pragma: no cover
        """Definition of cost model input."""
        ...

    @staticmethod
    @abstractmethod
    def _run(x: dict[str, np.ndarray]) -> CostModelOutput:  # pragma: no cover
        """Internal pure function executed by :meth:`run`.

        The ``x`` argument contains all model inputs as ``np.ndarray`` values.
        """
        ...

    def __init__(self, **kwargs):
        # make sure if developer overrides the input
        # definition as a method it still works...
        self._cm_input = (
            self._cm_input_def
            if isinstance(self._cm_input_def, dict)
            else self._cm_input_def()
        )
        self._set_input(**kwargs)

    def _set_input(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self._set_input_key(key, value)

    def _set_input_key(self, key: str, value) -> None:
        assert (
            key in self._cm_input.keys()
        ), f"Invalid input '{key}' key not found in the input definition."

        if isinstance(self._cm_input_def[key], Quant):
            units = self._cm_input[key].units
            try:
                quant = (
                    value.to(units) if isinstance(value, Quant) else Quant(value, units)
                )
            except pint.errors.DimensionalityError:
                raise ValueError(
                    f"Invalid unit for '{key}'; "
                    f"Expected [{units}] and got [{value.units}]."
                )
            self._cm_input[key] = quant
        else:
            self._cm_input[key] = value

    def run(self, **kwargs) -> CostModelOutput:
        """Run the cost model and return :class:`CostModelOutput`.

        Parameters
        ----------
        **kwargs:
            Values to update in the model input before execution. Each value can
            be a plain ``float`` or a :class:`~costmodels.units.Quant`.

        Returns
        -------
        CostModelOutput
            Output object containing ``capex`` and ``opex`` as ``np.ndarray``
            values.
        """

        self._set_input(**kwargs)
        cmo = self._run(_input_dict_to_magnitudes(self._cm_input))

        if anp.any(anp.isnan(cmo.capex)) or anp.any(anp.isnan(cmo.opex)):
            raise ValueError(
                f"NaNs detected in CostModelOutput (capex = {cmo.capex} or opex = {cmo.opex}). "
                f"Effective inputs (magnitudes) to _run method: {str(self._cm_input)}",
            )

        return cmo


class ICostModel(CostModel):
    """Minimal example cost model."""

    @property
    def _cm_input_def(self) -> dict:
        return {
            "a": Quant(2, "m"),
            "b": np.array([2.0, 3.0]),
            "dv": Quant(np.nan, "m"),
        }

    @staticmethod
    def _run(x: dict) -> CostModelOutput:
        capex = anp.abs(
            anp.sin(x["dv"] ** 2 / x["b"] + x["a"] * anp.cos(x["dv"])),
        )
        opex = anp.abs(
            anp.cos(x["dv"] ** 2 / x["a"] + x["b"] * anp.sin(x["dv"])),
        )
        return CostModelOutput(capex=anp.mean(capex), opex=anp.mean(opex))


if __name__ == "__main__":
    cm = ICostModel(a=Quant(3, "m"))
    x0 = {"dv": [12.0, 12.0], "b": np.array([2.0, 2.0])}

    def objective(x):
        out = cm.run(**x)
        return (out.opex + out.capex) ** 2

    value, gradv = value_and_grad(objective)(x0)
    print(f"Objective value: {value}")
    print(f"Gradient value: {gradv}")

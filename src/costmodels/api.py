from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from numbers import Number

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import pint

from costmodels.units import Quant


@tree_util.register_pytree_node_class
@dataclass
class CostModelOutput:
    capex: float
    opex: float

    def tree_flatten(self):
        # dynamic values propagated by jax AD
        children = (self.capex, self.opex)
        # no static data needed
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, _, children):
        # reconstruct the object from children
        return cls(*children)


def _input_dict_to_magnitudes(idict):
    """Convert all Quant values in the input dictionary to their magnitudes."""
    return {
        key: (
            jnp.array(value.magnitude, dtype=jnp.float32)
            if isinstance(value, Quant)
            else value
        )
        for key, value in idict.items()
    }


class CostModel(ABC):
    """Base class for all the cost models."""

    @property
    @abstractmethod
    def _cm_input_def(self) -> dict:  # pragma: no cover
        """Definition of cost model input."""
        ...

    @staticmethod
    @abstractmethod
    def _run(x: dict):  # pragma: no cover
        """Internal function to run the cost model."""
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

        if isinstance(self._cm_input[key], bool):
            assert isinstance(
                value, type(self._cm_input[key])
            ), f"Invalid type for '{key}', must be {type(self._cm_input[key])}"
            self._cm_input[key] = value
        elif isinstance(self._cm_input[key], Enum):
            if not isinstance(value, Enum):
                value = self._cm_input[key].__class__(value)
            self._cm_input[key] = value
        elif isinstance(self._cm_input[key], (Quant, Number, jnp.number)):
            is_quant_expected = isinstance(self._cm_input[key], Quant)
            if is_quant_expected:
                # the default value is a Quantity and
                # we try to assign units to provided number
                units = self._cm_input[key].units
                try:
                    quant = (
                        value.to(units)
                        if isinstance(value, Quant)
                        else Quant(value, units)
                    )
                except pint.errors.DimensionalityError:
                    raise ValueError(
                        f"Invalid unit for '{key}'; "
                        f"Expected [{units}] and got [{value.units}]."
                    )
            else:
                # keep a unitless number; input
                # specification does not expect a unit
                quant = value
            self._cm_input[key] = quant
        else:
            raise ValueError(
                f"Invalid type for '{key}'. Only numeric values, "
                f"pint.Quantity or Enum are allowed."
            )

    def run(self, **kwargs):
        self._set_input(**kwargs)
        cmo = self._run(_input_dict_to_magnitudes(self._cm_input))

        def _check_and_warn_for_nans_callback(capex_val, opex_val, inputs_val):
            import warnings  # fmt:skip

            import numpy as np  # fmt:skip

            if np.any(np.isnan(capex_val)) or np.any(np.isnan(opex_val)):
                warnings.warn(
                    f"NaNs detected in CostModelOutput (capex or opex). "
                    f"Effective inputs (magnitudes) to _run method: {str(inputs_val)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None  # Callbacks for side-effects should return None

        jax.debug.callback(
            _check_and_warn_for_nans_callback,
            cmo.capex,
            cmo.opex,
            _input_dict_to_magnitudes(self._cm_input),
        )

        return cmo

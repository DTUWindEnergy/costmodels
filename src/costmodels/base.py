import hashlib
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from numbers import Number

import numpy as np
from costmodels.utils import np2scalar

CAPEX_KEY = "capex"
OPEX_KEY = "opex"


class CostModel(ABC):
    """Base class for all the cost models."""

    def __init__(self, **kwargs):
        self._cm_input = (
            self._cm_input_def
            if isinstance(self._cm_input_def, dict)
            else self._cm_input_def()
        )

        self._cache_hash = None
        self._cache_res = None
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
                "value": value,
                "default": self._cm_input_def[key],
                "type": type(value),
            }
        return formatted_inputs

    def print_input(self) -> None:
        """Show all inputs of the cost model."""
        print(f"{self.__class__.__name__} inputs:")
        for k, v in self.list_input().items():
            print(f"  {k}")
            print(f"\tValue: {v['value']}")
            print(f"\tType: {v['type']}")

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
            elif isinstance(self._cm_input[key], (Number, np.number)):
                assert isinstance(
                    value, (Number, np.number)
                ), f"Invalid type for '{key}'"
                self._cm_input[key] = value
            else:
                self._cm_input[key] = value

    def run(self, **kwargs) -> dict:
        """The output of model is cached for the same input parameters."""
        if not kwargs and self._cache_res is not None:
            return self._cache_res

        self._set_input(**kwargs)
        input_hash = hashlib.md5(pickle.dumps(self._cm_input)).hexdigest()
        if self._cache_hash == input_hash:
            return self._cache_res

        self._cache_hash = input_hash
        self._cache_res = self._run()
        return self._cache_res

    @abstractmethod
    def _run(self) -> dict:  # pragma: no cover
        """Abstract method to run the cost model."""
        pass

    @property
    def _capex(self):
        """Get the CAPEX of the cost model."""
        return self.run()[CAPEX_KEY]

    @property
    def _opex(self):
        """Get the CAPEX of the cost model."""
        return self.run()[OPEX_KEY]

    # TODO: handle multidimensional inputs/outputs
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

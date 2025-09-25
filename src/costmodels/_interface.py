import dataclasses
from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np


def cost_input_dataclass(cls):
    # Collect fields and their types
    annotations = getattr(cls, "__annotations__", {})
    new_fields = []
    for name, annot_type in annotations.items():
        # Check if the field already has a default value
        value = getattr(cls, name, dataclasses.MISSING)

        if value is not dataclasses.MISSING and isinstance(
            value, (jnp.ndarray, np.ndarray, list)
        ):
            field = dataclasses.field(default_factory=lambda v=value: jnp.array(v))
            new_fields.append((name, annot_type, field))
            continue

        if value is not dataclasses.MISSING:
            new_fields.append((name, annot_type, dataclasses.field(default=value)))
            continue

        new_fields.append(
            (name, annot_type, dataclasses.field(default=dataclasses.MISSING))
        )

    # Create a new dataclass with updated defaults
    new_cls = dataclasses.make_dataclass(
        cls.__name__,
        new_fields,
        bases=cls.__bases__,
        namespace=dict(cls.__dict__),
    )
    return new_cls


@dataclass
class CostOutput:
    capex: float | jnp.floating  # MEUR
    opex: float | jnp.floating  # MEUR/year

    def __post_init__(self):
        self.capex = jnp.asarray(self.capex).squeeze()
        self.opex = jnp.asarray(self.opex).squeeze()


@cost_input_dataclass
class CostInput:
    """Base class for cost model inputs."""

    def __init__(self, **_):  # pragma: no cover
        raise NotImplementedError(
            f"{self.__class__.__name__} is an abstract base class. "
            "Please implement a concrete subclass with specific fields."
        )


class CostModel:
    # subclass must set this to a concrete dataclass
    _inputs_cls = CostInput

    # Initialize base (static) inputs with a dataclass of inputs
    def __init__(self, **kwargs):
        if not hasattr(self._inputs_cls, "__dataclass_fields__"):
            raise TypeError(f"{self._inputs_cls} must be a dataclass")
        self.base_inputs_dict = kwargs

    # Convenience: mutate only run time variables between calls
    def run(self, **runtime_overrides) -> Dict[str, Any]:
        try:
            inputs = self._inputs_cls(**{**self.base_inputs_dict, **runtime_overrides})
        except TypeError as e:
            raise TypeError(
                f"Error calling {self.__class__.__name__} with provided inputs. "
                f"Please check that all required fields are provided. {e}."
            ) from e

        output = self._run(inputs)
        if isinstance(output, dict):
            output = CostOutput(output["capex"], output["opex"])

        return output

    # Subclasses implement their internals here
    def _run(self, inputs: CostInput) -> Dict[str, Any]:  # pragma: no cover
        _ = inputs
        raise NotImplementedError

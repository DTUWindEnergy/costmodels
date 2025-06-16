from dataclasses import MISSING, dataclass, field, make_dataclass, replace
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np


def _cost_input_dataclass(cls):
    # Collect fields and their types
    annotations = getattr(cls, "__annotations__", {})
    new_fields = []
    for name, annot_type in annotations.items():
        # Check if the field already has a default value
        value = getattr(cls, name, MISSING)
        if value is not MISSING:
            new_fields.append((name, annot_type, field(default=value)))
            continue

        # If no default value, set based on type
        if annot_type is float:
            new_fields.append(
                (
                    name,
                    annot_type,
                    field(default_factory=lambda: jnp.nan),
                )
            )
        elif annot_type in (jnp.ndarray, np.ndarray):
            new_fields.append(
                (
                    name,
                    annot_type,
                    field(default_factory=lambda: jnp.array([jnp.nan])),
                )
            )

    # Create a new dataclass with updated defaults
    new_cls = make_dataclass(
        cls.__name__,
        new_fields,
        bases=cls.__bases__,
        namespace=dict(cls.__dict__),
    )
    return dataclass(new_cls)


@dataclass
class CostOutput:
    capex: float
    opex: float


@_cost_input_dataclass
class _CostInput:
    """This is not a base class, but simply a filler for _inputs_cls in CostModel.
    It is used to ensure that subclasses of CostModel have a concrete dataclass
    with specific fields. It should not be used anywhere else.
    """

    def __init__(self, **_):  # pragma: no cover
        raise NotImplementedError(
            f"{self.__class__.__name__} is an abstract base class. "
            "Please implement a concrete subclass with specific fields."
        )


class CostModel:
    # subclass must set this to a concrete dataclass
    _inputs_cls = _CostInput

    # Initialize base (static) inputs with a dataclass of inputs
    def __init__(self, **kwargs):
        if not hasattr(self._inputs_cls, "__dataclass_fields__"):
            raise TypeError(f"{self._inputs_cls} must be a dataclass")
        self.base_inputs = self._inputs_cls(**kwargs)

    # Convenience: mutate only run time variables between calls
    def run(self, **runtime_overrides) -> Dict[str, Any]:
        inputs = replace(self.base_inputs, **runtime_overrides)
        output = self._run(inputs)
        if isinstance(output, dict):
            return CostOutput(**output)
        return output

    # Subclasses implement their internals here
    def _run(self, inputs: _CostInput) -> Dict[str, Any]:  # pragma: no cover
        _ = inputs
        raise NotImplementedError

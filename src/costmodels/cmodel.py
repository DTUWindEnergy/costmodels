import abc
import dataclasses
from dataclasses import dataclass
from typing import Generic, Type, TypeVar

import jax.numpy as jnp
import numpy as np


@dataclass
class CostOutput:
    """Standard output from a cost model. Usually the values should be floats
    converting them to jnp.arrays size 1 array, for consistency for autograd engine."""

    capex: float | jnp.floating | jnp.ndarray  # MEUR
    opex: float | jnp.floating | jnp.ndarray  # MEUR/year

    def __post_init__(self):
        self.capex = jnp.asarray([self.capex]).squeeze()
        self.opex = jnp.asarray([self.opex]).squeeze()


@dataclass
class CostInput(abc.ABC):
    """Abstract base class for cost model inputs."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        annotations = getattr(cls, "__annotations__", {})
        for name in annotations:
            if hasattr(cls, name):
                value = getattr(cls, name)
                if isinstance(value, (np.ndarray, jnp.ndarray, list)):
                    setattr(
                        cls,
                        name,
                        dataclasses.field(default_factory=lambda v=value: jnp.array(v)),
                    )
        dataclasses.dataclass(cls)


def static_field(default):
    """Helper to create a static dataclass field."""
    return dataclasses.field(default=default, metadata={"static": True})


CostInputType = TypeVar("CostInputType", bound=CostInput)


class CostModel(Generic[CostInputType]):
    # subclass must set this to a concrete dataclass
    _inputs_cls: Type[CostInputType]

    # Initialize base (static) inputs with a dataclass of inputs
    def __init__(self, **kwargs):
        if not hasattr(self, "_inputs_cls") or not issubclass(
            self._inputs_cls, CostInput
        ):
            raise TypeError(
                "Cannot instantiate CostModel without CostInput. "
                "Please implement a subclass for cost inputs and assign "
                "it to _inputs_cls in the CostModel subclass. Example:\n"
                "class MyCostInput(CostInput):\n"
                "    param1: float\n"
                "    param2: float = 10.0\n\n"
                "class MyCostModel(CostModel):\n"
                "    _inputs_cls = MyCostInput\n"
            )
        self.base_inputs_dict = kwargs

    # Convenience: mutate only run time variables between calls
    def run(self, **runtime_overrides) -> CostOutput:
        for key in runtime_overrides:
            field_info = self._inputs_cls.__dataclass_fields__.get(key)
            if field_info and field_info.metadata.get("static"):
                raise ValueError(
                    f"Cannot override static field '{key}' in {self._inputs_cls.__name__}. "
                    "Static fields must be set at initialization of the CostModel."
                )

        try:
            inputs = self._inputs_cls(**{**self.base_inputs_dict, **runtime_overrides})
        except TypeError as e:
            raise TypeError(
                f"Error calling {self.__class__.__name__} with provided inputs. "
                f"Please check that all required fields are provided. {e}."
            ) from e

        if not isinstance(output := self._run(inputs), CostOutput):
            output = CostOutput(capex=output["capex"], opex=output["opex"])

        return output

    # Subclasses implement their internals here
    def _run(self, inputs: CostInputType) -> CostOutput:  # pragma: no cover
        _ = inputs
        raise NotImplementedError

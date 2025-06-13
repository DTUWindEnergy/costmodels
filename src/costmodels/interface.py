from dataclasses import dataclass, field, replace
from enum import Enum as enum
from typing import Any, Dict


class Mutability(enum):
    STATIC = "static"  # never changes once model is constructed
    DESIGN = "design_var"  # tuned by optimiser but fixed during each solve


@dataclass
class CostInputs:
    a: int = field(metadata={"tag": Mutability.DESIGN})
    b: float = field(default=0.07, metadata={"tag": Mutability.STATIC})

    def __init__(self, **_):
        raise NotImplementedError(
            "CostInputs is an abstract base class. "
            "Please implement a concrete subclass with specific fields."
        )


class BaseCostModel:
    # subclass must set this to a concrete dataclass
    _inputs_cls = CostInputs

    # Initialize base (static) inputs with a dataclass of inputs
    def __init__(self, **kwargs):
        if not hasattr(self._inputs_cls, "__dataclass_fields__"):
            raise TypeError(f"{self._inputs_cls} must be a dataclass")
        self.base_inputs = self._inputs_cls(**kwargs)

    # Convenience: mutate only run time variables between calls
    def run(self, **runtime_overrides) -> Dict[str, Any]:
        inputs = replace(self.base_inputs, **runtime_overrides)
        return self._run(inputs)

    # Subclasses implement their internals here
    def _run(self, inputs: CostInputs) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class ExampleCostModelInputs:
    a: float = 2.1
    b: int = 2
    flag: bool = True
    dv: float = 0.0


import jax
import jax.numpy as jnp


class ExampleCostModel(BaseCostModel):
    _inputs_cls = ExampleCostModelInputs

    def _run(self, inputs: ExampleCostModelInputs) -> Dict[str, Any]:
        if inputs.flag:
            capex = abs(
                jnp.sin(inputs.dv**2 / inputs.b + inputs.a * jnp.cos(inputs.dv))
            )
        else:
            capex = 0.0
        opex = abs(jnp.cos(inputs.dv**2 / inputs.a + inputs.b * jnp.sin(inputs.dv)))
        return {"capex": capex, "opex": opex}


def test_example_cost_model():
    cm = ExampleCostModel(a=2.1, b=2, flag=True, dv=0.0)
    out = cm.run(dv=1.0)
    assert isinstance(out, dict)
    assert "capex" in out
    assert "opex" in out
    assert out["capex"] >= 0
    assert out["opex"] >= 0

    val, grad = jax.value_and_grad(lambda x: cm.run(dv=x)["capex"])(1.0)
    print(f"Value: {val}, Gradient: {grad}")

    val, grad = jax.value_and_grad(lambda x: cm.run(dv=x, flag=False)["capex"])(1.0)
    print(f"Value: {val}, Gradient: {grad}")


if __name__ == "__main__":
    test_example_cost_model()
    print("Example cost model test passed.")

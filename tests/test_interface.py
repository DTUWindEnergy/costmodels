from enum import Enum
from typing import Any, Dict

import jax
import jax.numpy as jnp
import pytest

from costmodels.cmodel import CostModel, CostOutput, cost_input_dataclass


@cost_input_dataclass
class ExampleCostModelInputs:
    dv: float
    a: float = 2.1
    b: int = 2
    flag: bool = True


class ExampleCostModel(CostModel):
    _inputs_cls = ExampleCostModelInputs

    def _run(self, inputs: ExampleCostModelInputs) -> CostOutput:
        capex = (
            jnp.abs(jnp.sin(inputs.dv**2 / inputs.b + inputs.a * jnp.cos(inputs.dv)))
            if inputs.flag
            else 0.0
        )
        opex = jnp.abs(jnp.cos(inputs.dv**2 / inputs.a + inputs.b * jnp.sin(inputs.dv)))
        return CostOutput(capex=capex, opex=opex)


def test_example_cost_model():
    cm = ExampleCostModel(a=2.1, b=3.3, flag=True)
    out = cm.run(dv=1.0)
    assert isinstance(out, CostOutput)
    assert out.capex >= 0
    assert out.opex >= 0

    val, grad = jax.value_and_grad(lambda x: cm.run(dv=x).capex)(1.0)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad)

    val, grad = jax.value_and_grad(lambda x: cm.run(dv=x, flag=False).capex)(1.0)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad)

    # check that jit works on value_and_grad
    fn = jax.value_and_grad(lambda x: (cm.run(dv=x).capex + cm.run(dv=x).opex))
    jit_fn = jax.jit(fn)
    jit_val, jit_grad = jit_fn(3.0)
    assert jnp.isfinite(jit_val)
    assert jnp.isfinite(jit_grad)


def test_model_does_not_run_with_required_value_missing():
    cm = ExampleCostModel(a=2.1, b=3.3, flag=True)
    # missing required input 'dv'
    with pytest.raises(TypeError):
        cm.run()


def test_array_input_with_shape_one():
    cm = ExampleCostModel(a=2.1, b=3.3, flag=True, dv=jnp.array([1.0]))
    out = cm.run()
    # should always be scalar values
    assert jnp.isscalar(out.capex) and jnp.isscalar(out.opex)

    # value and grad should work with array inputs
    val, grad = jax.value_and_grad(lambda x: cm.run(dv=x).capex)(jnp.array([1.0]))
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad)


@cost_input_dataclass
class ArrayInputs:
    arr: jnp.ndarray = jnp.array([1.0, 2.0])


class ArrayModel(CostModel):
    _inputs_cls = ArrayInputs

    def _run(self, inputs: ArrayInputs) -> Dict[str, Any]:
        return {"capex": float(jnp.sum(inputs.arr)), "opex": 0.0}


def test_default_array_is_unique_per_instance():
    inp_a = ArrayInputs()
    inp_b = ArrayInputs()
    assert inp_a.arr is not inp_b.arr
    assert id(inp_a.arr) != id(inp_b.arr)

    model_a = ArrayModel()
    model_b = ArrayModel()
    assert model_a.run() != model_b.run(arr=jnp.array([2.0, 3.0]))


class DummyEnum(Enum):
    RED = 1
    BLUE = 2


@cost_input_dataclass
class EnumModelInputs:
    value: float = 1.0
    color: DummyEnum = DummyEnum.RED


class EnumModel(CostModel):
    _inputs_cls = EnumModelInputs

    def _run(self, inputs: EnumModelInputs) -> Dict[str, Any]:
        factor = 1.0 if inputs.color is DummyEnum.RED else 2.0
        return {"capex": inputs.value * factor, "opex": inputs.value}


def test_enum_field_default_used_without_overrides():
    cm = EnumModel()
    out = cm.run()
    assert isinstance(out, CostOutput)
    assert jnp.allclose(out.capex, 1.0)
    assert jnp.allclose(out.opex, 1.0)

    val, grad = jax.value_and_grad(lambda x: cm.run(value=x).capex)(1.0)
    assert jnp.allclose(val, 1.0)
    assert jnp.allclose(grad, 1.0)

    out_blue = cm.run(color=DummyEnum.BLUE, value=1.0)
    assert jnp.allclose(out_blue.capex, 2.0)
    assert jnp.allclose(out_blue.opex, 1.0)

    val_blue, grad_blue = jax.value_and_grad(
        lambda x: cm.run(color=DummyEnum.BLUE, value=x).capex
    )(1.0)
    assert jnp.allclose(val_blue, 2.0)
    assert jnp.allclose(grad_blue, 2.0)

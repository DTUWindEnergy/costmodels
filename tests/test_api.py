import jax
import jax.numpy as jnp
import pytest

from costmodels.api import CostModel, CostModelOutput
from costmodels.units import Quant


class ExampleCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {"a": Quant(2, "m"), "b": 2, "flag": True, "dv": Quant(0.0, "m")}

    @staticmethod
    def _run(x: dict) -> CostModelOutput:
        capex = jnp.abs(jnp.sin(x["dv"] ** 2 / x["b"] + x["a"] * jnp.cos(x["dv"])))
        opex = jnp.abs(jnp.cos(x["dv"] ** 2 / x["a"] + x["b"] * jnp.sin(x["dv"])))
        return CostModelOutput(capex=capex, opex=opex)


def test_run_returns_costmodeloutput():
    cm = ExampleCostModel()
    out = cm.run(dv=1.0)
    assert isinstance(out, CostModelOutput)
    assert out.capex.shape == ()
    assert out.opex.shape == ()


def test_invalid_unit_raises():
    cm = ExampleCostModel()
    with pytest.raises(ValueError):
        cm.run(dv=Quant(1.0, "s"))


def test_invalid_key_raises():
    cm = ExampleCostModel()
    with pytest.raises(AssertionError):
        cm.run(nonexistent=1)


def test_bool_validation():
    cm = ExampleCostModel()
    with pytest.raises(AssertionError):
        cm.run(flag=1)


def test_grad_through_model():
    cm = ExampleCostModel()
    grad_fn = jax.grad(lambda x: cm.run(dv=x).capex)
    val = grad_fn(1.0)
    assert jnp.isfinite(val)

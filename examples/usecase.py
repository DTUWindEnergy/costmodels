import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from costmodels.api import CostModel, CostModelOutput
from costmodels.finance import (
    LCO,
    Depreciation,
    Inflation,
    Product,
    Technology,
    finances,
)
from costmodels.units import Quant


class DummyCostModel(CostModel):
    """Minimal example cost model."""

    @property
    def _cm_input_def(self) -> dict:
        return {
            "a": Quant(2, "m"),
            "b": Quant(3, "EUR"),
            "dv": Quant(jnp.nan, "m"),
        }

    @staticmethod
    def _run(x: dict) -> CostModelOutput:
        capex = jnp.abs(jnp.sin(x["dv"] ** 2 / x["b"] + x["a"] * jnp.cos(x["dv"])))
        opex = jnp.abs(jnp.cos(x["dv"] ** 2 / x["a"] + x["b"] * jnp.sin(x["dv"])))
        return CostModelOutput(capex=1e6 * capex, opex=1e3 * opex)


if __name__ == "__main__":
    cm_wind = DummyCostModel(a=Quant(3, "m"))
    cm_solar = DummyCostModel(a=Quant(4, "m"))

    p_wind = np.random.rand(8760) * 1000  # Example wind production
    p_solar = np.random.rand(8760) * 1000  # Example solar production
    p_wind_non = np.zeros_like(p_wind)  # Non-revenue production for wind
    p_solar_non = np.zeros_like(p_solar)  # Non-revenue production for solar

    wind_plant = Technology(
        name="wind_plant",
        CAPEX=0,
        OPEX=0,
        lifetime=25,
        t0=0,
        WACC=0.06,
        phasing_yr=[-1, 0],
        phasing_capex=[1, 1],
        production=np.tile(p_wind, 25),
        non_revenue_production=np.tile(p_wind_non, 25),
        product=Product.SPOT_ELECTRICITY,
    )
    solar_plant = Technology(
        name="solar_plant",
        CAPEX=0,
        OPEX=0,
        lifetime=25,
        t0=0,
        WACC=0.06,
        phasing_yr=[-1, 0],
        phasing_capex=[1, 1],
        production=np.tile(p_solar, 25),
        non_revenue_production=np.tile(p_solar_non, 25),
        product=Product.SPOT_ELECTRICITY,
    )

    spot_price = np.random.uniform(30, 70, size=8760)
    product_prices = {Product.SPOT_ELECTRICITY: np.tile(spot_price, 25)}
    inflation_yr = [-3, 0, 1, 25]
    inflation = [0.10, 0.10, 0.06, 0.06]
    inflation_yr_ref = 0  # inflation index is computed with respect to this year
    inflation = Inflation(inflation, inflation_yr, inflation_yr_ref)
    depreciation_yr = [0, 25]
    depreciation = [0, 1]
    depreciation = Depreciation(depreciation_yr, depreciation)
    tax_rate = 0.22
    DEVEX = 0

    def objective(x0, x1):
        wind_cmo = cm_wind.run(**x0)
        solar_cmo = cm_solar.run(**x1)
        wind_plant.CAPEX = wind_cmo.capex
        wind_plant.OPEX = wind_cmo.opex
        solar_plant.CAPEX = solar_cmo.capex
        solar_plant.OPEX = solar_cmo.opex
        return finances(
            technologies=[wind_plant, solar_plant],
            product_prices=product_prices,
            shared_capex=0.0,
            inflation=inflation,
            tax_rate=tax_rate,
            depreciation=depreciation,
            devex=DEVEX,
        )["NPV"]

    x0 = {"dv": 3.0}
    x1 = {"dv": 2.0}

    value, grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(x0, x1)
    print(f"Objective value: {value}")
    print(f"Grad: {grad}")

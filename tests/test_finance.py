import os
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from costmodels.finance import (
    LCO,
    Depreciation,
    Inflation,
    Product,
    Technology,
    _inflation_index,
    finances,
)

################ UNIT TESTS #######################


@pytest.mark.parametrize(
    "years, inflation, expected",
    [
        # === Test Case 1: Constant inflation rate ===
        # Simple case with ref_year = 0
        (
            np.arange(5),
            Inflation(rate=0.05, year_ref=0),
            (1.05) ** np.arange(5),
        ),
        # Constant rate with a different ref_year
        (
            np.arange(5),
            Inflation(rate=0.02, year_ref=2),
            (1.02) ** (np.arange(5) - 2),
        ),
        # === Test Case 2: Variable inflation rate ===
        # Years match the inflation data points exactly
        (
            np.array([0, 1, 2, 3]),
            Inflation(year=(0, 1, 2, 3), rate=(0.02, 0.03, 0.04, 0.05), year_ref=0),
            np.array([1.0, 1.03, 1.03 * 1.04, 1.03 * 1.04 * 1.05]),
        ),
        # Years require linear interpolation
        (
            np.arange(5),  # years = [0, 1, 2, 3, 4]
            Inflation(year=(0, 2, 4), rate=(0.02, 0.04, 0.06), year_ref=0),
            # Interpolated infl rates: [0.02, 0.03, 0.04, 0.05, 0.06]
            np.array([1.0, 1.03, 1.0712, 1.12476, 1.1922456]),
        ),
        # Years require clipping (extrapolation)
        (
            np.arange(6),  # years = [0, 1, 2, 3, 4, 5]
            Inflation(year=(2, 4), rate=(0.03, 0.05), year_ref=2),
            # Clipped/interpolated infl rates: [0.03, 0.03, 0.03, 0.04, 0.05, 0.05]
            np.array([1.03**-2, 1.03**-1, 1.0, 1.04, 1.04 * 1.05, 1.04 * 1.05**2]),
        ),
        # === Test Case 3: Reference year handling ===
        # ref_year is not in years array, should default to index 0 for normalization
        (
            np.arange(5),
            Inflation(rate=0.1, year_ref=10),
            (1.1) ** np.arange(5),  # Same as if ref_year=0
        ),
        # === Test Case 4: Scalar input for years ===
        # Scalar input that is the ref_year
        (
            0,
            Inflation(rate=0.05, year_ref=0),
            np.array(1.0),
        ),
        # Scalar input that is not the ref_year
        (
            3,
            Inflation(rate=0.05, year_ref=0),
            np.array(1.0),
        ),
        # No reference provided
        (
            np.arange(5),
            Inflation(rate=0.05),
            (1.05) ** np.arange(5),
        ),
        # Scalar input with variable inflation
        (
            3,
            Inflation(year=(0, 5), rate=(0.02, 0.08), year_ref=0),
            np.array(1.0),
        ),
    ],
    ids=[
        "constant_rate_ref_zero",
        "constant_rate_ref_middle",
        "variable_rate_exact_match",
        "variable_rate_interpolation",
        "variable_rate_clipping",
        "ref_year_not_in_years",
        "scalar_year_is_ref",
        "scalar_year_not_ref_constant_rate",
        "no_reference",
        "scalar_year_not_ref_variable_rate",
    ],
)
def test_inflation_index_scenarios(years, inflation, expected):
    """Test _inflation_index with various scenarios."""
    result = _inflation_index(years, inflation)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


################ INTEGRATION TESTS ################


def test_finances_run_against_reference_from_hydesign_0():
    CAPEX_wind = 4.17179442e08
    CAPEX_solar = 67000000
    CAPEX_bess = 13850000
    CAPEX_p2x = 6.415e08
    CAPEX_shared = 62982000
    OPEX_wind = 8626811.29405334
    OPEX_solar = 1350000
    OPEX_bess = 0
    OPEX_p2x = 14224811.77489952

    ts_inputs = pd.read_csv(
        Path(os.path.dirname(__file__)) / Path("../tests/data/finance_inputs_p2x.csv"),
        index_col=0,
        sep=";",
    )

    plt.plot(
        ts_inputs.hpp_t + ts_inputs.hpp_curt_t + ts_inputs.P_ptg_t - ts_inputs.b_t,
        ts_inputs.wind_t_ext + ts_inputs.solar_t_ext,
        ".",
    )

    p_wind = (
        ts_inputs.hpp_t
        / (ts_inputs.wind_t_ext + ts_inputs.solar_t_ext)
        * ts_inputs.wind_t_ext
    )
    p_solar = (
        ts_inputs.hpp_t
        / (ts_inputs.wind_t_ext + ts_inputs.solar_t_ext)
        * ts_inputs.solar_t_ext
    )
    p_wind = np.nan_to_num(p_wind)
    p_solar = np.nan_to_num(p_solar)
    p_bess = ts_inputs.hpp_t - p_wind - p_solar

    p_wind_non = (
        ts_inputs.P_ptg_t
        / (ts_inputs.wind_t_ext + ts_inputs.solar_t_ext)
        * ts_inputs.wind_t_ext
    )
    p_solar_non = (
        ts_inputs.P_ptg_t
        / (ts_inputs.wind_t_ext + ts_inputs.solar_t_ext)
        * ts_inputs.solar_t_ext
    )
    p_wind_non = np.nan_to_num(p_wind_non)
    p_solar_non = np.nan_to_num(p_solar_non)
    p_bess_non = ts_inputs.P_ptg_t - p_wind_non - p_solar_non

    technologies = {
        "wind": {
            "CAPEX": CAPEX_wind,
            "OPEX": OPEX_wind,
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.06,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
            "product": Product.SPOT_ELECTRICITY,
            "production": np.tile(p_wind, 25),
            "non_revenue_production": np.tile(p_wind_non, 25),
        },
        "solar": {
            "CAPEX": CAPEX_solar,
            "OPEX": OPEX_solar,
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.06,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
            "product": Product.SPOT_ELECTRICITY,
            "production": np.tile(p_solar, 25),
            "non_revenue_production": np.tile(p_solar_non, 25),
        },
        "bess": {
            "CAPEX": CAPEX_bess,
            "OPEX": OPEX_bess,
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.06,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
            "product": Product.SPOT_ELECTRICITY,
            "production": np.tile(p_bess, 25),
            "non_revenue_production": np.tile(p_bess_non, 25),
        },
        "p2x": {
            "CAPEX": CAPEX_p2x,
            "OPEX": OPEX_p2x,
            "consumption": sum(ts_inputs.P_ptg_t * ts_inputs.price_t_ext),
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.08,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
            "product": Product.HYDROGEN,
            "production": np.tile(ts_inputs.m_H2_t, 25),
            "non_revenue_production": 0 * np.tile(p_bess_non, 25),
        },
    }

    technologies = [
        Technology(
            name=k,
            capex=v["CAPEX"],
            opex=v["OPEX"],
            lifetime=v["lifetime"],
            t0=v["t0"],
            wacc=v["WACC"],
            phasing_yr=v["phasing_yr"],
            phasing_capex=v["phasing_capex"],
            production=v["production"],
            non_revenue_production=v["non_revenue_production"],
            product=v["product"],
            consumption=v.get("consumption", 0),
        )
        for k, v in technologies.items()
    ]

    product_prices = {
        Product.SPOT_ELECTRICITY: np.tile(ts_inputs["price_t_ext"], 25),
        Product.HYDROGEN: 5 * np.ones(25 * 8760),
    }

    lcos = (  # levelized definitions
        LCO("LCOE", ["wind", "solar", "bess"], True),
        LCO("LCOH", ["p2x"], False),
    )

    # Inflation will be linearly interpolated at integer year values
    inflation_yr = [-3, 0, 1, 25]
    inflation = [0.10, 0.10, 0.06, 0.06]
    inflation_yr_ref = 0  # inflation index is computed with respect to this year
    inflation = Inflation(inflation, inflation_yr, inflation_yr_ref)

    # depreciation
    depreciation_yr = [0, 25]
    depreciation = [0, 1]

    depreciation = Depreciation(depreciation_yr, depreciation)

    tax_rate = 0.22
    DEVEX = 0
    shared_capex = CAPEX_shared

    res = finances(
        technologies,
        product_prices,
        shared_capex,
        inflation,
        tax_rate,
        depreciation,
        DEVEX,
        lcos,
    )
    ref = {
        "cashflow": np.array(
            [
                -1.18662256e09,
                9.75998111e07,
                1.02829263e08,
                1.08372482e08,
                1.14248294e08,
                1.20476655e08,
                1.27078718e08,
                1.34076904e08,
                1.41494982e08,
                1.49358144e08,
                1.57693096e08,
                1.66528145e08,
                1.75893297e08,
                1.85820358e08,
                1.96343043e08,
                2.07497089e08,
                2.19320377e08,
                2.31853063e08,
                2.45137710e08,
                2.59219436e08,
                2.74146066e08,
                2.89968293e08,
                3.06739854e08,
                3.24517708e08,
                3.43362234e08,
                3.63337431e08,
            ]
        ),
        "NPV": 7.37912942e08,
        "IRR": 11.82257 / 100,
        "CAPEX": np.float64(1186622556.690909),
        "OPEX": np.array(
            [
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
                24201623.06895286,
            ]
        ),
        "break_even_prices": {
            "spot_electricity": -890.1452745578449,
            "hydrogen": 3.290614937396693,
        },
        "LCOE": np.float64(32.88913551461165),
        "LCOH": np.float64(5.196622705493022),
    }

    np.testing.assert_allclose(res["cashflow"], ref["cashflow"])
    np.testing.assert_allclose(res["NPV"], ref["NPV"])
    np.testing.assert_allclose(res["IRR"], ref["IRR"])
    np.testing.assert_allclose(res["CAPEX"], ref["CAPEX"])
    np.testing.assert_allclose(res["OPEX"], ref["OPEX"])
    np.testing.assert_allclose(res["LCOE"], ref["LCOE"])
    np.testing.assert_allclose(res["LCOH"], ref["LCOH"])

    # TODO: test break even prices once it's implemented
    # np.testing.assert_allclose(res["break_even_prices"]["spot_electricity"], ref["break_even_prices"]["spot_electricity"], rtol=1e-5)
    # np.testing.assert_allclose(res["break_even_prices"]["hydrogen"], ref["break_even_prices"]["hydrogen"], rtol=1e-5)


def test_finances_against_reference_from_hydesign_1():
    # import os
    # import timeit
    # from pathlib import Path
    # import numpy
    # import pandas as pd

    CAPEX_wind = 2.41170504e08
    CAPEX_solar = 66125000
    CAPEX_bess = 9882866.10284274
    CAPEX_shared = 61122845.07042254
    OPEX_wind = 4262488.80495959
    OPEX_solar = 1331250
    OPEX_bess = 0
    ts_inputs = pd.read_csv(
        Path(os.path.dirname(__file__)) / Path("./data/finance_inputs.csv"),
        index_col=0,
        sep=";",
    )
    technologies = {
        "wind": {
            "CAPEX": CAPEX_wind,
            "OPEX": OPEX_wind,
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.06,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
        },
        "solar": {
            "CAPEX": CAPEX_solar,
            "OPEX": OPEX_solar,
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.06,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
        },
        "batt": {
            "CAPEX": CAPEX_bess,
            "OPEX": OPEX_bess,
            "lifetime": 25,
            "t0": 0,
            "WACC": 0.06,
            "phasing_yr": [-1, 0],
            "phasing_capex": [
                1,
                1,
            ],
        },
    }
    product_prices = {Product.SPOT_ELECTRICITY: np.asarray(ts_inputs["price_t"])}

    production_sample = ts_inputs.get("p_wind", None)
    zeros = np.zeros_like(production_sample)

    # Originally the solar has battery production added here...
    # {
    #     "name": "solar_power",
    #     "technology": "solar",
    #     "production": ts_inputs["p_solar"] + ts_inputs["p_batt"],
    #     "product": "spot_electricity",
    # },
    technologies = [
        Technology(
            name=k,
            capex=v["CAPEX"],
            opex=v["OPEX"],
            lifetime=v["lifetime"],
            t0=v["t0"],
            wacc=v["WACC"],
            phasing_yr=v["phasing_yr"],
            phasing_capex=v["phasing_capex"],
            production=np.asarray(ts_inputs.get(f"p_{k}", zeros)),
            non_revenue_production=np.zeros(len(ts_inputs.get(f"p_{k}", zeros))),
            product=Product.SPOT_ELECTRICITY,
        )
        for k, v in technologies.items()
    ]

    inflation = Inflation(
        year=[-3, 0, 1, 25],
        rate=[0.10, 0.10, 0.06, 0.06],
        year_ref=0,  # inflation index is computed with respect to this year
    )

    depreciation = Depreciation(
        year=[0, 25],
        rate=[0, 1],
    )

    tax_rate = 0.22
    DEVEX = 0
    shared_capex = CAPEX_shared

    res = finances(
        technologies,
        product_prices,
        shared_capex,
        inflation,
        tax_rate,
        depreciation,
        DEVEX,
    )

    ref = {
        "cashflow": np.array(
            [
                -3.71423011e08,
                2.24513680e07,
                2.33846642e07,
                2.43226626e07,
                2.53007504e07,
                2.63201665e07,
                2.73823554e07,
                2.84887682e07,
                2.96408093e07,
                3.08397247e07,
                3.20871903e07,
                3.33846531e07,
                3.47335186e07,
                3.61353925e07,
                3.75917414e07,
                3.91034056e07,
                4.06717494e07,
                4.22984174e07,
                4.39854907e07,
                4.57343056e07,
                4.75457839e07,
                4.94213055e07,
                5.13608503e07,
                5.33661798e07,
                5.54385615e07,
                5.76183758e07,
            ]
        ),
        "NPV": 55399626.7,
        "IRR": 7.26525611 / 100,
        "CAPEX": 371423011.2610241,
        "OPEX": np.array(
            [
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
                5593738.80495959,
            ]
        ),
        "break_even_prices": {"spot_electricity": 29.123712413268382},
        "LCOE": 43.39325844791887,
    }

    np.testing.assert_allclose(res["cashflow"], ref["cashflow"])
    np.testing.assert_allclose(res["NPV"], ref["NPV"])
    np.testing.assert_allclose(res["IRR"], ref["IRR"])
    np.testing.assert_allclose(res["CAPEX"], ref["CAPEX"])
    np.testing.assert_allclose(res["OPEX"], ref["OPEX"])
    np.testing.assert_allclose(res["LCOE"], ref["LCOE"])

    def grad_func(production):
        """Function to compute IRR gradient."""
        technologies[0].production = production
        technologies[1].production = production
        technologies[2].production = production
        return finances(
            technologies,
            product_prices,
            shared_capex,
            inflation,
            tax_rate,
            depreciation,
            DEVEX,
        )["IRR"]

    production_sample = np.asarray(production_sample)
    val, grad = jax.value_and_grad(grad_func)(production_sample)

    assert np.isfinite(val), "IRR value is not finite"
    assert np.all(np.isfinite(grad)), "IRR gradient contains non-finite values"

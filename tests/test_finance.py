import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from costmodels.finance import (
    LCO,
    Depreciation,
    Inflation,
    Product,
    Technology,
    finances,
)


def test_finances_run_against_reference_from_hydesign():
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
    np.nan_to_num(p_wind)
    np.nan_to_num(p_solar)
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
    np.nan_to_num(p_wind_non)
    np.nan_to_num(p_solar_non)
    p_bess_non = ts_inputs.P_ptg_t - p_wind_non - p_solar_non

    # technologies:
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
            CAPEX=v["CAPEX"],
            OPEX=v["OPEX"],
            lifetime=v["lifetime"],
            t0=v["t0"],
            WACC=v["WACC"],
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
    for ref_key, ref_value in ref.items():
        if isinstance(ref_value, dict):
            continue  # skip break_even_prices
            # rsv, rfv = rsv["spot_electricity"], rfv["spot_electricity"]
        np.testing.assert_allclose(res[ref_key], ref_value)

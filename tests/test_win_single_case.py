import os
import platform
from pathlib import Path

import pytest

from costmodels import DTUOffshoreCM as DTUOCM

from .utils.DTU_CostModel_org import DTUOffshoreCostModel
from .utils.winutil import (
    dtu_offshore_cm_input_map,
    dtu_offshore_cm_output_map,
    run_excel,
)


@pytest.mark.skipif(platform.system() != "Windows", reason="Only run on Windows")
def test_win_single_case():
    params = {
        "rated_power": 3.111111111111111,
        "rotor_diameter": 0.060314403509210746,
        "rotor_speed": 9.444444444444445,
        "hub_height": 20.111486515663536,
        "profit": 0.01,
        "capacity_factor": 0.3333333333333333,
        "decline_factor": -0.02,
        "nwt": 290,
        "project_lifetime": 27,
        "wacc": 0.07222222222222223,
        "inflation": 0.08,
        "opex": 30.0,
        "devex": 11.11111111111111,
        "abex": 5.555555555555555,
        "water_depth": 33.33333333333333,
        "electrical_cost": 0.0,
        "foundation_option": 0,
        "eprice": 0.2,
    }

    cm = DTUOCM()

    # fractions need to be converted to percentages
    adaptedp = params.copy()
    # decline factor is negated inside the model
    for key in ["decline_factor", "profit", "capacity_factor", "wacc", "inflation"]:
        adaptedp[key] *= -100 if key == "decline_factor" else 100
    adaptedp["lifetime"] = adaptedp.pop("project_lifetime")
    cmi = DTUOCM.Input(**adaptedp)

    # run the model
    results: DTUOCM.Output = cm.run(cmi)
    # run model with original excel
    input_map = dtu_offshore_cm_input_map(**params)
    output_map = dtu_offshore_cm_output_map()
    excel_file = Path(os.path.dirname(__file__), "data/WTcostmodel_v12.xlsx")
    assert excel_file.exists()
    excel_result = run_excel(
        file_path=excel_file,
        input_map=input_map,
        output_map=output_map,
    )

    assert excel_result["OPEX net (EURO)"] == results.opex.to("MEUR").m


@pytest.mark.skipif(platform.system() != "Windows", reason="Only run on Windows")
def test_original_dtu_cm_implementation_win_excel():

    params = {
        "rated_power": 3.111111111111111,
        "rotor_diameter": 0.060314403509210746,
        "rotor_speed": 9.444444444444445,
        "hub_height": 20.111486515663536,
        "profit": 0.01,
        "capacity_factor": 0.3333333333333333,
        "decline_factor": -0.02,
        "nwt": 290,
        "project_lifetime": 27,
        "wacc": 0.07222222222222223,
        "inflation": 0.08,
        "opex": 30.0,
        "devex": 11.11111111111111,
        "abex": 5.555555555555555,
        "water_depth": 33.33333333333333,
        "electrical_cost": 0.0,
        "foundation_option": 0,
    }

    cm = DTUOffshoreCostModel(**params)
    results = cm.run()
    results["CO2 emission (kg CO2 eq))"] = results[
        "Total Co2 emission per turbine (kg CO2 eq)"
    ].mean()

    input_map = dtu_offshore_cm_input_map(**params)
    output_map = dtu_offshore_cm_output_map()
    excel_file = Path(os.path.dirname(__file__), "data/WTcostmodel_v12.xlsx")
    assert excel_file.exists()
    res = run_excel(file_path=excel_file, input_map=input_map, output_map=output_map)
    assert results["OPEX net (EURO)"] == res["OPEX net (EURO)"]

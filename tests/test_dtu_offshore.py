import warnings

import numpy as np

from costmodels.models import DTUOffshoreCostModel

from .utils.DTU_CostModel_org import DTUOffshoreCostModel as FafasDTUOffshoreCostModel


def test_monte_carlo_agains_original_dtu_offshore_implementation():
    n_samples = 100
    num = 10
    sp = 350
    clearance = 20
    sample_parameters = dict(
        rated_power=np.linspace(1, 20, num),
        rotor_diameter=np.sqrt(4 * np.linspace(1, 20, num) * 10**6 / (np.pi * sp)),
        rotor_speed=np.linspace(5, 10, num),
        hub_height=clearance
        + np.sqrt(4 * np.linspace(1, 20, num) * 10**6 / (np.pi * sp)) / 2,
        profit=np.linspace(0.01, 0.1, num),
        capacity_factor=np.linspace(0.3, 0.6, num),
        decline_factor=np.linspace(0.02, 0.01, num),
        nwt=np.arange(10, 400, 40),
        project_lifetime=np.arange(15, 25),
        wacc=np.linspace(0.05, 0.1, num),
        inflation=np.linspace(0.01, 0.1, num),
        opex=np.linspace(10, 40, num),
        devex=np.linspace(0, 20, num),
        abex=np.linspace(0, 10, num),
        water_depth=np.linspace(0, 100, num),
        electrical_cost=np.linspace(0, 10, num),
        foundation_option=np.arange(5),
    )

    shape = 16 * (10,) + (5,)
    size = 10**16 * 5

    np.random.seed(42)
    sample_nos = np.random.uniform(high=size, size=n_samples)
    parameter_idxs = [
        np.unravel_index(int(sample_no), shape) for sample_no in sample_nos
    ]

    failed = False
    for parameter_idx in parameter_idxs:
        params = {
            key: sample_parameters[key][idx]
            for key, idx in zip(sample_parameters.keys(), parameter_idx)
        }
        params["AEP"] = np.array([1.0])

        # Run original DTU model implementation
        origcm = FafasDTUOffshoreCostModel(**params)
        results = origcm.run()
        results["CO2 emission (kg CO2 eq))"] = results[
            "Total Co2 emission per turbine (kg CO2 eq)"
        ].mean()
        results.pop("Total Co2 emission per turbine (kg CO2 eq)")
        results.pop("Turbine cost (EURO)")

        # Adapt parameters for the concise implementation
        adapted_params = params.copy()
        adapted_params.pop("AEP")
        adapted_params["lifetime"] = adapted_params.pop("project_lifetime")
        if "eprice" not in adapted_params:
            adapted_params["eprice"] = 0.2
        adapted_params["aep"] = np.array([1.0])

        cm = DTUOffshoreCostModel(**adapted_params)
        cmo = cm.run()
        results_our = cm._details

        new_results_mapped = {
            "AEP net (MWh)": results_our["aep_net"],
            "AEP discount (MWh)": results_our["aep_discount"],
            "DEVEX net (EURO)": results_our["devex_net"] * 1e6,
            "DEVEX discount (EURO)": results_our["devex_discount"] * 1e6,
            "CAPEX net (EURO)": cmo.capex * 1e6,
            "CAPEX discount (EURO)": results_our["capex_discount"] * 1e6,
            "OPEX net (EURO)": cmo.opex * 1e6 * cm.lifetime,
            "OPEX discount (EURO)": results_our["opex_discount"] * 1e6,
            "LCOE (EURO/MWh)": results_our["lcoe"],
        }

        for k, v in new_results_mapped.items():
            if k in results:
                if abs(v - results[k]) / max(abs(v), abs(results[k]), 1e-6) >= 1e-5:
                    warnings.warn(
                        f"Mismatch in {k}: {v} vs {results[k]}; Parameters: {params}"
                    )
                    failed = True

    assert not failed


def test_dtu_offshore_gradients():
    # Test the gradients of the DTU Offshore Cost Model
    params = {
        "rated_power": 3.111111111111111,
        "rotor_diameter": 80,
        "rotor_speed": 9.444444444444445,
        "hub_height": 20.111486515663536,
        "profit": 0.01,
        "capacity_factor": 0.3333333333333333,
        "decline_factor": -0.02,
        "nwt": 290,
        "project_lifetime": 25,
        "wacc": 0.07222222222222223,
        "inflation": 0.08,
        "opex": 30.0,
        "devex": 11.11111111111111,
        "abex": 5.555555555555555,
        "water_depth": 33.33333333333333,
        "electrical_cost": 0.0,
        "foundation_option": 1,
        "eprice": 0.2,
        "aep": np.array([1.0]),
    }

    adapted_params = params.copy()
    adapted_params["lifetime"] = adapted_params.pop("project_lifetime")

    cm = DTUOffshoreCostModel(**adapted_params)

    def func(x):
        res = cm.run(water_depth=x)
        return res.capex

    import jax

    value, grad = jax.value_and_grad(func)(params["water_depth"])
    assert value is not None
    assert grad is not None
    assert jax.numpy.isfinite(value)
    assert jax.numpy.isfinite(grad)
    assert grad != 0, "Gradient should not be zero for non-trivial input."


# THE TESTS BELOW ARE FOR THE EXCEL IMPLEMENTATION
# BUT THE WINDOWS DOCKER IMAGE DOES NOT SUPPORT OPENNING
# EXCEL SHEET IN THERE !!! UNCOMMENT FOR LOCAL TESTING

# import os
# import platform
# import warnings
# from pathlib import Path

# import numpy as np
# import pytest

# from costmodels import DTUOffshoreCostModel as DTUOCM

# from .utils.winutil import (
#     dtu_offshore_cm_input_map,
#     dtu_offshore_cm_output_map,
#     run_excel,
# )


# @pytest.mark.skipif(platform.system() != "Windows", reason="Only run on Windows")
# def test_win_single_case():
#     params = {
#         "rated_power": 3.111111111111111,
#         "rotor_diameter": 80,
#         "rotor_speed": 9.444444444444445,
#         "hub_height": 20.111486515663536,
#         "profit": 0.01,
#         "capacity_factor": 0.3333333333333333,
#         "decline_factor": -0.02,
#         "nwt": 290,
#         "project_lifetime": 25,
#         "wacc": 0.07222222222222223,
#         "inflation": 0.08,
#         "opex": 30.0,
#         "devex": 11.11111111111111,
#         "abex": 5.555555555555555,
#         "water_depth": 33.33333333333333,
#         "electrical_cost": 0.0,
#         "foundation_option": 0,
#         "eprice": 0.2,
#     }

#     # fractions need to be converted to percentages
#     adaptedp = params.copy()
#     # decline factor is negated inside the model
#     for key in ["decline_factor", "profit", "capacity_factor", "wacc", "inflation"]:
#         adaptedp[key] *= -100 if key == "decline_factor" else 100
#     adaptedp["lifetime"] = adaptedp.pop("project_lifetime")

#     cm = DTUOffshoreCostModel(**adaptedp)

#     # run the model
#     results = cm.run()
#     # run model with original excel
#     input_map = dtu_offshore_cm_input_map(**params)
#     output_map = dtu_offshore_cm_output_map()
#     excel_file = Path(os.path.dirname(__file__), "data/WTcostmodel_v12.xlsx")
#     assert excel_file.exists()
#     excel_result = run_excel(
#         file_path=excel_file,
#         input_map=input_map,
#         output_map=output_map,
#     )
#     np.testing.assert_allclose(
#         excel_result["OPEX net (EURO)"], results["opex"].to("EUR").m
#     )


# @pytest.mark.skipif(platform.system() != "Windows", reason="Only run on Windows")
# def test_original_dtu_cm_implementation_win_excel():
#     params = {
#         "rated_power": 3.111111111111111,
#         "rotor_diameter": 80,
#         "rotor_speed": 9.444444444444445,
#         "hub_height": 20.111486515663536,
#         "profit": 0.01,
#         "capacity_factor": 0.3333333333333333,
#         "decline_factor": -0.02,
#         "nwt": 290,
#         "project_lifetime": 25,
#         "wacc": 0.07222222222222223,
#         "inflation": 0.08,
#         "opex": 30.0,
#         "devex": 11.11111111111111,
#         "abex": 5.555555555555555,
#         "water_depth": 33.33333333333333,
#         "electrical_cost": 0.0,
#         "foundation_option": 0,
#     }

#     cm = FafasDTUOffshoreCostModel(**params)
#     results = cm.run()
#     results["CO2 emission (kg CO2 eq))"] = results[
#         "Total Co2 emission per turbine (kg CO2 eq)"
#     ].mean()

#     input_map = dtu_offshore_cm_input_map(**params)
#     output_map = dtu_offshore_cm_output_map()
#     excel_file = Path(os.path.dirname(__file__), "data/WTcostmodel_v12.xlsx")
#     assert excel_file.exists()
#     res = run_excel(file_path=excel_file, input_map=input_map, output_map=output_map)
#     np.testing.assert_allclose(results["OPEX net (EURO)"], res["OPEX net (EURO)"])


# Works , no need for testing it all the time !!!
# Works , no need for testing it all the time !!!
# Works , no need for testing it all the time !!!

# @pytest.mark.skipif(platform.system() != "Windows", reason="Only runs on Windows")
# def test_monte_carlo_excel_comparison():
#     n_samples = 100
#     num = 10
#     sp = 350
#     clearance = 20
#     sample_parameters = dict(
#         rated_power=np.linspace(1, 20, num),
#         rotor_diameter=np.sqrt(4 * np.linspace(1, 20, num) * 10**6 / (np.pi * sp)),
#         rotor_speed=np.linspace(5, 10, num),
#         hub_height=clearance
#         + np.sqrt(4 * np.linspace(1, 20, num) * 10**6 / (np.pi * sp)) / 2,
#         profit=np.linspace(0.01, 0.1, num),
#         capacity_factor=np.linspace(0.3, 0.6, num),
#         decline_factor=np.linspace(-0.02, -0.01, num),
#         nwt=np.arange(10, 400, 40),
#         project_lifetime=np.arange(15, 25),
#         wacc=np.linspace(0.05, 0.1, num),
#         inflation=np.linspace(0.01, 0.1, num),
#         opex=np.linspace(10, 40, num),
#         devex=np.linspace(0, 20, num),
#         abex=np.linspace(0, 10, num),
#         water_depth=np.linspace(0, 100, num),
#         electrical_cost=np.linspace(0, 10, num),
#         foundation_option=np.arange(5),
#     )

#     shape = 16 * (10,) + (5,)
#     size = 10**16 * 5

#     np.random.seed(42)
#     sample_nos = np.random.uniform(high=size, size=n_samples)
#     parameter_idxs = [
#         np.unravel_index(int(sample_no), shape) for sample_no in sample_nos
#     ]

#     excel_file = Path(os.path.dirname(__file__), "data/WTcostmodel_v12.xlsx")
#     assert excel_file.exists()

#     metrics = [
#         "AEP net (MWh)",
#         "AEP discount (MWh)",
#         "DEVEX net (EURO)",
#         "DEVEX discount (EURO)",
#         "CAPEX net (EURO)",
#         "CAPEX discount (EURO)",
#         "OPEX net (EURO)",
#         "OPEX discount (EURO)",
#         "LCOE (EURO/MWh)",
#     ]

#     failed = False
#     for parameter_idx in parameter_idxs:
#         params = {
#             key: sample_parameters[key][idx]
#             for key, idx in zip(sample_parameters.keys(), parameter_idx)
#         }

#         input_map = dtu_offshore_cm_input_map(**params)
#         output_map = dtu_offshore_cm_output_map()
#         excel_file = Path(os.path.dirname(__file__), "data/WTcostmodel_v12.xlsx")
#         assert excel_file.exists()
#         excel_result = run_excel(
#             file_path=excel_file,
#             input_map=input_map,
#             output_map=output_map,
#         )
#         # res_excel.append(excel_result)

#         adapted_params = params.copy()
#         for key in ["decline_factor", "profit", "capacity_factor", "wacc", "inflation"]:
#             adapted_params[key] *= -100 if key == "decline_factor" else 100
#         adapted_params["lifetime"] = adapted_params.pop("project_lifetime")
#         if "eprice" not in adapted_params:
#             adapted_params["eprice"] = 0.2

#         new_cm = DTUOffshoreCostModel(**adapted_params)
#         results_new = new_cm.run()

#         new_results_mapped = {
#             "AEP net (MWh)": results_new["aep_net"].m,
#             "AEP discount (MWh)": results_new["aep_discount"].m,
#             "DEVEX net (EURO)": results_new["devex_net"].to("EUR").m,
#             "DEVEX discount (EURO)": results_new["devex_discount"].to("EUR").m,
#             "CAPEX net (EURO)": results_new["capex"].to("EUR").m,
#             "CAPEX discount (EURO)": results_new["capex_discount"].to("EUR").m,
#             "OPEX net (EURO)": results_new["opex"].to("EUR").m,
#             "OPEX discount (EURO)": results_new["opex_discount"].to("EUR").m,
#             "LCOE (EURO/MWh)": results_new["lcoe"].to("EUR/MWh").m,
#         }

#         for key in metrics:
#             org_k_val = new_results_mapped[key]
#             if "AEP" in key:
#                 org_k_val = org_k_val * params["project_lifetime"]
#             excel_k_val = excel_result[key]
#             if (np.abs(org_k_val - excel_k_val) > 1).any():
#                 warnings.warn(
#                     f"Warning: {key} values are not close enough. Original: {org_k_val}, Excel: {excel_k_val}; Parameters: {params}"
#                 )
#                 failed = True

#     assert not failed
#     from .utils.winutil import ExcelManager
#     ExcelManager.close_app()


# @pytest.mark.skipif(platform.system() != "Windows", reason="Only run on Windows")
# def test_original_dtu_cm_implementation_win_excel_monte_carlo():
#     n_samples = 100
#     num = 10
#     sp = 350
#     clearance = 20
#     sample_parameters = dict(
#         rated_power=np.linspace(1, 20, num),
#         rotor_diameter=np.sqrt(4 * np.linspace(1, 20, num) * 10**6 / (np.pi * sp)),
#         rotor_speed=np.linspace(5, 10, num),
#         hub_height=clearance
#         + np.sqrt(4 * np.linspace(1, 20, num) * 10**6 / (np.pi * sp)) / 2,
#         profit=np.linspace(0.01, 0.1, num),
#         capacity_factor=np.linspace(0.3, 0.6, num),
#         decline_factor=np.linspace(-0.02, -0.01, num),
#         nwt=np.arange(10, 400, 40),
#         project_lifetime=np.arange(15, 25),
#         wacc=np.linspace(0.05, 0.1, num),
#         inflation=np.linspace(0.01, 0.1, num),
#         opex=np.linspace(10, 40, num),
#         devex=np.linspace(0, 20, num),
#         abex=np.linspace(0, 10, num),
#         water_depth=np.linspace(0, 100, num),
#         electrical_cost=np.linspace(0, 10, num),
#         foundation_option=np.arange(5),
#     )

#     shape = 16 * (10,) + (5,)
#     size = 10**16 * 5

#     np.random.seed(42)
#     sample_nos = np.random.uniform(high=size, size=n_samples)
#     parameter_idxs = [
#         np.unravel_index(int(sample_no), shape) for sample_no in sample_nos
#     ]

#     res_original = []
#     res_excel = []
#     excel_file = Path(os.path.dirname(__file__), "data/WTcostmodel_v12.xlsx")
#     assert excel_file.exists()

#     metrics = [
#         "AEP net (MWh)",
#         "AEP discount (MWh)",
#         "DEVEX net (EURO)",
#         "DEVEX discount (EURO)",
#         "CAPEX net (EURO)",
#         "CAPEX discount (EURO)",
#         "OPEX net (EURO)",
#         "OPEX discount (EURO)",
#         "LCOE (EURO/MWh)",
#     ]

#     failed = False
#     for parameter_idx in parameter_idxs:
#         params = {
#             key: sample_parameters[key][idx]
#             for key, idx in zip(sample_parameters.keys(), parameter_idx)
#         }

#         # Run original DTU model implementation
#         cm = FafasDTUOffshoreCostModel(**params)
#         results = cm.run()
#         res_original.append(results)

#         # Run Excel implementation
#         input_map = dtu_offshore_cm_input_map(**params)
#         output_map = dtu_offshore_cm_output_map()
#         excel_result = run_excel(
#             file_path=excel_file,
#             input_map=input_map,
#             output_map=output_map,
#             reuse_excel=True,
#         )
#         res_excel.append(excel_result)

#         for key in metrics:
#             org_k_val = results[key]
#             if "AEP" in key:
#                 org_k_val = org_k_val * params["project_lifetime"]
#             excel_k_val = excel_result[key]
#             if (np.abs(org_k_val - excel_k_val) > 1).any():
#                 warnings.warn(
#                     f"Warning: {key} values are not close enough. Original: {org_k_val}, Excel: {excel_k_val}; Parameters: {params}"
#                 )
#                 failed = True

#     assert not failed
#     from .utils.winutil import ExcelManager

#     ExcelManager.close_app()

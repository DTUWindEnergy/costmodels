from costmodels import DTUOffshoreCostModel
from costmodels.units import Quant

from .utils.DTU_CostModel_org import DTUOffshoreCostModel as FafasDTUOffshoreCostModel


def test_dtu_offshore():
    cm = DTUOffshoreCostModel(
        rated_power=Quant(5.111111111111111, "MW"),
        rotor_diameter=0.060314403509210746,
        rotor_speed=9.444444444444445,
        hub_height=20.111486515663536,
        profit=Quant(1, "%"),
        capacity_factor=Quant(33.333, "%"),
        decline_factor=Quant(2, "%"),
        nwt=290,
        lifetime=27,
        wacc=0.07222222222222223,
        inflation=Quant(8, "%"),
        opex=0.0,
        devex=11.11111111111111,
        abex=5.555555555555555,
        water_depth=33.33333333333333,
        electrical_cost=0.0,
        foundation_option=0,
        eprice=Quant(0.2, "EUR/kWh"),
        aep=[5 * 1e6 * 8760, 5 * 1e6 * 8760],
    )
    cm_output = cm.run()

    cm_output_aep = cm.run(aep=1.0)

    assert cm_output_aep["aep_net"] != cm_output["aep_net"]

    grads = cm.grad("npv", ("eprice",))
    assert "eprice" in grads.keys()


def test_agains_original_dtu_offshore_implementation():
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
        "foundation_option": 2,
    }

    origcm = FafasDTUOffshoreCostModel(**params)
    results = origcm.run()
    results["CO2 emission (kg CO2 eq))"] = results[
        "Total Co2 emission per turbine (kg CO2 eq)"
    ].mean()
    results.pop("Total Co2 emission per turbine (kg CO2 eq)")
    results.pop("Turbine cost (EURO)")

    adapted_params = params.copy()
    for key in ["decline_factor", "profit", "capacity_factor", "wacc", "inflation"]:
        adapted_params[key] *= -100 if key == "decline_factor" else 100
    adapted_params["lifetime"] = adapted_params.pop("project_lifetime")
    if "eprice" not in adapted_params:
        adapted_params["eprice"] = 0.2

    cm = DTUOffshoreCostModel(**adapted_params)
    results_our = cm.run()

    new_results_mapped = {
        "AEP net (MWh)": results_our["aep_net"].m,
        "AEP discount (MWh)": results_our["aep_discount"].m,
        "DEVEX net (EURO)": results_our["devex_net"].to("EUR").m,
        "DEVEX discount (EURO)": results_our["devex_discount"].to("EUR").m,
        "CAPEX net (EURO)": results_our["capex"].to("EUR").m,
        "CAPEX discount (EURO)": results_our["capex_discount"].to("EUR").m,
        "OPEX net (EURO)": results_our["opex"].to("EUR").m,
        "OPEX discount (EURO)": results_our["opex_discount"].to("EUR").m,
        "LCOE (EURO/MWh)": results_our["lcoe"].to("EUR/MWh").m,
    }

    for k, v in new_results_mapped.items():
        if k in results:
            assert abs(v - results[k]) < 1e-3, f"Mismatch in {k}: {v} vs {results[k]}"

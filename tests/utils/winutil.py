# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:10:46 2024

@author: mikf
"""
import copy


class ExcelManager:
    """Singleton manager for Excel application"""

    _instance = None
    _app = None

    @classmethod
    def get_app(cls, visible=False):
        """Get or create an Excel application instance"""
        if cls._app is None:
            import platform

            assert platform.system() in [
                "Windows",
                "Darwin",
            ], "Only Windows and MacOS are supported"
            import xlwings as xw

            cls._app = xw.App(visible=visible)
        return cls._app

    @classmethod
    def close_app(cls):
        """Close the Excel application if it exists"""
        if cls._app is not None:
            cls._app.kill()
            cls._app = None


class Excel:
    def __init__(self, visible=False):
        import platform

        assert platform.system() in [
            "Windows",
            "Darwin",
        ], "Only Windows and MacOS are supported"
        import xlwings as xw

        self.app = xw.App(visible=visible)

    def __enter__(self):
        return self.app

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.app.kill()


def run_excel(
    file_path="WTcostmodel_v12.xlsx",
    input_map={
        "sheet_name": [
            {
                "name": "Power (kW)",
                "cell": "A2",
                "value": 7e3,
            }
        ],
    },
    output_map={
        "sheet_name": [
            {
                "name": "LCOE (EURO/MWh)",
                "cell": "M7",
            }
        ],
    },
    reuse_excel=False,
    close_excel=False,
):
    """
    Run Excel calculations with given inputs and return outputs

    Args:
        file_path: Path to the Excel file
        input_map: Dictionary mapping sheet names to lists of input cells and values
        output_map: Dictionary mapping sheet names to lists of output cells to read
        reuse_excel: Whether to reuse an existing Excel application
        close_excel: Whether to close Excel after this operation
    """
    import platform

    assert platform.system() in [
        "Windows",
        "Darwin",
    ], "Only Windows and MacOS are supported"
    import xlwings as xw

    if reuse_excel:
        app = ExcelManager.get_app()
        wb = app.books.open(file_path)
        try:
            for sheet, input_list in input_map.items():
                sht = wb.sheets[sheet]
                for inputs in input_list:
                    sht[inputs["cell"]].value = inputs["value"]
            res = copy.deepcopy(output_map)
            for sheet, output_list in res.items():
                sht = wb.sheets[sheet]
                for n, outputs in enumerate(output_list):
                    res[sheet][n]["value"] = sht[outputs["cell"]].value
            out = simplify_output(res)
            return out
        finally:
            wb.close()
            if close_excel:
                ExcelManager.close_app()
    else:
        # Original implementation using context manager
        with Excel() as app:
            wb = app.books.open(file_path)
            for sheet, input_list in input_map.items():
                sht = wb.sheets[sheet]
                for inputs in input_list:
                    sht[inputs["cell"]].value = inputs["value"]
            res = copy.deepcopy(output_map)
            for sheet, output_list in res.items():
                sht = wb.sheets[sheet]
                for n, outputs in enumerate(output_list):
                    res[sheet][n]["value"] = sht[outputs["cell"]].value
        out = simplify_output(res)
        return out


def dtu_offshore_cm_input_map(
    rated_power=7,
    rotor_diameter=154,
    rotor_speed=7,
    hub_height=106.7,
    profit=0.04,
    capacity_factor=0.424,
    decline_factor=-0.015,
    nwt=37,
    project_lifetime=25,
    wacc=0.08,
    inflation=0.03,
    opex=20,
    devex=10,
    abex=0,
    foundation_option=1,
    water_depth=18,
    electrical_cost=0,
    **kwargs,
):
    input_map = {
        "Main": [
            {
                "name": "Rated power (MW)",
                "cell": "C5",
                "value": rated_power,
            },
            {
                "name": "Rotor diameter (m)",
                "cell": "C6",
                "value": rotor_diameter,
            },
            {
                "name": "Rotor speed (rpm)",
                "cell": "C7",
                "value": rotor_speed,
            },
            {
                "name": "Tower height (m)",
                "cell": "C8",
                "value": hub_height,
            },
            {
                "name": "Profit (-)",
                "cell": "C15",
                "value": profit,
            },
            {
                "name": "Capacity factor (-)",
                "cell": "C23",
                "value": capacity_factor,
            },
            {
                "name": "AEP decl. factor (-)",
                "cell": "C24",
                "value": decline_factor,
            },
            {
                "name": "No of turbines (-)",
                "cell": "C25",
                "value": nwt,
            },
            {
                "name": "Project life time (years)",
                "cell": "C26",
                "value": project_lifetime,
            },
            {
                "name": "Nominal WACC (-)",
                "cell": "C27",
                "value": wacc,
            },
            {
                "name": "Infaltion (-)",
                "cell": "C28",
                "value": inflation,
            },
            {
                "name": "OPEX (EURO/kW/year)",
                "cell": "C30",
                "value": opex,
            },
            {
                "name": "DEVEX (EURO/kW)",
                "cell": "C31",
                "value": devex,
            },
            {
                "name": "ABEX (EURO)",
                "cell": "C32",
                "value": abex,
            },
            {
                "name": "Foundation option",
                "cell": "B35",
                "value": foundation_option,
            },
            {
                "name": "Water depth (m)",
                "cell": "B36",
                "value": water_depth,
            },
            {
                "name": "Electrical infrastructure (MEURO/MW)",
                "cell": "C40",
                "value": electrical_cost,
            },
        ],
    }
    return input_map


def dtu_offshore_cm_output_map():
    output_map = {
        "Main": [
            {
                "name": "LCOE (EURO/MWh)",
                "cell": "H27",
            },
            {
                "name": "Total CAPEX (EURO))",
                "cell": "H32",
            },
            {
                "name": "CO2 emission (kg CO2 eq))",
                "cell": "H44",
            },
        ],
        "LCOE_Calc_Original": [
            {
                "name": "AEP net (MWh)",
                "cell": "F40",
            },
            {
                "name": "AEP discount (MWh)",
                "cell": "F41",
            },
            {
                "name": "DEVEX net (EURO)",
                "cell": "F43",
            },
            {
                "name": "DEVEX discount (EURO)",
                "cell": "F44",
            },
            {
                "name": "CAPEX net (EURO)",
                "cell": "F46",
            },
            {
                "name": "CAPEX discount (EURO)",
                "cell": "F47",
            },
            {
                "name": "OPEX net (EURO)",
                "cell": "F49",
            },
            {
                "name": "OPEX discount (EURO)",
                "cell": "F50",
            },
        ],
    }
    return output_map


def run_dtu_offshore_cost_model_excel(
    rated_power=7,
    rotor_diameter=154,
    rotor_speed=7,
    hub_height=106.7,
    profit=0.04,
    capacity_factor=0.424,
    decline_factor=-0.015,
    nwt=37,
    project_lifetime=25,
    wacc=0.08,
    inflation=0.03,
    opex=20,
    devex=10,
    abex=0,
    foundation_option=1,
    water_depth=18,
    electrical_cost=0,
    filepath="WTcostmodel_v12.xlsx",
):
    input_map = dtu_offshore_cm_input_map(
        rated_power=rated_power,
        rotor_diameter=rotor_diameter,
        rotor_speed=rotor_speed,
        hub_height=hub_height,
        profit=profit,
        capacity_factor=capacity_factor,
        decline_factor=decline_factor,
        nwt=nwt,
        project_lifetime=project_lifetime,
        wacc=wacc,
        inflation=inflation,
        opex=opex,
        devex=devex,
        abex=abex,
        foundation_option=foundation_option,
        water_depth=water_depth,
        electrical_cost=electrical_cost,
    )
    output_map = dtu_offshore_cm_output_map()
    return run_excel(file_path=filepath, input_map=input_map, output_map=output_map)


def simplify_output(output_map):
    dic = {}
    for _, v in output_map.items():
        dic.update({x["name"]: x["value"] for x in v})
    return dic


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    rps = np.arange(1, 25)

    ress = []
    for rp in rps:
        ress.append(run_dtu_offshore_cost_model_excel(rated_power=rp))

    lcoe = [x["LCOE (EURO/MWh)"] for x in ress]

    plt.figure()
    plt.plot(rps, lcoe)

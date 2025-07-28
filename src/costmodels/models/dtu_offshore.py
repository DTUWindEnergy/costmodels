import dataclasses
from enum import Enum

import jax.numpy as jnp

np = jnp

from costmodels._interface import (
    CostInput,
    CostModel,
    CostOutput,
    cost_input_dataclass,
)


class Foundation(Enum):
    NONE = 0
    MONOPILE = 1
    GRAVITY = 2
    JACKET = 3
    FLOATING_MOCKUP = 4


class Currency(Enum):
    DKK = "DKK"
    EURO = "EURO"
    DKK_KW = "DKK/KW"
    EURO_KW = "EURO/KW"


@cost_input_dataclass
class DTUOffshoreCostInput(CostInput):
    lifetime: float  # years
    rated_power: float  # MW
    rotor_speed: float  # rpm
    rotor_diameter: float  # m
    hub_height: float  # m
    capacity_factor: float  # %
    nwt: float
    water_depth: float  # m
    foundation_option: Foundation = Foundation.MONOPILE
    profit: float = 0.01  # %
    decline_factor: float = 0.01  # %
    wacc: float = 0.07  # %
    devex: float = 0.0  # EUR/kW
    abex: float = 0.0  # EUR
    electrical_cost: float = 0.0  # MEUR/MW
    currency: Currency = Currency.EURO_KW
    eur_to_dkk: float = 7.54
    aep: jnp.ndarray  # MWh
    inflation: float = 0.02  # %
    opex: float = 0.02  # EUR/kW
    eprice: float = 0.1  # EUR/kWh


class DTUOffshoreCostModel(CostModel):
    """
    Parameters:
        rated_power: Rated power of the wind turbine.
        rotor_speed: Speed of the rotor.
        rotor_diameter: Diameter of the rotor.
        hub_height: Height of the tower.
        foundation_option: Option for foundation (0: none, 1: monopile, 2: gravity, 3: jacket, 4: floating mockup)
        water_depth: Depth of water for offshore installation.
        currency: Currency for financial calculations ('DKK', 'EURO', 'DKK/KW', 'EURO/KW')
        eur_to_dkk: Rate of change of Dkk to Euro
        wacc: Weighted Average Cost of Capital in nominal terms.
        devex (float or None): Development expenditures.
        decline_factor (float): Annual Energy Production decline factor.
        inflation (float): Inflation rate.
        lifetime (int): Project lifespan in years.
        opex (float or None): Operational expenditures.
        abex (float or None): Asset-based expenditures.
        capacity_factor (float or None): Capacity factor.
        profit: Profit margin.
        AEP (float, array): Annual Energy Production, from Pywake.
        nwt (int or None): Number of wind turbines.
        electrical_cost (int): Electrical infrastructure cost in MEURO/MW
    """

    _inputs_cls = DTUOffshoreCostInput

    # --- Helper Methods (Not Cached) ---

    def convert_currency(self, foundation_cost):
        """
        Convert the foundation cost to the specified currency.
        """
        rates = {
            "DKK": 1,
            "EURO": 1 / self.eur_to_dkk,
            "DKK/KW": 1 / 1000,
            "EURO/KW": 1 / (self.eur_to_dkk * 1000),
        }
        key = self.currency.value if isinstance(self.currency, Enum) else self.currency
        return foundation_cost * rates.get(key, 1)

    def CalculateFoundationCost(self):
        """
        Calculate the foundation cost based on the water depth and foundation option,
        and convert it to the specified currency.
        """
        # Define foundation cost coefficients and intercepts
        monopile_coeff_sq = 1000
        monopile_coeff_lin = 100000
        monopile_intercept = 1500000
        gravity_coeff_sq = 6000
        gravity_coeff_lin = -100000
        gravity_intercept = 1500000
        jacket_coeff_sq = 4000
        jacket_coeff_lin = -25000
        jacket_intercept = 3000000
        floating_mockup_cost_per_kw_eur = 1250

        # Calculate the foundation cost based on foundation option and water depth
        costs = {
            Foundation.NONE.value: 0.0,
            Foundation.MONOPILE.value: monopile_coeff_sq * (self.water_depth**2)
            + monopile_coeff_lin * self.water_depth
            + monopile_intercept,
            Foundation.GRAVITY.value: gravity_coeff_sq * (self.water_depth**2)
            + gravity_coeff_lin * self.water_depth
            + gravity_intercept,
            Foundation.JACKET.value: jacket_coeff_sq * (self.water_depth**2)
            + jacket_coeff_lin * self.water_depth
            + jacket_intercept,
            Foundation.FLOATING_MOCKUP.value: floating_mockup_cost_per_kw_eur
            * self.eur_to_dkk
            * 1000,
        }
        # Default to monopile if option not found
        foundation_cost_dkk = costs.get(
            self.foundation_option.value,
            monopile_coeff_sq * (self.water_depth**2)
            + monopile_coeff_lin * self.water_depth
            + monopile_intercept,
        )
        return self.convert_currency(foundation_cost_dkk)

    def reformat_input(self, **kwargs):
        nwt = kwargs["nwt"]

        for k, v in kwargs.items():
            vmag = v
            if (
                k
                in (
                    "rated_power",
                    "rotor_speed",
                    "rotor_diameter",
                    "hub_height",
                    "water_depth",
                )
                and np.size(v) == 1
            ):
                setattr(self, k, np.tile(vmag, nwt))
                continue
            if k in ("nwt", "lifetime"):
                setattr(self, k, int(vmag))
                continue
            setattr(self, k, vmag)

    # --- Main Calculation Method ---

    def _run(self, inputs: DTUOffshoreCostInput) -> CostOutput:
        # Convert foundation_option to Enum if it's not already
        if not isinstance(inputs.foundation_option, Foundation):
            inputs = dataclasses.replace(
                inputs, foundation_option=Foundation(inputs.foundation_option)
            )

        self.reformat_input(**dataclasses.asdict(inputs))
        if self.nwt == 0:
            raise ValueError(
                "Number of turbines (nwt) must be provided for this calculation."
            )

        # --- Base Calculations ---
        rotor_torque_factor = 1.1
        rotor_torque = (
            rotor_torque_factor * 60 * self.rated_power / (2 * np.pi * self.rotor_speed)
        )
        rotor_area = np.pi * (self.rotor_diameter / 2) ** 2

        # --- Mass Calculation Parameters ---
        # Coefficients
        blade_mass_coeff = 1.65
        hub_structure_mass_coeff = 0.5
        hub_computer_mass_coeff = 0.0
        pitch_bearings_mass_coeff = 0.4
        pitch_actuator_mass_coeff = 0.15
        hub_secondary_mass_coeff = 5.0
        spinner_mass_coeff = 10.0
        main_shaft_mass_coeff = 0.02
        main_bearings_mass_coeff = 0.02
        main_bearing_housing_mass_coeff = 0.03
        gearbox_mass_coeff = 12500.0
        coupling_brake_mass_coeff = 500.0
        generator_mass_coeff = 1800.0
        cooling_mass_coeff = 500.0
        power_converter_mass_coeff = 1000.0
        controller_mass_coeff = 100.0
        bedplate_mass_coeff = 1.2
        yaw_system_mass_coeff = 0.1
        canopy_mass_coeff = 1500.0
        nacelle_secondary_mass_coeff = 1000.0
        tower_structure_mass_coeff = 0.25
        tower_internals_mass_coeff = 100.0
        power_cables_mass_coeff = 25.0
        main_transformer_mass_coeff = 2500.0
        tower_secondary_mass_coeff = 500.0
        # Intercepts
        blade_mass_intercept = 0.0
        hub_structure_mass_intercept = 6000.0
        hub_computer_mass_intercept = 200.0
        pitch_bearings_mass_intercept = 500.0
        pitch_actuator_mass_intercept = 500.0
        hub_secondary_mass_intercept = 500.0
        spinner_mass_intercept = 0.0
        main_shaft_mass_intercept = 0.0
        main_bearings_mass_intercept = 0.0
        main_bearing_housing_mass_intercept = 0.0
        gearbox_mass_intercept = 0.0
        coupling_brake_mass_intercept = 0.0
        generator_mass_intercept = 0.0
        cooling_mass_intercept = 0.0
        power_converter_mass_intercept = 0.0
        controller_mass_intercept = 200.0
        bedplate_mass_intercept = 0.0
        yaw_system_mass_intercept = 0.0
        canopy_mass_intercept = 1000.0
        nacelle_secondary_mass_intercept = 1000.0
        tower_structure_mass_intercept = 0.0
        tower_internals_mass_intercept = 1000.0
        power_cables_mass_intercept = 0.0
        main_transformer_mass_intercept = 0.0
        tower_secondary_mass_intercept = 1000.0
        # Exponents
        blade_mass_exp = 2.5
        hub_structure_mass_exp = 2.5
        hub_computer_mass_exp = 1.0
        pitch_bearings_mass_exp = 2.5
        pitch_actuator_mass_exp = 2.5
        hub_secondary_mass_exp = 1.0
        spinner_mass_exp = 1.0
        main_shaft_mass_exp = 2.8
        main_bearings_mass_exp = 2.5
        main_bearing_housing_mass_exp = 2.5
        gearbox_mass_exp = 1.0
        coupling_brake_mass_exp = 1.0
        generator_mass_exp = 1.0
        cooling_mass_exp = 1.0
        power_converter_mass_exp = 1.0
        controller_mass_exp = 1.0
        bedplate_mass_exp = 2.0
        yaw_system_mass_exp = 2.5
        canopy_mass_exp = 1.0
        nacelle_secondary_mass_exp = 1.0
        tower_structure_mass_exp = 1.0
        tower_internals_mass_exp = 1.0
        power_cables_mass_exp = 1.0
        main_transformer_mass_exp = 1.0
        tower_secondary_mass_exp = 1.0

        # --- Mass Calculations ---
        rotor_radius = self.rotor_diameter / 2
        total_blade_mass = (
            blade_mass_coeff * (rotor_radius**blade_mass_exp) + blade_mass_intercept
        )
        hub_structure_mass = (
            hub_structure_mass_coeff * (rotor_radius**hub_structure_mass_exp)
            + hub_structure_mass_intercept
        )
        hub_computer_mass = (
            hub_computer_mass_coeff * (self.rotor_diameter**hub_computer_mass_exp)
            + hub_computer_mass_intercept
        )
        pitch_bearings_mass = (
            pitch_bearings_mass_coeff * (rotor_radius**pitch_bearings_mass_exp)
            + pitch_bearings_mass_intercept
        )
        pitch_actuator_system_mass = (
            pitch_actuator_mass_coeff * (rotor_radius**pitch_actuator_mass_exp)
            + pitch_actuator_mass_intercept
        )
        hub_secondary_equipment_mass = (
            hub_secondary_mass_coeff * (self.rotor_diameter**hub_secondary_mass_exp)
            + hub_secondary_mass_intercept
        )
        spinner_mass = (
            spinner_mass_coeff * (self.rotor_diameter**spinner_mass_exp)
            + spinner_mass_intercept
        )
        main_shaft_mass = (
            main_shaft_mass_coeff * (self.rotor_diameter**main_shaft_mass_exp)
            + main_shaft_mass_intercept
        )
        main_bearings_mass = (
            main_bearings_mass_coeff * (self.rotor_diameter**main_bearings_mass_exp)
            + main_bearings_mass_intercept
        )
        main_bearing_housing_mass = (
            main_bearing_housing_mass_coeff
            * (self.rotor_diameter**main_bearing_housing_mass_exp)
            + main_bearing_housing_mass_intercept
        )
        gearbox_mass = (
            gearbox_mass_coeff * (rotor_torque**gearbox_mass_exp)
            + gearbox_mass_intercept
        )
        coupling_plus_brake_system_mass = (
            coupling_brake_mass_coeff * (self.rated_power**coupling_brake_mass_exp)
            + coupling_brake_mass_intercept
        )
        generator_mass = (
            generator_mass_coeff * (self.rated_power**generator_mass_exp)
            + generator_mass_intercept
        )
        cooling_mass = (
            cooling_mass_coeff * (self.rated_power**cooling_mass_exp)
            + cooling_mass_intercept
        )
        power_converter_mass = (
            power_converter_mass_coeff * (self.rated_power**power_converter_mass_exp)
            + power_converter_mass_intercept
        )
        controller_mass = (
            controller_mass_coeff * (self.rated_power**controller_mass_exp)
            + controller_mass_intercept
        )
        bedplate_mass = (
            bedplate_mass_coeff * (self.rotor_diameter**bedplate_mass_exp)
            + bedplate_mass_intercept
        )
        yaw_system_mass = (
            yaw_system_mass_coeff * (self.rotor_diameter**yaw_system_mass_exp)
            + yaw_system_mass_intercept
        )
        canopy_mass = (
            canopy_mass_coeff * (self.rated_power**canopy_mass_exp)
            + canopy_mass_intercept
        )
        nacell_secondary_equipment_mass = (
            nacelle_secondary_mass_coeff
            * (self.rated_power**nacelle_secondary_mass_exp)
            + nacelle_secondary_mass_intercept
        )
        tower_structure_mass = (
            tower_structure_mass_coeff
            * (self.hub_height * rotor_area) ** tower_structure_mass_exp
            + tower_structure_mass_intercept
        )
        tower_internals_mass = (
            tower_internals_mass_coeff * (self.hub_height**tower_internals_mass_exp)
            + tower_internals_mass_intercept
        )
        power_cables_mass = (
            power_cables_mass_coeff
            * ((self.rated_power * self.hub_height) ** power_cables_mass_exp)
            + power_cables_mass_intercept
        )
        main_transformer_mass = (
            main_transformer_mass_coeff * (self.rated_power**main_transformer_mass_exp)
            + main_transformer_mass_intercept
        )
        tower_secondary_equipment_mass = (
            tower_secondary_mass_coeff * (self.rated_power**tower_secondary_mass_exp)
            + tower_secondary_mass_intercept
        )

        # --- Total Mass Calculations ---
        hub_total_mass = (
            hub_structure_mass
            + pitch_bearings_mass
            + pitch_actuator_system_mass
            + hub_computer_mass
            + hub_secondary_equipment_mass
            + spinner_mass
        )
        nacelle_total_mass = (
            main_shaft_mass
            + main_bearings_mass
            + main_bearing_housing_mass
            + gearbox_mass
            + coupling_plus_brake_system_mass
            + generator_mass
            + cooling_mass
            + power_converter_mass
            + controller_mass
            + bedplate_mass
            + yaw_system_mass
            + canopy_mass
            + nacell_secondary_equipment_mass
        )
        tower_total_mass = (
            tower_structure_mass
            + tower_internals_mass
            + power_cables_mass
            + main_transformer_mass
            + tower_secondary_equipment_mass
        )
        bom_total_mass = (
            total_blade_mass + hub_total_mass + nacelle_total_mass + tower_total_mass
        )

        # --- Cost Calculation Parameters ---
        # Rates (Cost per unit mass)
        blade_cost_rate = 15.0
        hub_structure_cost_rate = 2.5
        hub_computer_cost_rate = 50.0
        pitch_bearings_cost_rate = 8.0
        pitch_actuator_cost_rate = 8.0
        hub_secondary_cost_rate = 8.0
        spinner_cost_rate = 10.0
        main_shaft_cost_rate = 5.0
        main_bearings_cost_rate = 15.0
        main_bearing_housing_cost_rate = 2.5
        gearbox_cost_rate = 8.0
        coupling_brake_cost_rate = 8.0
        generator_cost_rate = 8.0
        cooling_cost_rate = 8.0
        power_converter_cost_rate = 30.0
        controller_cost_rate = 50.0
        bedplate_cost_rate = 2.5
        yaw_system_cost_rate = 6.0
        canopy_cost_rate = 10.0
        nacelle_secondary_cost_rate = 10.0
        tower_structure_cost_rate = 3.0
        tower_internals_cost_rate = 8.0
        power_cables_cost_rate = 8.0
        main_transformer_cost_rate = 8.0
        tower_secondary_cost_rate = 10.0
        # Coefficients for non-mass-based costs
        material_overhead_cost_coeff = 0.03
        direct_labor_cost_coeff = 0.10
        overhead_cost_coeff = 0.05
        r_and_d_cost_coeff = 0.025
        sg_and_a_cost_coeff = 0.05
        warranty_accruals_cost_coeff = 0.03
        financing_cost_coeff = 0.017778
        transport_cost_coeff = 0.2
        transport_cost_intercept = 10000.0
        transport_cost_exp = 1.0
        harbor_storage_assy_cost_coeff = 0.0  # Simplified from original
        harbor_storage_assy_cost_intercept = 0.0
        harbor_storage_assy_cost_exp = 1.0
        installation_commiss_cost_coeff = 0.0  # Simplified from original
        installation_commiss_cost_intercept = 0.0
        installation_commiss_cost_exp = 1.0

        # --- Cost Calculations ---
        blade_total_cost = total_blade_mass * blade_cost_rate
        hub_structure_cost = hub_structure_mass * hub_structure_cost_rate
        hub_computer_cost = hub_computer_mass * hub_computer_cost_rate
        pitch_bearings_cost = pitch_bearings_mass * pitch_bearings_cost_rate
        pitch_actuator_system_cost = (
            pitch_actuator_system_mass * pitch_actuator_cost_rate
        )
        hub_secondary_equipment_cost = (
            hub_secondary_equipment_mass * hub_secondary_cost_rate
        )
        spinner_cost = spinner_mass * spinner_cost_rate
        main_shaft_cost = main_shaft_mass * main_shaft_cost_rate
        main_bearings_cost = main_bearings_mass * main_bearings_cost_rate
        main_bearing_housing_cost = (
            main_bearing_housing_mass * main_bearing_housing_cost_rate
        )
        gearbox_cost = gearbox_mass * gearbox_cost_rate
        coupling_plus_brake_system_cost = (
            coupling_plus_brake_system_mass * coupling_brake_cost_rate
        )
        generator_cost = generator_mass * generator_cost_rate
        cooling_cost = cooling_mass * cooling_cost_rate
        power_converter_cost = power_converter_mass * power_converter_cost_rate
        controller_cost = controller_mass * controller_cost_rate
        bedplate_cost = bedplate_mass * bedplate_cost_rate
        yaw_system_cost = yaw_system_mass * yaw_system_cost_rate
        canopy_cost = canopy_mass * canopy_cost_rate
        nacell_secondary_equipment_cost = (
            nacell_secondary_equipment_mass * nacelle_secondary_cost_rate
        )
        tower_structure_cost = tower_structure_mass * tower_structure_cost_rate
        tower_internals_cost = tower_internals_mass * tower_internals_cost_rate
        power_cables_cost = power_cables_mass * power_cables_cost_rate
        main_transformer_cost = main_transformer_mass * main_transformer_cost_rate
        tower_secondary_equipment_cost = (
            tower_secondary_equipment_mass * tower_secondary_cost_rate
        )

        # --- Total Cost Calculations ---
        hub_total_cost = (
            hub_structure_cost
            + pitch_bearings_cost
            + pitch_actuator_system_cost
            + hub_computer_cost
            + hub_secondary_equipment_cost
            + spinner_cost
        )
        nacelle_total_cost = (
            main_shaft_cost
            + main_bearings_cost
            + main_bearing_housing_cost
            + gearbox_cost
            + coupling_plus_brake_system_cost
            + generator_cost
            + cooling_cost
            + power_converter_cost
            + controller_cost
            + bedplate_cost
            + yaw_system_cost
            + canopy_cost
            + nacell_secondary_equipment_cost
        )
        tower_total_cost = (
            tower_structure_cost
            + tower_internals_cost
            + power_cables_cost
            + main_transformer_cost
            + tower_secondary_equipment_cost
        )
        bom_total_cost = (
            blade_total_cost + hub_total_cost + nacelle_total_cost + tower_total_cost
        )
        material_overhead_cost = bom_total_cost * material_overhead_cost_coeff
        direct_labor_cost = bom_total_cost * direct_labor_cost_coeff
        direct_production_cost = (
            bom_total_cost + material_overhead_cost + direct_labor_cost
        )
        overhead_cost = direct_production_cost * overhead_cost_coeff
        r_and_d_cost = direct_production_cost * r_and_d_cost_coeff
        sg_and_a_cost = direct_production_cost * sg_and_a_cost_coeff
        total_production_cost = (
            direct_production_cost + overhead_cost + r_and_d_cost + sg_and_a_cost
        )
        warranty_accruals_cost = total_production_cost * warranty_accruals_cost_coeff
        financing_cost = total_production_cost * financing_cost_coeff
        transport_cost = (
            transport_cost_coeff * (bom_total_mass**transport_cost_exp)
            + transport_cost_intercept
        )
        harbor_storage_assy_cost = (
            harbor_storage_assy_cost_coeff
            * (self.rated_power**harbor_storage_assy_cost_exp)
            + harbor_storage_assy_cost_intercept
        )
        installation_commiss_cost = (
            installation_commiss_cost_coeff
            * (self.rated_power**installation_commiss_cost_exp)
            + installation_commiss_cost_intercept
        )
        total_additional_cost = (
            warranty_accruals_cost
            + financing_cost
            + transport_cost
            + harbor_storage_assy_cost
            + installation_commiss_cost
        )
        total_cost_calculation = total_additional_cost + total_production_cost
        profit_calculation = -(1 - 1 / (1 - self.profit)) * total_cost_calculation
        sales_price_calculation = total_cost_calculation + profit_calculation

        # --- CO2 Emission Calculation Parameters ---
        # Emission Factors (kg CO2 / kg mass)
        blade_co2_factor = 4.00
        hub_structure_co2_factor = 1.83
        hub_computer_co2_factor = 3.00
        pitch_bearings_co2_factor = 1.83
        pitch_actuator_co2_factor = 1.83
        hub_secondary_co2_factor = 1.83
        spinner_co2_factor = 4.00
        main_shaft_co2_factor = 1.83
        main_bearings_co2_factor = 1.83
        main_bearing_housing_co2_factor = 1.83
        gearbox_co2_factor = 1.83
        coupling_brake_co2_factor = 1.83
        generator_co2_factor = 6.00
        cooling_co2_factor = 2.00
        power_converter_co2_factor = 4.00
        controller_co2_factor = 1.83
        bedplate_co2_factor = 1.83
        yaw_system_co2_factor = 1.83
        canopy_co2_factor = 4.00
        nacelle_secondary_co2_factor = 1.83
        tower_structure_co2_factor = 1.83
        tower_internals_co2_factor = 2.00
        power_cables_co2_factor = 4.00
        main_transformer_co2_factor = 4.00
        tower_secondary_co2_factor = 2.00

        # --- CO2 Emission Calculations ---
        blade_co2_emission = blade_co2_factor * total_blade_mass
        hub_structure_co2_emission = hub_structure_co2_factor * hub_structure_mass
        hub_computer_co2_emission = hub_computer_co2_factor * hub_computer_mass
        pitch_bearings_co2_emission = pitch_bearings_co2_factor * pitch_bearings_mass
        pitch_actuator_system_co2_emission = (
            pitch_actuator_co2_factor * pitch_actuator_system_mass
        )
        hub_secondary_equipment_co2_emission = (
            hub_secondary_co2_factor * hub_secondary_equipment_mass
        )
        spinner_co2_emission = spinner_co2_factor * spinner_mass
        main_shaft_co2_emission = main_shaft_co2_factor * main_shaft_mass
        main_bearings_co2_emission = main_bearings_co2_factor * main_bearings_mass
        main_bearing_housing_co2_emission = (
            main_bearing_housing_co2_factor * main_bearing_housing_mass
        )
        gearbox_co2_emission = gearbox_co2_factor * gearbox_mass
        coupling_plus_brake_system_co2_emission = (
            coupling_brake_co2_factor * coupling_plus_brake_system_mass
        )
        generator_co2_emission = generator_co2_factor * generator_mass
        cooling_co2_emission = cooling_co2_factor * cooling_mass
        power_converter_co2_emission = power_converter_co2_factor * power_converter_mass
        controller_co2_emission = controller_co2_factor * controller_mass
        bedplate_co2_emission = bedplate_co2_factor * bedplate_mass
        yaw_system_co2_emission = yaw_system_co2_factor * yaw_system_mass
        canopy_co2_emission = canopy_co2_factor * canopy_mass
        nacell_secondary_equipment_co2_emission = (
            nacelle_secondary_co2_factor * nacell_secondary_equipment_mass
        )
        tower_structure_co2_emission = tower_structure_co2_factor * tower_structure_mass
        tower_internals_co2_emission = tower_internals_co2_factor * tower_internals_mass
        power_cables_co2_emission = power_cables_co2_factor * power_cables_mass
        main_transformer_co2_emission = (
            main_transformer_co2_factor * main_transformer_mass
        )
        tower_secondary_equipment_co2_emission = (
            tower_secondary_co2_factor * tower_secondary_equipment_mass
        )

        total_co2_emission = (
            blade_co2_emission
            + hub_structure_co2_emission
            + hub_computer_co2_emission
            + pitch_bearings_co2_emission
            + pitch_actuator_system_co2_emission
            + hub_secondary_equipment_co2_emission
            + spinner_co2_emission
            + main_shaft_co2_emission
            + main_bearings_co2_emission
            + main_bearing_housing_co2_emission
            + gearbox_co2_emission
            + coupling_plus_brake_system_co2_emission
            + generator_co2_emission
            + cooling_co2_emission
            + power_converter_co2_emission
            + controller_co2_emission
            + bedplate_co2_emission
            + yaw_system_co2_emission
            + canopy_co2_emission
            + nacell_secondary_equipment_co2_emission
            + tower_structure_co2_emission
            + tower_internals_co2_emission
            + power_cables_co2_emission
            + main_transformer_co2_emission
            + tower_secondary_equipment_co2_emission
        )

        # --- Foundation/BOP Cost ---
        foundation_cost = (
            self.CalculateFoundationCost()
        )  # Per kW based on currency conversion
        # Assuming electrical_cost is MEUR/MW -> EUR/kW
        electrical_cost_per_kw = self.electrical_cost * 1_000_000 / 1_000
        bop_cost = foundation_cost + electrical_cost_per_kw  # Cost per kW

        # --- LCOE Calculations ---
        real_wacc = (1 + self.wacc) / (1 + self.inflation) - 1
        devex_total = np.sum(self.devex * self.rated_power * 1000)  # Total farm DEVEX
        capex_turbine_tower_per_kw = sales_price_calculation / (
            self.rated_power * 1000
        )  # Cost per kW
        capex_bop_per_kw = bop_cost  # Already per kW
        capex_wt_total = (
            (capex_bop_per_kw + capex_turbine_tower_per_kw) * self.rated_power * 1000
        )  # Total cost per WT
        capex_total_net = np.sum(capex_wt_total)  # Total farm CAPEX (Net)
        opex_total_annual = np.sum(
            self.opex * self.rated_power * 1000
        )  # Total annual OPEX for farm
        # abex_total = np.sum(self.abex * self.rated_power * 1000) # Not used in LCOE as abexDiscount is 0

        # AEP Calculation
        if np.isnan(self.aep).any() and np.isnan(self.capacity_factor).any():
            raise ValueError(
                "Either Capacity Factor (capacity_factor) or AEP must be provided."
            )
        if not np.isnan(self.aep).any():
            aep_wind_farm_annual = np.sum(
                self.aep
            )  # Assumes self.aep is total annual AEP per turbine
        else:  # Use capacity factor
            # Ensure capacity_factor is broadcast correctly if single value provided
            cf = (
                self.capacity_factor
                if np.size(self.capacity_factor) == self.nwt
                else np.tile(self.capacity_factor, self.nwt)
            )
            aep_wind_farm_annual = np.sum(
                cf * self.rated_power * (365 * 24)
            )  # AEP in MWh

        # Discount Factors
        discount_factor_wacc_r = [
            1 / (1 + real_wacc) ** year for year in range(-2, self.lifetime)
        ]
        discount_factor_wacc_n = [
            1 / (1 + self.wacc) ** year for year in range(-2, self.lifetime)
        ]

        # Net and Discounted AEP
        aep_net_list = jnp.array(
            [
                aep_wind_farm_annual * ((1 + self.decline_factor) ** year)
                for year in range(self.lifetime)
            ]
        )
        aep_net_total = np.sum(aep_net_list)  # Total net AEP over lifetime
        # Discount AEP relative to year 0 (start of operation)
        aep_discount_list = [
            aep_net_list[year] * (1 / (1 + real_wacc) ** year)
            for year in range(self.lifetime)
        ]
        aep_discount_total = np.sum(
            jnp.array(aep_discount_list)
        )  # Total discounted AEP over lifetime

        # Net and Discounted DEVEX
        devex_years = range(-2, 0)  # Years -2 and -1
        devex_net_list = [devex_total / len(devex_years) for _ in devex_years]
        devex_net = np.sum(jnp.array(devex_net_list))
        devex_discount_list = [
            (devex_total / len(devex_years)) * discount_factor_wacc_n[indx]
            for indx, _ in enumerate(devex_years)
        ]
        devex_discount = np.sum(jnp.array(devex_discount_list))

        # Net and Discounted CAPEX
        capex_net = capex_total_net
        capex_base_year_index = 1  # Year -1 index in discount_factor_wacc_n
        capex_discount = capex_net * discount_factor_wacc_n[capex_base_year_index]

        # Net and Discounted OPEX
        opex_years = range(self.lifetime)  # Years 0 to lifetime-1
        opex_net_list = [
            opex_total_annual * ((1 + self.inflation) ** year) for year in opex_years
        ]
        opex_net = np.sum(jnp.array(opex_net_list))
        opex_base_year_index = 2  # Year 0 index in discount_factor_wacc_n
        opex_discount_list = [
            opex_net_list[year] * discount_factor_wacc_n[year + opex_base_year_index]
            for year in opex_years
        ]
        opex_discount = np.sum(jnp.array(opex_discount_list))

        # Net and Discounted ABEX (currently zero based on original logic)
        abex_net = 0.0
        abex_discount = 0.0

        # LCOE Calculation
        lcoe_numerator = devex_discount + capex_discount + opex_discount + abex_discount
        lcoe_denominator = aep_discount_total  # Use total discounted AEP
        if lcoe_denominator == 0:
            # Handle division by zero if discounted AEP is zero
            lcoe = np.inf
        else:
            # LCOE is Discounted Costs / Discounted AEP
            lcoe = lcoe_numerator / lcoe_denominator

        # --- Final Output ---

        all_outputs = {
            "production_net": aep_net_total,
            "production_discount": aep_discount_total,
            "aep_net": aep_net_total / self.lifetime,
            "aep_discount": aep_discount_total / self.lifetime,
            "devex_net": devex_net / 1e6,
            "devex_discount": devex_discount / 1e6,
            "capex_discount": capex_discount / 1e6,
            "opex_discount": opex_discount / 1e6,
            "co2_emission_per_wt": total_co2_emission,
            "cost_per_wt": total_cost_calculation / 1e6,
            "lcoe": lcoe,
            "capex": capex_net / 1e6,
            "opex": opex_net / 1e6 / self.lifetime,
        }

        capex_value = all_outputs.pop("capex")
        opex_value = all_outputs.pop("opex")
        self._details = all_outputs

        return CostOutput(capex=capex_value, opex=opex_value)

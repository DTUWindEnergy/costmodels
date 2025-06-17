from costmodels._interface import CostModel, CostOutput, cost_input_dataclass


@cost_input_dataclass
class PowerToHydrogenCostInput:
    """Input parameters for :class:`PowerToHydrogenCostModel`.

    All values are numeric and unitless. Costs are expressed in EUR.
    """

    electrolyzer_capacity: float  # MW
    hydrogen_storage_capacity: float  # kg
    mean_hydrogen_offtake: float  # kg
    electrolyzer_capex_cost: float = 800000.0  # EUR/MW
    electrolyzer_opex_cost: float = 16000.0  # EUR/MW/year
    electrolyzer_power_electronics_cost: float = 0.0  # EUR/MW
    water_cost: float = 4.0  # EUR/m**3
    water_treatment_cost: float = 2.0  # EUR/m**3
    water_consumption: float = 9.4  # l/kg
    storage_capex_cost: float = 300.0  # EUR/kg
    storage_opex_cost: float = 3.0  # EUR/kg/year
    transportation_cost: float = 5.0  # EUR/kg/km
    transportation_distance: float = 0.0  # km
    plant_lifetime: float = 25.0  # year
    dispatch_intervals_per_hour: float = 1.0  # 1/h


class PowerToHydrogenCostModel(CostModel):
    _inputs_cls = PowerToHydrogenCostInput

    def _run(self, inputs: PowerToHydrogenCostInput) -> CostOutput:
        yearly_intervals = 365 * 24 * inputs.dispatch_intervals_per_hour
        lifetime_dispatch_intervals = inputs.plant_lifetime * yearly_intervals
        electrolyzer_capacity = inputs.electrolyzer_capacity
        hydrogen_storage_capacity = inputs.hydrogen_storage_capacity
        mean_hydrogen_offtake = inputs.mean_hydrogen_offtake

        electrolyzer_capex_cost = inputs.electrolyzer_capex_cost
        electrolyzer_opex_cost = inputs.electrolyzer_opex_cost
        electrolyzer_power_electronics_cost = inputs.electrolyzer_power_electronics_cost
        water_cost = inputs.water_cost
        water_treatment_cost = inputs.water_treatment_cost
        water_consumption = inputs.water_consumption / 1e3  # convert l/kg to m**3/kg
        storage_capex_cost = inputs.storage_capex_cost
        storage_opex_cost = inputs.storage_opex_cost
        transportation_cost = inputs.transportation_cost
        transportation_distance = inputs.transportation_distance

        CAPEX = (
            electrolyzer_capacity
            * (electrolyzer_capex_cost + electrolyzer_power_electronics_cost)
            + storage_capex_cost * hydrogen_storage_capacity
            + (
                mean_hydrogen_offtake
                * lifetime_dispatch_intervals
                * transportation_cost
                * transportation_distance
            )
        )
        water_consumption_cost = (
            mean_hydrogen_offtake
            * yearly_intervals
            * water_consumption
            * (water_cost + water_treatment_cost)
        )  # annual mean water consumption to produce hydrogen over an year
        OPEX = (
            electrolyzer_capacity * electrolyzer_opex_cost
            + storage_opex_cost * hydrogen_storage_capacity
            + water_consumption_cost
        )

        return CostOutput(capex=CAPEX / 1e6, opex=OPEX / 1e6)

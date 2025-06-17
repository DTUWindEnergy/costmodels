from costmodels._interface import CostModel, CostOutput, cost_input_dataclass


@cost_input_dataclass
class SharedCostInput:
    area: float  # km*km
    grid_capacity: float  # MW
    hpp_BOS_soft_cost: float = 119_940.0  # EUR/MW
    hpp_grid_connection_cost: float = 50_000.0  # EUR/MW
    land_cost: float = 300_000.0  # EUR/km**2


class SharedCostModel(CostModel):
    _inputs_cls = SharedCostInput

    def _run(self, inputs: SharedCostInput) -> CostOutput:
        capex = (
            inputs.hpp_BOS_soft_cost + inputs.hpp_grid_connection_cost
        ) * inputs.grid_capacity + inputs.land_cost * inputs.area
        return CostOutput(capex=capex / 1e6, opex=0.0)

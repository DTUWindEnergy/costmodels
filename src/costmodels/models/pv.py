from costmodels._interface import CostModel, CostOutput, cost_input_dataclass


@cost_input_dataclass
class PVCostInput:
    """Input parameters for :class:`PVCostModel`."""

    solar_capacity: float  # MW
    dc_ac_ratio: float = 1.5
    panel_cost: float = 1.1e5  # EUR/MW
    hardware_installation_cost: float = 1e5  # EUR/MW
    inverter_cost: float = 2e4  # EUR/MW
    fixed_onm_cost: float = 4.5e3  # EUR/MW


class PVCostModel(CostModel):
    """Simple photovoltaic plant cost model."""

    _inputs_cls = PVCostInput

    def _run(self, inputs: PVCostInput) -> CostOutput:
        capex = (
            (inputs.panel_cost + inputs.hardware_installation_cost) * inputs.dc_ac_ratio
            + inputs.inverter_cost
        ) * inputs.solar_capacity
        opex = inputs.fixed_onm_cost * inputs.solar_capacity * inputs.dc_ac_ratio

        return CostOutput(capex=capex, opex=opex)

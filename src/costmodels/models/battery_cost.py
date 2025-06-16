import jax.numpy as jnp
import numpy as np

from costmodels._interface import CostModel, CostOutput, cost_input_dataclass


@cost_input_dataclass
class BatteryCostInput:
    """Input parameters for :class:`BatteryCostModel`. All values are numeric and
    unitless. Costs are expressed in EUR.
    """

    battery_power: float  # MW
    battery_energy: float  # MWh
    state_of_health: jnp.ndarray  # %
    battery_energy_cost: float = 62000.0  # EUR/MWh
    battery_power_cost: float = 16000.0  # EUR/MW
    battery_BOP_installation_commissioning_cost: float = 80000.0  # EUR/MW
    battery_control_system_cost: float = 2250.0  # EUR/MW
    battery_energy_onm_cost: float = 0.0  # EUR/MWh
    plant_lifetime: float = 25.0  # years
    dispatch_intervals_per_hour: float = 1.0  # 1/h
    battery_price_reduction_per_year: float = 0.1


class BatteryCostModel(CostModel):
    """Simple battery cost model."""

    _inputs_cls = BatteryCostInput

    def _run(self, inputs: BatteryCostInput) -> dict[str, float]:
        # total number of dispatch intervals over the plant lifetime
        lifetime_dispatch_intervals = (
            inputs.plant_lifetime * 365 * 24 * inputs.dispatch_intervals_per_hour
        )
        age = np.arange(int(lifetime_dispatch_intervals)) / (
            lifetime_dispatch_intervals / inputs.plant_lifetime
        )

        state_of_health = np.asarray(inputs.state_of_health, dtype=float)
        ii_battery_change = np.where(
            (state_of_health > 0.99) & (np.append(1, np.diff(state_of_health)) > 0)
        )[0]
        year_new_battery = np.unique(np.floor(age[ii_battery_change]))

        factor = 1.0 - inputs.battery_price_reduction_per_year
        N_beq = np.sum([factor**iy for iy in year_new_battery])

        capex = (
            N_beq * (inputs.battery_energy_cost * inputs.battery_energy)
            + (
                inputs.battery_power_cost
                + inputs.battery_BOP_installation_commissioning_cost
                + inputs.battery_control_system_cost
            )
            * inputs.battery_power
        )

        opex = inputs.battery_energy_onm_cost * inputs.battery_energy

        return CostOutput(capex=capex / 1e6, opex=opex / 1e6)  # MEUR

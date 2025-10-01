import jax.numpy as jnp

from ..cmodel import CostInput, CostModel, CostOutput, static_field


class BatteryCostInput(CostInput):
    """Input parameters for :class:`BatteryCostModel`."""

    battery_power: float  # MW
    battery_energy: float  # MWh
    state_of_health: jnp.ndarray  # %
    battery_energy_cost: float = 62000.0  # EUR/MWh
    battery_power_cost: float = 16000.0  # EUR/MW
    battery_BOP_installation_commissioning_cost: float = 80000.0  # EUR/MW
    battery_control_system_cost: float = 2250.0  # EUR/MW
    battery_energy_onm_cost: float = 0.0  # EUR/MWh
    plant_lifetime: float = static_field(25.0)  # years
    dispatch_intervals_per_hour: float = static_field(1.0)  # n
    battery_price_reduction_per_year: float = static_field(0.1)  # %


class BatteryCostModel(CostModel):
    """Simple battery cost model."""

    _inputs_cls = BatteryCostInput

    def _run(self, inputs: BatteryCostInput) -> CostOutput:
        # total number of dispatch intervals over the plant lifetime
        lifetime_dispatch_intervals = int(
            inputs.plant_lifetime * 365 * 24 * inputs.dispatch_intervals_per_hour
        )
        age = jnp.arange(lifetime_dispatch_intervals) / (
            lifetime_dispatch_intervals / inputs.plant_lifetime
        )
        # pad with infinity to avoid out-of-bounds access
        age_padded = jnp.append(age, jnp.inf)

        state_of_health = jnp.asarray(inputs.state_of_health, dtype=float)
        battery_change = jnp.append(1, jnp.diff(state_of_health)) > 0
        ii_battery_change = jnp.where(
            (state_of_health > 0.99) & battery_change,
            size=lifetime_dispatch_intervals,
            fill_value=lifetime_dispatch_intervals,
        )[0]

        year_new_battery = jnp.floor(age_padded[ii_battery_change])
        # remove fill values from valid years
        year_new_battery = jnp.where(
            year_new_battery > inputs.plant_lifetime, -1, year_new_battery
        )

        # get unique years of battery change
        # by creating a mask of shape (plant_lifetime, plant_lifetime)
        # that is true if the year is present in the array of years
        # and then summing over the columns
        possible_years = jnp.arange(inputs.plant_lifetime)
        mask = year_new_battery[:, None] == possible_years[None, :]
        years_present = jnp.any(mask, axis=0)

        factor = 1.0 - inputs.battery_price_reduction_per_year
        N_beq = jnp.sum(jnp.where(years_present, factor**possible_years, 0.0))

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

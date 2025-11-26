import jax.numpy as jnp
from jax.scipy.special import gamma as gamma_function

from costmodels.cmodel import CostInput, CostModel, CostOutput, static_field


def _lifetime_aware_model(gammas, components, LIFETIME, capex):
    years = jnp.arange(1, LIFETIME + 1)
    n_components = len(components)

    # Pre-extract component arrays to avoid repeated list comprehensions
    betas = jnp.array([comp["beta"] for comp in components])
    failed_after_design_life = jnp.array(
        [comp["failed_after_design_life"] for comp in components]
    )
    C_replace = jnp.array([comp["C_replace"] for comp in components])
    T_replace_days = jnp.array([comp["T_replace_days"] for comp in components])
    downtime = jnp.array([comp["downtime"] for comp in components])
    cost_fraction_of_capex = jnp.array(
        [comp["cost_fraction_of_capex"] for comp in components]
    )
    number = jnp.array([comp["number"] for comp in components])
    gammas = jnp.array(gammas)

    # Vectorized eta calculation
    etas = LIFETIME / (
        (-1 * jnp.log(1 - failed_after_design_life - 0.0000001)) ** (1 / betas)
    )

    # Vectorized derived metrics
    b10s = etas * (-1 * jnp.log(1 - 0.1)) ** (1 / betas)
    mttfs = etas * gamma_function(1 + 1 / betas)

    # Vectorized Weibull calculation for all components and years at once
    # Shape: (n_components, n_years)
    Di = (gammas[:, None] * years[None, :]) / etas[:, None]
    Fi = 1 - jnp.exp(-(Di ** betas[:, None]))

    # Calculate incremental failure rates
    Fi_with_zero = jnp.concatenate([jnp.zeros((n_components, 1)), Fi[:, :-1]], axis=1)
    lambda_series_specific = Fi - Fi_with_zero

    # Vectorized cost and downtime calculations
    replace_time = lambda_series_specific * downtime[:, None] * number[:, None]
    replace_cost = (
        lambda_series_specific
        * capex
        * cost_fraction_of_capex[:, None]
        * number[:, None]
    )

    # Aggregate across components
    lambda_series = jnp.sum(lambda_series_specific, axis=0)
    opex_series = jnp.sum(lambda_series_specific * C_replace[:, None], axis=0)
    downtime_series = jnp.sum(
        lambda_series_specific * T_replace_days[:, None] * 24, axis=0
    )

    # Build result dictionaries (keep for compatibility, but minimize Python ops)
    res_dicts = []
    for i in range(n_components):
        res_dict = {
            "lambda": lambda_series_specific[i],
            "Replace Time": replace_time[i],
            "Replace Cost": replace_cost[i],
            "Total downtime [days]": jnp.sum(replace_time[i]),
            "Total downtime [life-time fraction]": jnp.sum(replace_time[i])
            / 365
            / LIFETIME,
            "Total cost [MEUR]": jnp.sum(replace_cost[i]),
            "Total cost [CAPEX/year]": jnp.sum(replace_cost[i]) / capex / LIFETIME,
        }
        res_dicts.append(res_dict)
        # Update component dict with computed values (if needed externally)
        components[i]["b10"] = b10s[i]
        components[i]["mttf"] = mttfs[i]
        components[i]["eta_ref"] = etas[i]

    # Add fixed costs
    fixed_downtime = 200.0  # hours/year
    fixed_opex = 0.05  # per year

    downtime_series = downtime_series + fixed_downtime
    opex_series = opex_series + fixed_opex

    results_dict = {
        "Year": years,
        "Downtime_hours": downtime_series,
        "OPEX_fraction_CAPEX": opex_series,
        "Failure_rate": lambda_series,
    }
    return results_dict, res_dicts


def lifetime_aware_model(
    gamma_refs, gamma_controls, components, lifetime, opex, capex, scale_to=0.03
):
    _, res_dicts_ref = _lifetime_aware_model(gamma_refs, components, lifetime, capex)
    _, res_dicts_control = _lifetime_aware_model(
        gamma_controls, components, lifetime, capex
    )

    # Vectorized extraction from result dicts
    ref_maintenance = (
        jnp.array([r["Total cost [CAPEX/year]"] for r in res_dicts_ref]) + 1e-12
    )
    control_maintenance = jnp.array(
        [r["Total cost [CAPEX/year]"] for r in res_dicts_control]
    )
    ref_downtime = (
        jnp.array([r["Total downtime [days]"] for r in res_dicts_ref]) + 1e-12
    )
    control_downtime = jnp.array(
        [r["Total downtime [days]"] for r in res_dicts_control]
    )

    results_df = {
        "Name": [c["name"] for c in components],
        "Number": [c["number"] for c in components],
        "%-failed after design life": [
            c["failed_after_design_life"] * 100 for c in components
        ],
        "Shape parameter": [c["beta"] for c in components],
        "Scale design": [c["eta_ref"] for c in components],
        "B10-life design": [c["b10"] for c in components],
        "MTTF design": [c["mttf"] for c in components],
        "Cost [% of CAPEX]": [c["cost_fraction_of_capex"] * 100 for c in components],
        "Down-time [days]": [c["downtime"] for c in components],
        "Ref maintenance": ref_maintenance,
        "New maintenance": control_maintenance,
        "Cost reduction": jnp.nan_to_num(
            1 - jnp.divide(control_maintenance, ref_maintenance)
        ),
        "Ref Down-time": ref_downtime,
        "New Down-time": control_downtime,
        "Downtime reduction": jnp.nan_to_num(
            1 - jnp.divide(control_downtime, ref_downtime)
        ),
    }
    Total_maintenance_cost_ref = jnp.sum(ref_maintenance)

    scaler = (scale_to - opex / 100) / Total_maintenance_cost_ref

    Total_maintenance_cost_ref *= scaler

    Total_ONM_cost_ref = Total_maintenance_cost_ref + opex / 100
    Total_maintenance_cost_control = jnp.sum(control_maintenance) * scaler
    Total_ONM_cost_control = Total_maintenance_cost_control + opex / 100

    return results_df, Total_ONM_cost_ref, Total_ONM_cost_control


class VariableOPEXInput(CostInput):
    gamma_refs: list
    gamma_controls: list
    lifetime: float = static_field()
    components: list = static_field()
    capex: float = static_field()
    opex: float = static_field()


class VariableOPEXModel(CostModel):
    """Minimal example cost model."""

    _inputs_cls = VariableOPEXInput

    def _run(self, inputs: VariableOPEXInput) -> CostOutput:
        _, _, Total_ONM_cost_control = lifetime_aware_model(
            inputs.gamma_refs,
            inputs.gamma_controls,
            inputs.components,
            inputs.lifetime,
            inputs.opex,
            inputs.capex,
        )

        return CostOutput(
            capex=0.0,
            opex=Total_ONM_cost_control,
        )

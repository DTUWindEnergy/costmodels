import jax
import jax.numpy as jnp

from costmodels.models.variable_opex import (
    VariableOPEXModel,
    lifetime_aware_model,
)


def test_variable_opex_model():
    # Component parameters (hardcoded for demonstration)
    _LIFETIME = 25  # years
    _base_capex_fraction = 0.2
    _base_downtime = 90  # days
    _gamma_refs = [0.988, 0.984, 0.988, 0.988, 0.988, 0.988, 0.999, 0.984, 0.984]
    _gamma_controls = [0.925, 0.902, 0.925, 0.925, 0.925, 0.925, 0.999, 0.901, 0.901]
    _CAPEX = 64.5  # MEUR
    _OPEX = 1.42  # % of CAPEX per year

    components = [
        {
            "name": "Main bearing",
            "beta": 2.5,  # shape parameter
            "C_replace": 0.45,
            "T_replace_days": 90,
            "number": 1,
            "failed_after_design_life": 0.25,
            "cost_fraction_of_capex": 3 * _base_capex_fraction,
            "downtime": 3 * _base_downtime,
        },
        {
            "name": "Blades",
            "beta": 2.5,
            "C_replace": 0.45,
            "T_replace_days": 90,
            "number": 3,
            "failed_after_design_life": 0.1,
            "cost_fraction_of_capex": 1 * _base_capex_fraction,
            "downtime": 1 * _base_downtime,
        },
        {
            "name": "Pitch bearing",
            "beta": 2.5,
            "C_replace": 0.15,
            "T_replace_days": 30,
            "number": 3,
            "failed_after_design_life": 0.1,
            "cost_fraction_of_capex": 1 * _base_capex_fraction,
            "downtime": 1 * _base_downtime,
        },
        {
            "name": "Gearbox",
            "beta": 2.5,
            "C_replace": 0.3,
            "T_replace_days": 60,
            "number": 1,
            "failed_after_design_life": 0.25,
            "cost_fraction_of_capex": 2 * _base_capex_fraction,
            "downtime": 2 * _base_downtime,
        },
        {
            "name": "Main shaft",
            "beta": 2.5,
            "C_replace": 0.3,
            "T_replace_days": 60,
            "number": 0,
            "failed_after_design_life": 0.1,
            "cost_fraction_of_capex": 3 * _base_capex_fraction,
            "downtime": 3 * _base_downtime,
        },
        {
            "name": "Yaw bearing",
            "beta": 2.5,
            "C_replace": 0.3,
            "T_replace_days": 60,
            "number": 0,
            "failed_after_design_life": 0.1,
            "cost_fraction_of_capex": 3 * _base_capex_fraction,
            "downtime": 3 * _base_downtime,
        },
        {
            "name": "Tower",
            "beta": 2.5,
            "C_replace": 0.3,
            "T_replace_days": 60,
            "number": 0,
            "failed_after_design_life": 0.01,
            "cost_fraction_of_capex": 4 * _base_capex_fraction,
            "downtime": 4 * _base_downtime,
        },
        {
            "name": "Generator",
            "beta": 2.5,
            "C_replace": 0.3,
            "T_replace_days": 60,
            "number": 1,
            "failed_after_design_life": 0.1,
            "cost_fraction_of_capex": 2 * _base_capex_fraction,
            "downtime": 2 * _base_downtime,
        },
        {
            "name": "Power converter",
            "beta": 2.5,
            "C_replace": 0.3,
            "T_replace_days": 60,
            "number": 0,
            "failed_after_design_life": 0.1,
            "cost_fraction_of_capex": 1 * _base_capex_fraction,
            "downtime": 1 * _base_downtime,
        },
    ]

    results_df, Total_ONM_cost_ref, Total_ONM_cost_control = lifetime_aware_model(
        _gamma_refs, _gamma_controls, components, _LIFETIME, _OPEX, _CAPEX
    )

    print(results_df)
    print(f"Total ONM cost (ref): {Total_ONM_cost_ref * 100:.2f} % of CAPEX per year")
    print(
        f"Total ONM cost (control): {Total_ONM_cost_control * 100:.2f} % of CAPEX per year"
    )

    cm = VariableOPEXModel(
        lifetime=_LIFETIME,
        components=components,
        capex=_CAPEX,
        opex=_OPEX,
        gamma_refs=_gamma_refs,
    )

    out = cm.run(gamma_controls=_gamma_controls)
    assert jnp.isclose(
        out.opex * 100, Total_ONM_cost_control * 100, rtol=1e-5, atol=1e-5
    )

    @jax.jit
    def objective(gamma_controls) -> jnp.ndarray:
        return cm.run(gamma_controls=gamma_controls).opex

    value, grad = jax.value_and_grad(objective)(jnp.array(_gamma_controls))

    assert jnp.isclose(value * 100, Total_ONM_cost_control * 100, rtol=1e-5, atol=1e-5)
    assert jnp.isfinite(grad).all()

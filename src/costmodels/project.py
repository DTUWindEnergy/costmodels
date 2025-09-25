from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp

from .finance import (
    FINANCE_OUT_NPV_KEY,
    LCO,
    Depreciation,
    Inflation,
    Technology,
    finances,
)


@dataclass
class Project:
    """Helper object to compute project finances."""

    technologies: list[Technology]
    product_prices: dict
    inflation: Inflation = field(default_factory=lambda: Inflation())
    depreciation: Depreciation = field(default_factory=lambda: Depreciation())
    shared_capex: float = 0.0
    tax_rate: float = 0.0
    devex: float = 0.0
    lcos: tuple[LCO] | None = None

    def __post_init__(self):
        self._compiled_npv_value_and_gradients = jax.jit(
            jax.value_and_grad(
                lambda a, b, c: self._npv(
                    productions=a, cost_model_args=b, finance_args=c
                ),
                argnums=(0, 1, 2),
                has_aux=True,
            )
        )
        self.product_prices = _jaxify_potentially_nested_dict(self.product_prices)

    def _npv(
        self, productions: dict, cost_model_args: dict, finance_args: dict
    ) -> float:
        techs = []
        for t in self.technologies:
            updated_t = t
            if t.cost_model and (
                t.name in cost_model_args or t.capex is None or t.opex is None
            ):
                cost_output = t.cost_model.run(**cost_model_args.get(t.name, {}))
                if t.capex is None:
                    updated_t = replace(updated_t, capex=cost_output.capex * 1e6)
                if t.opex is None:
                    updated_t = replace(updated_t, opex=cost_output.opex * 1e6)
            if t.name in productions:
                updated_t = replace(updated_t, production=productions[t.name])
            techs.append(updated_t)

        finance_inputs = {**finance_args}
        defaults = [
            ("shared_capex", self.shared_capex),
            ("inflation", self.inflation),
            ("depreciation", self.depreciation),
            ("tax_rate", self.tax_rate),
            ("devex", self.devex),
            ("lcos", self.lcos),
        ]
        for key, value in defaults:
            if key not in finance_inputs:
                finance_inputs[key] = value

        project_finance = finances(
            technologies=techs,
            product_prices=self.product_prices,
            **finance_inputs,
        )

        npv = project_finance.pop(FINANCE_OUT_NPV_KEY)
        return npv, project_finance

    def npv(
        self,
        productions: dict = {},
        cost_model_args: dict = {},
        finance_args: dict = {},
        return_aux: bool = False,
    ) -> float | tuple[float, dict]:
        """Return project Net Present Value for the given parameters."""

        productions = _jaxify_potentially_nested_dict(productions)
        cost_model_args = _jaxify_potentially_nested_dict(cost_model_args)
        finance_args = _jaxify_potentially_nested_dict(finance_args)

        npv, aux = self._compiled_npv_value_and_gradients(
            productions, cost_model_args, finance_args
        )[0]

        if return_aux:
            return npv, aux

        return npv

    def npv_grad(
        self,
        productions: dict = {},
        cost_model_args: dict = {},
        finance_args: dict = {},
    ) -> tuple:
        """Return NPV gradient with respect to
        cost model arguments, productions and finance arguments."""

        productions = _jaxify_potentially_nested_dict(productions)
        cost_model_args = _jaxify_potentially_nested_dict(cost_model_args)
        finance_args = _jaxify_potentially_nested_dict(finance_args)

        grads = self._compiled_npv_value_and_gradients(
            productions, cost_model_args, finance_args
        )[1]
        return tuple([g for g in grads if g])  # drop empty grads

    def npv_value_and_grad(
        self,
        productions: dict = {},
        cost_model_args: dict = {},
        finance_args: dict = {},
    ) -> tuple:
        """Return NPV value and gradient with respect to
        cost model arguments,productions and finance arguments."""

        productions = _jaxify_potentially_nested_dict(productions)
        cost_model_args = _jaxify_potentially_nested_dict(cost_model_args)
        finance_args = _jaxify_potentially_nested_dict(finance_args)

        return self._compiled_npv_value_and_gradients(
            productions, cost_model_args, finance_args
        )


def _jaxify_potentially_nested_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _jaxify_potentially_nested_dict(v)
        else:
            d[k] = jnp.asarray(v, dtype=float)
    return d

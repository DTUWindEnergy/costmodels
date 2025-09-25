from dataclasses import dataclass, field, replace

import jax

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

    def _npv(
        self, productions: dict, cost_model_args: dict, finance_args: dict
    ) -> float:
        techs = []
        for t in self.technologies:
            updated_t = t
            if t.cost_model and t.name in cost_model_args:
                cost_output = t.cost_model.run(**cost_model_args[t.name])
                if t.capex is None:
                    updated_t = replace(updated_t, capex=cost_output.capex * 1e6)
                if t.opex is None:
                    updated_t = replace(updated_t, opex=cost_output.opex * 1e6)
            if t.name in productions:
                updated_t = replace(updated_t, production=productions[t.name])
            techs.append(updated_t)

        finance_inputs = {**finance_args}
        if "shared_capex" not in finance_inputs:
            finance_inputs["shared_capex"] = self.shared_capex
        if "inflation" not in finance_inputs:
            finance_inputs["inflation"] = self.inflation
        if "depreciation" not in finance_inputs:
            finance_inputs["depreciation"] = self.depreciation
        if "tax_rate" not in finance_inputs:
            finance_inputs["tax_rate"] = self.tax_rate
        if "devex" not in finance_inputs:
            finance_inputs["devex"] = self.devex
        if "lcos" not in finance_inputs:
            finance_inputs["lcos"] = self.lcos

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
    ) -> float:  # TODO: return variable is wrong !!!
        """Return project Net Present Value for the given parameters."""

        # TODO: should make sure all inputss are jax arrays and float; no ints or lists

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
        return self._compiled_npv_value_and_gradients(
            productions, cost_model_args, finance_args
        )

from dataclasses import dataclass, field, replace

import jax

from .finance import LCO, Depreciation, Inflation, Technology, finances


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

    def _npv(self, productions: dict, cost_model_args: dict) -> float:
        techs = []
        for t in self.technologies:
            updated_t = t
            if t.cost_model and t.name in cost_model_args:
                cost_output = t.cost_model.run(**cost_model_args[t.name])
                updated_t = replace(
                    updated_t,
                    capex=cost_output.capex,
                    opex=cost_output.opex,
                )
            if t.name in productions:
                updated_t = replace(updated_t, production=productions[t.name])
            techs.append(updated_t)

        return finances(
            technologies=techs,
            product_prices=self.product_prices,
            shared_capex=self.shared_capex,
            inflation=self.inflation,
            tax_rate=self.tax_rate,
            depreciation=self.depreciation,
            devex=self.devex,
            lcos=self.lcos,
        )["NPV"]

    def npv(self, productions: dict = {}, cost_model_args: dict = {}) -> float:
        """Return project Net Present Value for the given parameters."""
        return self._npv(productions, cost_model_args)

    def npv_grad(self, productions: dict = {}, cost_model_args: dict = {}) -> tuple:
        """Return NPV gradient with respect to cost model arguments and productions."""
        if not hasattr(self, "grad_fn"):
            self.grad_fn = jax.grad(self._npv, argnums=(0, 1))
        return self.grad_fn(productions, cost_model_args)

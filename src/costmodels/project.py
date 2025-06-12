from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .finance import LCO, Depreciation, Inflation, Technology, finances


@dataclass
class Project:
    """Helper object to compute project finances."""

    technologies: list[Technology]
    product_prices: dict
    shared_capex: float = 0.0
    inflation: Inflation | None = None
    tax_rate: float = 0.0
    depreciation: Depreciation | None = None
    devex: float = 0.0
    lcos: tuple[LCO] | None = None

    def npv(self) -> float:
        """Return project Net Present Value."""
        return finances(
            technologies=self.technologies,
            product_prices=self.product_prices,
            shared_capex=self.shared_capex,
            inflation=self.inflation,
            tax_rate=self.tax_rate,
            depreciation=self.depreciation,
            devex=self.devex,
            lcos=self.lcos,
        )["NPV"]

    def npv_and_grad_production(self, tech_name: str):
        """Return NPV and its gradient w.r.t. production of ``tech_name``."""
        idx = next(i for i, t in enumerate(self.technologies) if t.name == tech_name)
        production = self.technologies[idx].production

        def objective(prod):
            techs = [
                replace(t, production=prod) if j == idx else t
                for j, t in enumerate(self.technologies)
            ]
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

        value, grad = jax.value_and_grad(objective)(jnp.asarray(production))
        return value, grad

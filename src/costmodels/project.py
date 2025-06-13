from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .finance import LCO, Depreciation, Inflation, Technology, finances


@dataclass
class Project:
    """Helper object to compute project finances."""

    technologies: list[Technology]
    product_prices: dict
    inflation: Inflation
    depreciation: Depreciation
    shared_capex: float = 0.0
    tax_rate: float = 0.0
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

    def npv_and_grad_production(self, productions: dict[str, jnp.ndarray]):
        """Return NPV and its gradient with respect to technology production.

        Parameters
        ----------
        productions:
            A mapping from technology names to production values. Gradients are
            returned for all productions in the mapping as a dictionary with the
            same keys.
        """

        def objective(prod_dict):
            techs = [
                replace(t, production=prod_dict[t.name]) if t.name in prod_dict else t
                for t in self.technologies
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

        value, grad = jax.value_and_grad(objective)(productions)
        return value, grad

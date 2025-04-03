from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import py_wake.literature
from py_wake.utils.gradients import autograd, set_vjp

from costmodels.base import CostModel
from costmodels.units import Quant


class EPriceAwareCostModel(CostModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.run = set_vjp([self._get_run_jac()])(self.run)
        # self._run = set_vjp([self._get_run_jac()])(self._run)

    @property
    def _cm_input_def(self) -> dict:
        return {
            "production": Quant(jnp.array([]), "GWh"),
            "eprice": Quant(jnp.array([]), "EUR/kWh"),
            "lifetime": 0,
            "opex": Quant(0, "EUR"),
            "capex": Quant(0, "MEUR"),
        }

    def _run(self) -> dict:
        assert len(self.production) == len(
            self.eprice
        ), "Energy production and electricity price time series must have the same length; got {} and {}".format(
            len(self.production), len(self.eprice)
        )

        irr, irr_grad = jax.value_and_grad(self.calculate_irr_jax, 0)(
            self.production.m,
            self.eprice.to("EUR/GWh").m,
            self.capex.to("EUR").m,
            self.opex.to("EUR").m,
            self.lifetime,
        )

        return {"irr": Quant(float(irr) * 100, "%"), "grad_irr": np.array(irr_grad)}

    # def _get_run_jac(self):
    #     def _f(fa, inputs): # fmt:skip
    #         print(f"{inputs=}")
    #         print(f"{fa=}")
    #         def _g(g): # fmt:skip
    #             return g * self._run()["grad_irr"]
    #         return _g
    #     return _f

    @staticmethod
    @partial(jax.jit, static_argnames=("lifetime"))
    def calculate_irr_jax(energy_values, price_values, capex, opex, lifetime):
        revenues = energy_values * price_values
        annual_revenue = jnp.sum(revenues)
        annual_cashflow = annual_revenue - opex
        cashflows = jnp.concatenate(
            [jnp.array([-capex]), jnp.ones(lifetime) * annual_cashflow]
        )
        res = jnp.roots(cashflows[::-1], strip_zeros=False)
        mask = (res.imag == 0) & (res.real > 0)

        def true_fn(args):
            res, mask = args
            rates = 1.0 / res.real - 1.0
            valid_rates = jnp.where(mask, jnp.abs(rates), jnp.inf)
            best_idx = jnp.argmin(valid_rates)
            return rates[best_idx]

        def false_fn(_):
            return jnp.nan

        return jax.lax.cond(jnp.any(mask), true_fn, false_fn, (res, mask))


if __name__ == "__main__":
    cost_model = EPriceAwareCostModel(
        opex=Quant(10, "kEUR"),
        capex=Quant(20, "MEUR"),
        lifetime=20,
    )

    NTS = 100
    # somewhat realistic and smooth electricity price time series
    eprice = np.random.normal(0.06, 0.02, NTS)
    # smoothen the time series
    eprice = np.convolve(eprice, np.ones(24) / 24, mode="same")
    # deal with boundaries
    eprice[:24] = eprice[24]
    eprice[-24:] = eprice[-24]
    # add some bias that sometimes bring the price below 0
    eprice += np.random.normal(0, 0.05, NTS)

    if 0:
        import matplotlib.pyplot as plt  # fmt:skip
        plt.plot(eprice)
        plt.xlabel("Hour")
        plt.ylabel("Electricity price (EUR/kWh)")
        plt.title("Electricity Price Time Series")
        plt.grid()
        plt.show()
        exit(0)

    res = cost_model.run(
        production=Quant(np.array([0.1, 2, 3]), "GWh"),
        eprice=Quant(np.array([0.1, 0.06, 0.07]), "EUR/kWh"),
    )
    print(res)
    import py_wake
    from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site

    site = Hornsrev1Site()
    windTurbines = V80()
    x, y = [0, 1000, 2000], [0, 0, 0]
    wfm = py_wake.literature.Niayifar_PorteAgel_2016(site, windTurbines)

    WSS = np.random.weibull(2, NTS) * 5.0
    WDD = np.zeros(NTS) + 270

    def power(x, y):
        power_ilk = wfm(
            x=x,
            y=y,
            time=True,
            ws=WSS,
            wd=WDD,
            return_simulationResult=False,
        )[2]
        power_ts = power_ilk.sum(axis=0).reshape(-1)
        return power_ts

    dpowerdx_func = autograd(power, vector_interdependence=True, argnum=0)
    dpowerdy_func = autograd(power, vector_interdependence=True, argnum=1)
    # dpowerdx = dpowerdx_func(x, y)
    # print(dpowerdx)

    def irr_obj_func(x, y):
        power_ilk = wfm(
            x=x,
            y=y,
            time=True,
            ws=WSS,
            wd=WDD,
            return_simulationResult=False,
        )[2]
        power_ts = power_ilk.sum(axis=0).reshape(-1)
        res = cost_model.run(
            production=Quant(power_ts, "kWh"),
            eprice=Quant(eprice, "EUR/kWh"),
        )
        return -res["irr"].m

    dirr_obj_func = autograd(  # gradient
        irr_obj_func, vector_interdependence=True, argnum=[0, 1]
    )
    print(dirr_obj_func(x, y))

    exit(0)

    def dirrdxdy_func(x, y):
        dpowerdx = dpowerdx_func(x, y)
        dpowerdy = dpowerdy_func(x, y)

        power_ilk = wfm(
            x=x,
            y=y,
            time=True,
            ws=WSS,
            wd=WDD,
            return_simulationResult=False,
        )[2]
        power_ts = power_ilk.sum(axis=0).reshape(-1)
        res = cost_model.run(
            production=Quant(power_ts, "kWh"),
            eprice=Quant(eprice, "EUR/kWh"),
        )
        dirrdpower = res["grad_irr"]

        dirrdx = -np.dot(dirrdpower, dpowerdx)
        dirrdy = -np.dot(dirrdpower, dpowerdy)
        return dirrdx, dirrdy

    print(irr_obj_func(x, y))
    print(dirrdxdy_func(x, y))

    # dirr_obj_func = fd(  # gradient
    #     irr_obj_func, vector_interdependence=True, argnum=[0, 1], step=1
    # )
    # dirr = dirr_obj_func(x, y)
    # print(dirr)

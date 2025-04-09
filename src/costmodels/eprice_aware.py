from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import py_wake.literature
from py_wake.utils.gradients import autograd

from costmodels.base import CostModel
from costmodels.units import Quant

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


class EPriceAwareCostModel(CostModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        capex=Quant(4, "MEUR"),
        lifetime=20,
    )

    NTS = 8760
    # somewhat realistic and smooth electricity price time series
    eprice = np.random.normal(0.06, 0.01, NTS)
    # smoothen the time series
    eprice = np.convolve(eprice, np.ones(24) / 24, mode="same")
    # deal with boundaries
    eprice[:24] = eprice[24]
    eprice[-24:] = eprice[-24]
    # add some bias that sometimes bring the price below 0
    eprice += np.random.normal(0, 0.01, NTS)

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
    NWT = 20
    x, y = np.linspace(0, 2_000, NWT), np.zeros(NWT)
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

    # dpowerdxdy_func = autograd(power, vector_interdependence=True, argnum=[0, 1])
    # dpowerdx_func = autograd(power, vector_interdependence=True, argnum=0)
    # dpowerdy_func = autograd(power, vector_interdependence=True, argnum=1)

    from autograd.extend import defvjp, primitive

    @primitive
    def power_to_irr(power_ts):
        res = cost_model.run(
            production=Quant(power_ts, "Wh"),
            eprice=Quant(eprice, "EUR/kWh"),
        )
        return res["irr"].m

    def irr_from_xy(x, y):
        power_val = power(x, y)
        return -power_to_irr(power_val)

    def grad_custom_fn(power_ts):
        grad_irr = cost_model.run(
            production=Quant(power_ts, "Wh"),
            eprice=Quant(eprice, "EUR/kWh"),
        )["grad_irr"]
        return grad_irr

    defvjp(
        power_to_irr,
        lambda _, power_ts: lambda g: g * grad_custom_fn(power_ts),
    )
    dirrdxdy_func = autograd(irr_from_xy, vector_interdependence=True, argnum=[0, 1])

    def objective(dv):
        wtx = dv[: len(dv) // 2]
        wty = dv[len(dv) // 2 :]
        return irr_from_xy(wtx, wty)

    def objective_jacobian(dv):
        wtx = dv[: len(dv) // 2]
        wty = dv[len(dv) // 2 :]
        jx, jy = dirrdxdy_func(wtx, wty)
        jac = np.array([np.atleast_2d(jx), np.atleast_2d(jy)])
        # normalize the jacobian
        jac /= np.linalg.norm(jac, axis=1, keepdims=True)
        return jac

    # print(f"Objective function value: {objective(x, y)}")

    @jax.jit
    def spacing_constraint_between_turbines(dv):
        wtx = dv[: len(dv) // 2]
        wty = dv[len(dv) // 2 :]
        min_distance = 3 * windTurbines.diameter()
        n_turbines = len(wtx)
        constraints = []
        for i in range(n_turbines):
            for j in range(i + 1, n_turbines):
                dist = jnp.sqrt((wtx[i] - wtx[j]) ** 2 + (wty[i] - wty[j]) ** 2)
                constraints.append(dist - min_distance)
        return constraints

    def convert2numpy(arr):
        return np.array(arr).astype(np.float32)

    from scipy.optimize import minimize

    x0 = np.random.rand(NWT) * 1_000
    y0 = np.random.rand(NWT) * 1_000
    res = minimize(
        objective,
        np.array([x0, y0]).reshape(-1),
        jac=objective_jacobian,
        method="SLSQP",
        options={"maxiter": 10, "disp": True, "ftol": 1e-8},
        constraints={
            "type": "ineq",
            "fun": lambda dv: convert2numpy(spacing_constraint_between_turbines(dv)),
        },
        bounds=[(-100, 2_000)] * 2 * len(x0),
    )
    print(res)

    opt_x = res.x[: len(res.x) // 2]
    opt_y = res.x[len(res.x) // 2 :]
    flow_map = (
        wfm(opt_x, opt_y, time=True, ws=[WSS[100]], wd=[WDD[100]])
        .flow_map()
        .plot_wake_map()
    )
    import matplotlib.pyplot as plt  # fmt:skip
    plt.show()

    exit()

    import topfarm
    from topfarm import TopFarmProblem
    from topfarm.constraint_components.boundary import (
        CircleBoundaryConstraint,
    )
    from topfarm.constraint_components.spacing import SpacingConstraint
    from topfarm.cost_models.cost_model_wrappers import CostModelComponent
    from topfarm.easy_drivers import (
        EasyScipyOptimizeDriver,
    )
    from topfarm.plotting import XYPlotComp

    cost_comp = CostModelComponent(
        input_keys=[topfarm.x_key, topfarm.y_key],
        n_wt=NWT,
        cost_function=objective,
        cost_gradient_function=objective_jacobian,
    )

    constraints = [
        CircleBoundaryConstraint(
            [x.max() / 2, 0.0],
            radius=x.max() + 200,
        ),
        SpacingConstraint(
            min_spacing=300,
        ),
    ]

    opt_driver = EasyScipyOptimizeDriver(
        optimizer="SLSQP",
        maxiter=100,
        disp=True,
    )

    problem = TopFarmProblem(
        design_vars={"x": x, "y": y},
        n_wt=NWT,
        constraints=constraints,
        cost_comp=cost_comp,
        driver=opt_driver,
        plot_comp=XYPlotComp(
            save_plot_per_iteration=True,
            folder_name="figures",
            plot_initial=False,
        ),
    )
    cost_comp.check_partials()
    exit()
    _, state, recorder = problem.optimize()

    exit()
    import time

    start = time.time()
    result = dirrdxdy_func(x, y)
    print("Gradient with respect to x and y:", result)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

    def dirrdxdy_func2(x, y):
        """Manual; slow but works for verification"""
        import time

        start = time.time()
        dpowerdx = dpowerdx_func(x, y)
        dpowerdy = dpowerdy_func(x, y)
        end = time.time()
        print(f"Time for dpowerdx: {end - start:.2f} seconds")

        dirrdpower = cost_model.run(
            production=Quant(power(x, y), "kWh"),
            eprice=Quant(eprice, "EUR/kWh"),
        )["grad_irr"]

        dirrdx = -np.dot(dirrdpower, dpowerdx)
        dirrdy = -np.dot(dirrdpower, dpowerdy)
        return dirrdx, dirrdy

import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gamma, gammainc

from costmodels._interface import CostModel, CostOutput, cost_input_dataclass


@cost_input_dataclass
class MinimalisticCostInput:
    Pg: float = 7.0 * 10**6
    Nturb: float = 37.0
    Area: float = 65 * 10**6
    D: float = 154.0
    depth: float = 46.75
    L: float = 8.0
    AA: jnp.ndarray = jnp.array(
        [
            7.81993787,
            6.5474264,
            6.70129293,
            8.28121347,
            9.73116453,
            9.2493092,
            6.96107307,
            3.1630036,
            3.76551013,
            4.6669416,
            4.5387168,
            7.1521344,
            7.81993787,
        ]
    )
    Prop_A: jnp.ndarray = jnp.array(
        [
            1.23102344,
            2.06216461,
            5.10379159,
            26.0662667,
            47.89597244,
            12.82908865,
            2.08892594,
            0.48890485,
            0.2342071,
            0.43738423,
            0.44136423,
            1.12090622,
            1.23102344,
        ]
    )  # Probability of A
    kw: float = 2.72
    H: float = 106.7
    CT: float = 0.75
    CP: float = 0.48
    Lref: float = 20.0
    rho: float = 1.25
    Uin: float = 4.0
    Uout: float = 25.0
    z0: float = 0.0001
    kappa: float = 0.4
    f: float = 1.2e-4 * float(jnp.exp(4.0))
    lifetime: float = 20.0


class MinimalisticCostModel(CostModel):
    """
    Python implementation of: "Sørensen, J. N., & Larsen, G. C. (2021). A
    Minimalistic Prediction Model to Determine Energy Production and Costs of
    Offshore Wind Farms. Energies, 14(2), Article 448.
    https://doi.org/10.3390/en14020448"

    Parameters:

        Pg : float, optional
            Nameplate capacity (generator power) in W. The default is 7.0*10**6.
        Nturb : float, optional
            Number of turbines. The default is 37.
        Area : float, optional
            Area of wind farm in m^^2. The default is 65*10**6.
        D : float, optional
            Rotor diameter. The default is 154.
        depth : float, optional
            Water depth. The default is 46.75.
        L : float, optional
            Distance to the shore [km]. The default is 8.
        AA : list, optional
            Weibull parameter A.
        Prop_A : list, optional
            Probability of A.
        kw : float, optional
            Weibull parameter1. The default is 2.72.
        H : float, optional
            Tower height. The default is 106.7.
        CT : reference CT
        CP : reference CP
        Lref : reference distance to shore [km]
        rho : air density [kg/m3]
        Uin : Cut-in wind speed [m/s]
        Uout : Cut-out wind speed [m/s]
        YO : Years of operation
        z0 : Roughness length [m]
        kappa : Von Karman constant
        f : Coriolis parameter at latitude 55 degrees
    """

    _inputs_cls = MinimalisticCostInput

    def _run(self, inputs: MinimalisticCostInput) -> CostOutput:
        """Run minimalistic cost model.

        Parameters
        ----------
        mspec : MinimalisticCostModelInput
            Model input specification.

        Returns
        -------
        MinimalisticCostModelOutput
            Model output specification.
        """

        A_average = jnp.sum(inputs.AA * inputs.Prop_A) / jnp.sum(inputs.Prop_A)

        CT = inputs.CT
        CP = inputs.CP
        Lref = inputs.Lref
        rho = inputs.rho
        Uin = inputs.Uin
        Uout = inputs.Uout
        YO = inputs.lifetime
        z0 = inputs.z0
        kappa = inputs.kappa
        f = inputs.f
        Pg = inputs.Pg
        Nturb = inputs.Nturb
        Area = inputs.Area
        D = inputs.D
        depth = inputs.depth
        L = inputs.L
        kw = inputs.kw
        H = inputs.H

        # Derived input data
        Ur = (8 * Pg / (jnp.pi * rho * CP * D**2)) ** (1 / 3)
        # Rated wind speed
        Gx = gamma(1 + 1 / kw)
        Uh0 = Gx * A_average
        # Mean velocity at hub height
        Nrow = 3.5 * jnp.sqrt(Nturb)
        # Number of turbines affected by the free wind
        sr = jnp.sqrt(Area) / (D * (jnp.sqrt(Nturb) - 1))
        # Mean spacing in diameters
        Ctau = jnp.pi * CT / (8 * sr * sr)
        # Wake parameter
        alpha = 1 / (Ur**3 - Uin**3)
        beta = -(Uin**3) / (Ur**3 - Uin**3)

        # Geostrophic wind speed
        def body(_, g1):
            return Uh0 * (1.0 + jnp.log(g1 / (f * H)) / jnp.log(H / z0))

        G1 = Uh0 * (1.0 + jnp.log(Uh0 / (f * H)) / jnp.log(H / z0))
        G = lax.fori_loop(0, 10, body, G1)

        # Mean velocity at hub height without wake effects
        Uh0 = G / (
            1.0
            + jnp.log(G / (f * H)) / kappa * jnp.sqrt((kappa / jnp.log(H / z0)) ** 2)
        )

        # Power without wake effects
        eta0 = (
            alpha
            * A_average**3
            * gamma(1 + 3 / kw)
            * (
                gammainc(1 + 3 / kw, (Ur / A_average) ** kw)
                - gammainc(1 + 3 / kw, (Uin / A_average) ** kw)
            )
            + beta
            * (jnp.exp(-((Uin / A_average) ** kw)) - jnp.exp(-((Ur / A_average) ** kw)))
            + jnp.exp(-((Ur / A_average) ** kw))
            - jnp.exp(-((Uout / A_average) ** kw))
        )
        # Without wake effects
        Power0 = eta0 * Pg
        # Power prodction for a single turbine without wake effects

        # Mean velocity at hub height with wake effects
        # Uh = G/( 1. + np.log(G/(f*H))/kappa*np.sqrt( Ctau+(kappa/np.log(H/z0))**2 ) );
        # Auxiliary variables
        gam = jnp.log(G / (f * H))
        delta = jnp.log(H / z0)
        eps1 = (1 + gam / delta) / (
            1 + gam / kappa * jnp.sqrt(Ctau + (kappa / delta) ** 2)
        )
        eps2 = (1 + gam / delta) / (
            1 + gam / kappa * jnp.sqrt(Ctau * (Ur / Uout) ** 3.2 + (kappa / delta) ** 2)
        )
        # Power production with wake effects
        eta = (
            alpha
            * (eps1 * A_average) ** 3
            * gamma(1 + 3 / kw)
            * (
                gammainc(1 + 3 / kw, (Ur / (eps1 * A_average)) ** kw)
                - gammainc(1 + 3 / kw, (Uin / (eps1 * A_average)) ** kw)
            )
            + beta
            * (
                jnp.exp(-((Uin / (eps1 * A_average)) ** kw))
                - jnp.exp(-((Ur / (eps1 * A_average)) ** kw))
            )
            + jnp.exp(-((Ur / (eps1 * A_average)) ** kw))
            - jnp.exp(-((Uout / (eps2 * A_average)) ** kw))
        )
        Power = eta * Pg
        # Power prodction for a single turbine
        # Pinst = Nturb*Pg/1000000; # Total installed in MW

        Cturbines = 1.25 * (-0.15 * 10**6 + 0.92 * Pg) * Nturb  # €
        Ccables = 675.0 * sr * D * (Nturb - 1.0)  # Only grid between the turbines in €
        Cfm = Nturb * Pg * (depth**2 + 100 * depth + 1500) / 7500  # In €
        Cfj = Nturb * Pg * (4.5 * depth**2 - 35 * depth + 2500) / 7500  # In €
        Css = jnp.where(depth > 35, Cfj, Cfm)
        CAPEX = (Cturbines + Css + Ccables) / (0.81 - 0.06 * L / Lref)  # In €
        Pg_ref = 10**7
        F_om = jnp.select(
            [
                Pg < 0.5 * Pg_ref,
                (Pg >= 0.5 * Pg_ref) & (Pg < Pg_ref),
                (Pg >= Pg_ref) & (Pg < 2 * Pg_ref),
                Pg >= 2 * Pg_ref,
            ],
            [
                0.86 ** (-0.5 * Pg_ref / Pg),
                1.0 - 0.325 * (Pg - Pg_ref) / Pg_ref,
                1.0 - 0.14 * (Pg - Pg_ref) / Pg_ref,
                0.86 ** (0.5 * Pg / Pg_ref),
            ],
        )

        OPEX = (
            Nturb
            * Pg
            * (
                0.106 * F_om / (Power / Power0) * eta0
                + 0.8 * (365 * 24) * 10 ** (-6) * eta * (L - Lref)
            )
        )  # OPEX €/year

        OPEXtot = OPEX * YO
        aep_Wh = Pg * (365 * 24) * ((Nturb - Nrow) * eta + Nrow * eta0)
        _ = aep_Wh

        return CostOutput(capex=CAPEX / 10**6, opex=OPEXtot / 10**6)

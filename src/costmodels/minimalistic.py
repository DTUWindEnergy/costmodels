from typing import Annotated

import numpy as np
import numpy_financial as npf
from pydantic import Field
from pydantic_pint import PydanticPintValue
from scipy.special import gamma, gammainc

from costmodels.base import CostModel
from costmodels.units import Quant, getppq


class MinimalisticCM(CostModel):
    """
    Python implementation of: "Sørensen, J. N., & Larsen, G. C. (2021). A
    Minimalistic Prediction Model to Determine Energy Production and Costs of
    Offshore Wind Farms. Energies, 14(2), Article 448.
    https://doi.org/10.3390/en14020448"
    """

    class Input(CostModel.Input):
        """Parameters:
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

        Pg: float = 7.0 * 10**6
        Nturb: float = 37
        Area: float = 65 * 10**6
        D: float = 154
        depth: float = 46.75
        L: float = 8
        AA: list = [
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
        Prop_A: list = [
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
        ]  # Probability of A
        kw: float = 2.72
        H: float = 106.7
        CT: float = 0.75
        CP: float = 0.48
        Lref: float = 20.0
        rho: float = 1.25
        Uin: float = 4.0
        Uout: float = 25.0
        YO: int = Field(default=20, gt=0)  # years of operation
        z0: float = 0.0001
        kappa: float = 0.4
        f: float = 1.2e-4 * np.exp(4.0)

    class Output(CostModel.Output):
        aep: Annotated[Quant, getppq("GWh"), Field(gt=PydanticPintValue(0, "Wh"))]

    def run(self, mispec: Input) -> Output:
        """Run minimalistic cost model.

        Parameters
        ----------
        mspec : MinimalisticCMInput
            Model input specification.

        Returns
        -------
        MinimalisticCMOutput
            Model output specification.
        """

        A_average = sum(np.asarray(mispec.AA) * np.asarray(mispec.Prop_A)) / sum(
            np.asarray(mispec.Prop_A)
        )

        CT = mispec.CT
        CP = mispec.CP
        Lref = mispec.Lref
        rho = mispec.rho
        Uin = mispec.Uin
        Uout = mispec.Uout
        YO = mispec.YO
        z0 = mispec.z0
        kappa = mispec.kappa
        f = mispec.f
        Pg = mispec.Pg
        Nturb = mispec.Nturb
        Area = mispec.Area
        D = mispec.D
        depth = mispec.depth
        L = mispec.L
        kw = mispec.kw
        H = mispec.H

        # Derived input data
        Ur = (8 * Pg / (np.pi * rho * CP * D**2)) ** (1 / 3)
        # Rated wind speed
        Gx = gamma(1 + 1 / kw)
        Uh0 = Gx * A_average
        # Mean velocity at hub height
        Nrow = 3.5 * np.sqrt(Nturb)
        # Number of turbines affected by the free wind
        sr = np.sqrt(Area) / (D * (np.sqrt(Nturb) - 1))
        # Mean spacing in diameters
        Ctau = np.pi * CT / (8 * sr * sr)
        # Wake parameter
        alpha = 1 / (Ur**3 - Uin**3)
        beta = -(Uin**3) / (Ur**3 - Uin**3)

        # Geostrophic wind speed
        nmax = 10
        eps = 10 ** (-5)
        n = 0
        G1 = Uh0 * (1.0 + np.log(Uh0 / (f * H)) / np.log(H / z0))
        # first guess
        dG = np.abs(Uh0 - G1)
        while n < nmax and dG > eps:
            n = n + 1
            G2 = Uh0 * (1.0 + np.log(G1 / (f * H)) / np.log(H / z0))
            dG = np.abs(G2 - G1)
            G1 = G2
        G = G1

        # Mean velocity at hub height without wake effects
        Uh0 = G / (
            1.0 + np.log(G / (f * H)) / kappa * np.sqrt((kappa / np.log(H / z0)) ** 2)
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
            * (np.exp(-((Uin / A_average) ** kw)) - np.exp(-((Ur / A_average) ** kw)))
            + np.exp(-((Ur / A_average) ** kw))
            - np.exp(-((Uout / A_average) ** kw))
        )
        # Without wake effects
        Power0 = eta0 * Pg
        # Power prodction for a single turbine without wake effects

        # Mean velocity at hub height with wake effects
        # Uh = G/( 1. + np.log(G/(f*H))/kappa*np.sqrt( Ctau+(kappa/np.log(H/z0))**2 ) );
        # Auxiliary variables
        gam = np.log(G / (f * H))
        delta = np.log(H / z0)
        eps1 = (1 + gam / delta) / (
            1 + gam / kappa * np.sqrt(Ctau + (kappa / delta) ** 2)
        )
        eps2 = (1 + gam / delta) / (
            1 + gam / kappa * np.sqrt(Ctau * (Ur / Uout) ** 3.2 + (kappa / delta) ** 2)
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
                np.exp(-((Uin / (eps1 * A_average)) ** kw))
                - np.exp(-((Ur / (eps1 * A_average)) ** kw))
            )
            + np.exp(-((Ur / (eps1 * A_average)) ** kw))
            - np.exp(-((Uout / (eps2 * A_average)) ** kw))
        )
        Power = eta * Pg
        # Power prodction for a single turbine
        # Pinst = Nturb*Pg/1000000; # Total installed in MW

        Cturbines = 1.25 * (-0.15 * 10**6 + 0.92 * Pg) * Nturb  # €
        Ccables = 675.0 * sr * D * (Nturb - 1.0)  # Only grid between the turbines in €
        Cfm = Nturb * Pg * (depth**2 + 100 * depth + 1500) / 7500  # In €
        Cfj = Nturb * Pg * (4.5 * depth**2 - 35 * depth + 2500) / 7500  # I n€
        Css = Cfm
        if depth > 35:
            Css = Cfj
        CAPEX = (Cturbines + Css + Ccables) / (0.81 - 0.06 * L / Lref)  # In €
        Pg_ref = 10**7
        if Pg < 0.5 * Pg_ref:
            F_om = 0.86 ** (-0.5 * Pg_ref / Pg)
        if Pg >= 0.5 * Pg_ref and Pg < Pg_ref:
            F_om = 1.0 - 0.325 * (Pg - Pg_ref) / Pg_ref
        if Pg >= Pg_ref and Pg < 2 * Pg_ref:
            F_om = 1.0 - 0.14 * (Pg - Pg_ref) / Pg_ref
        if Pg >= 2 * Pg_ref:
            F_om = 0.86 ** (0.5 * Pg / Pg_ref)

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

        annual_revenue = aep_Wh * (mispec.eprice.magnitude / 1e3)
        annual_cashflow = annual_revenue - OPEX
        cashflows = [-CAPEX] + [
            annual_cashflow
            * ((1 + mispec.inflation.to_base_units().magnitude) ** (year - 1))
            for year in range(1, int(YO + 1))
        ]

        return self.Output(
            capex=Quant(CAPEX / 10**6, "MEUR"),
            opex=Quant(OPEXtot / 10**6, "MEUR"),
            aep=Quant(aep_Wh / 10**9, "GWh"),
            lcoe=Quant((CAPEX + OPEX * YO) / (YO * aep_Wh / 10**6), "EUR/MWh"),
            irr=Quant(npf.irr(cashflows) * 100, "%"),
            npv=Quant(
                npf.npv(mispec.inflation.to_base_units().magnitude, cashflows), "MEUR"
            ),
        )


if __name__ == "__main__":
    mcm = MinimalisticCM()

    cmi = mcm.Input(
        eprice=Quant(0.2, "EUR/kWh"),
        inflation=Quant(8, "%"),
    )
    grad = mcm.grad(cmi, "lcoe", ("depth", "Area"))
    print(grad)

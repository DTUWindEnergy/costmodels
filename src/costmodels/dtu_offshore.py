from enum import Enum
from functools import cache

import numpy as np

from costmodels.base import CostModel
from costmodels.units import Quant


class Foundation(Enum):
    NONE = 0
    MONOPILE = 1
    GRAVITY = 2
    JACKET = 3
    FLOATING_MOCKUP = 4


class DTUOffshoreCostModel(CostModel):
    """
    Parameters:
        rated_power: Rated power of the wind turbine.
        rotor_speed: Speed of the rotor.
        rotor_diameter: Diameter of the rotor.
        hub_height: Height of the tower.
        foundation_option: Option for foundation (0: none, 1: monopile, 2: gravity, 3: jacket, 4: floating mockup)
        water_depth: Depth of water for offshore installation.
        currency: Currency for financial calculations ('DKK', 'EURO', 'DKK/KW', 'EURO/KW')
        eur_to_dkk: Rate of change of Dkk to Euro
        wacc: Weighted Average Cost of Capital in nominal terms.
        devex (float or None): Development expenditures.
        decline_factor (float): Annual Energy Production decline factor.
        inflation (float): Inflation rate.
        lifetime (int): Project lifespan in years.
        opex (float or None): Operational expenditures.
        abex (float or None): Asset-based expenditures.
        capacity_factor (float or None): Capacity factor.
        profit: Profit margin.
        AEP (float, array): Annual Energy Production, from Pywake.
        nwt (int or None): Number of wind turbines.
        electrical_cost (int): Electrical infrastructure cost in MEURO/MW
    """

    @property
    def _cm_input_def(self) -> dict:
        return {
            "rated_power": Quant(np.nan, "MW"),
            "rotor_speed": Quant(np.nan, "rpm"),
            "rotor_diameter": Quant(np.nan, "m"),
            "hub_height": Quant(np.nan, "m"),
            "foundation_option": Foundation.MONOPILE,
            "profit": Quant(np.nan, "%"),
            "capacity_factor": Quant(np.nan, "%"),
            "decline_factor": Quant(np.nan, "%"),
            "nwt": np.nan,
            "wacc": Quant(np.nan, "%"),
            "devex": Quant(np.nan, "EUR/kW"),
            "water_depth": Quant(np.nan, "m"),
            "abex": Quant(np.nan, "EUR"),
            "electrical_cost": Quant(np.nan, "MEUR/MW"),
            "currency": "EURO/KW",
            "eur_to_dkk": 7.54,
            "aep": Quant(np.nan, "MWh"),
            "lifetime": np.nan,
            "inflation": Quant(np.nan, "%"),
            "opex": Quant(np.nan, "EUR/kW"),
            "eprice": Quant(np.nan, "EUR/kWh"),
        }

    @cache
    def RotorTorque(self):
        """
        Calculate and return rotor torque in Mega Newton-meters (MNm).
        Returns:
            np.ndarray or float: Rotor torque in MNm.
        """
        rotor_torque = 1.1 * 60 * self.rated_power / (2 * np.pi * self.rotor_speed)
        return rotor_torque

    @cache
    def RotorArea(self):
        """
        Calculate and return rotor area in square meters (m²).
        Returns:
            np.ndarray or float: Rotor area in m².
        """
        rotor_area = np.pi * (self.rotor_diameter / 2) ** 2
        return rotor_area

    @cache
    def SpecificPower(self):
        """
        Calculate and return specific power in W/m².
        Returns:
            np.ndarray or float: Specific power in W/m².
        """
        rotor_area = self.RotorArea()
        specific_power = 1_000_000 * self.rated_power / rotor_area
        return specific_power

    @cache
    def TipSpeed(self):
        """
        Calculate and return the tip speed in meters per second (m/s).
        Returns:
            np.ndarray or float: Tip speed in m/s.
        """
        tip_speed = (self.rotor_speed / 60) * 2 * np.pi * (self.rotor_diameter / 2)
        return tip_speed

    @cache
    def TotalBladeMass(
        self, mass_coeff=1.65, mass_intercept=0.0, user_exp=2.5
    ) -> float:
        """Calculate the total blade mass."""
        blade_mass = (
            mass_coeff * ((self.rotor_diameter / 2) ** user_exp) + mass_intercept
        )
        return blade_mass

    @cache
    def HubStructureMass(
        self, mass_coeff=0.5, mass_intercept=6000.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the hub structure."""
        hubstructure_mass = (
            mass_coeff * ((self.rotor_diameter / 2) ** user_exp) + mass_intercept
        )
        return hubstructure_mass

    @cache
    def HubComputerMass(
        self, mass_coeff=0.0, mass_intercept=200.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the hub computer."""
        hubcomputer_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return hubcomputer_mass

    @cache
    def PitchBearingsMass(
        self, mass_coeff=0.4, mass_intercept=500.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the pitch bearings."""
        pitchbearing_mass = (
            mass_coeff * ((self.rotor_diameter / 2) ** user_exp) + mass_intercept
        )
        return pitchbearing_mass

    @cache
    def PitchActuatorSystemMass(
        self, mass_coeff=0.15, mass_intercept=500.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the pitch actuator system."""
        pitch_actuatorsystem_mass = (
            mass_coeff * ((self.rotor_diameter / 2) ** user_exp) + mass_intercept
        )
        return pitch_actuatorsystem_mass

    @cache
    def HubSecondaryEquipmentMass(
        self, mass_coeff=5, mass_intercept=500.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of secondary equipment in the hub."""
        hub_secondary_equipment_mass = (
            mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        )
        return hub_secondary_equipment_mass

    @cache
    def SpinnerMass(self, mass_coeff=10.0, mass_intercept=0.0, user_exp=1.0) -> float:
        """Calculate the mass of the spinner."""
        spinner_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return spinner_mass

    @cache
    def MainShaftMass(self, mass_coeff=0.02, mass_intercept=0.0, user_exp=2.8) -> float:
        """Calculate the mass of the main shaft."""
        mainshaft_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return mainshaft_mass

    @cache
    def MainBearingsMass(
        self, mass_coeff=0.02, mass_intercept=0.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the main bearings."""
        main_bearingsmass = (
            mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        )
        return main_bearingsmass

    @cache
    def MainBearingHousingMass(
        self, mass_coeff=0.03, mass_intercept=0.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the main bearing housing."""
        main_bearinghousing_mass = (
            mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        )
        return main_bearinghousing_mass

    @cache
    def GearboxMass(
        self, mass_coeff=12500.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the gearbox based on torque."""
        gearbox_mass = mass_coeff * (self.RotorTorque() ** user_exp) + mass_intercept
        return gearbox_mass

    @cache
    def CouplingPlusBrakeSystemMass(
        self, mass_coeff=500.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the coupling plus brake system."""
        coupling_brakesystem_mass = (
            mass_coeff * (self.rated_power**user_exp) + mass_intercept
        )
        return coupling_brakesystem_mass

    @cache
    def GeneratorMass(
        self, mass_coeff=1800.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the generator."""
        generator_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return generator_mass

    @cache
    def CoolingMass(self, mass_coeff=500.0, mass_intercept=0.0, user_exp=1.0) -> float:
        """Calculate the mass of the cooling system."""
        cooling_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return cooling_mass

    @cache
    def PowerConverterMass(
        self, mass_coeff=1000.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the power converter."""
        powerconverter_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return powerconverter_mass

    @cache
    def ControllerMass(
        self, mass_coeff=100.0, mass_intercept=200.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the controller."""
        controller_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return controller_mass

    @cache
    def BedplateMass(self, mass_coeff=1.2, mass_intercept=0.0, user_exp=2.0) -> float:
        """Calculate the mass of the bedplate."""
        bedplate_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return bedplate_mass

    @cache
    def YawSystemMass(self, mass_coeff=0.1, mass_intercept=0.0, user_exp=2.5) -> float:
        """Calculate the mass of the yaw system."""
        yawsystem_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return yawsystem_mass

    @cache
    def CanopyMass(
        self, mass_coeff=1500.0, mass_intercept=1000.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the canopy."""
        canopy_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return canopy_mass

    @cache
    def NacellSecondaryEquipmentMass(
        self, mass_coeff=1000.0, mass_intercept=1000.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of nacelle secondary equipment."""
        nacell_secondaryequipment_mass = (
            mass_coeff * (self.rated_power**user_exp) + mass_intercept
        )
        return nacell_secondaryequipment_mass

    @cache
    def TowerStructureMass(
        self, mass_coeff=0.25, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the tower structure."""
        tower_structure_mass = (
            mass_coeff * (self.hub_height * self.RotorArea()) ** user_exp
            + mass_intercept
        )
        return tower_structure_mass

    @cache
    def TowerInternalsMass(
        self, mass_coeff=100.0, mass_intercept=1000.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of tower internals."""
        tower_internals_mass = mass_coeff * (self.hub_height**user_exp) + mass_intercept
        return tower_internals_mass

    @cache
    def PowerCablesMass(
        self, mass_coeff=25.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the power cables."""
        power_cables_mass = (
            mass_coeff * ((self.rated_power * self.hub_height) ** user_exp)
            + mass_intercept
        )
        return power_cables_mass

    @cache
    def MainTransformerMass(
        self, mass_coeff=2500.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the main transformer."""
        main_transformer_mass = (
            mass_coeff * (self.rated_power**user_exp) + mass_intercept
        )
        return main_transformer_mass

    @cache
    def TowerSecondaryEquipmentMass(
        self, mass_coeff=500.0, mass_intercept=1000.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of secondary equipment in the tower."""
        tower_secondaryequipment_mass = (
            mass_coeff * (self.rated_power**user_exp) + mass_intercept
        )
        return tower_secondaryequipment_mass

    @cache
    def HubTotalMass(self) -> float:
        """Calculate the total mass of the hub by summing its components."""
        return (
            self.HubStructureMass()
            + self.PitchBearingsMass()
            + self.PitchActuatorSystemMass()
            + self.HubComputerMass()
            + self.HubSecondaryEquipmentMass()
            + self.SpinnerMass()
        )

    @cache
    def NacelleTotalMass(self) -> float:
        """Calculate the total mass of the nacelle by summing its components."""
        return (
            self.MainShaftMass()
            + self.MainBearingsMass()
            + self.MainBearingHousingMass()
            + self.GearboxMass()
            + self.CouplingPlusBrakeSystemMass()
            + self.GeneratorMass()
            + self.CoolingMass()
            + self.PowerConverterMass()
            + self.ControllerMass()
            + self.BedplateMass()
            + self.YawSystemMass()
            + self.CanopyMass()
            + self.NacellSecondaryEquipmentMass()
        )

    @cache
    def TowerTotalMass(self) -> float:
        """Calculate the total mass of the tower by summing its components."""
        return (
            self.TowerStructureMass()
            + self.TowerInternalsMass()
            + self.PowerCablesMass()
            + self.MainTransformerMass()
            + self.TowerSecondaryEquipmentMass()
        )

    @cache
    def BOMTotalMass(self) -> float:
        """Calculate the total Bill of Materials (BOM) mass by summing all component masses."""
        return (
            self.TotalBladeMass()
            + self.HubTotalMass()
            + self.NacelleTotalMass()
            + self.TowerTotalMass()
        )

    @cache
    def BladeTotalCost(self, rate=15.0) -> float:
        blade_cost = self.TotalBladeMass() * rate

        return blade_cost

    @cache
    def HubStructureCost(self, rate=2.5) -> float:
        hubstructure_cost = self.HubStructureMass() * rate

        return hubstructure_cost

    @cache
    def HubComputerCost(self, rate=50.0) -> float:
        hubcomputer_cost = self.HubComputerMass() * rate

        return hubcomputer_cost

    @cache
    def PitchBearingsCost(self, rate=8.0) -> float:
        pitchbearing_cost = self.PitchBearingsMass() * rate

        return pitchbearing_cost

    @cache
    def PitchActuatorSystemCost(self, rate=8.0) -> float:
        pitch_actuatorsystem_cost = self.PitchActuatorSystemMass() * rate

        return pitch_actuatorsystem_cost

    @cache
    def HubSecondaryEquipmentCost(self, rate=8.0) -> float:
        hub_secondary_equipment_cost = self.HubSecondaryEquipmentMass() * rate

        return hub_secondary_equipment_cost

    @cache
    def SpinnerCost(self, rate=10.0) -> float:
        spinner_cost = self.SpinnerMass() * rate

        return spinner_cost

    @cache
    def MainShaftCost(self, rate=5.0) -> float:
        mainshaft_cost = self.MainShaftMass() * rate

        return mainshaft_cost

    @cache
    def MainBearingsCost(self, rate=15.0) -> float:
        main_bearings_cost = self.MainBearingsMass() * rate

        return main_bearings_cost

    @cache
    def MainBearingHousingCost(self, rate=2.5) -> float:
        main_bearinghousing_cost = self.MainBearingHousingMass() * rate

        return main_bearinghousing_cost

    @cache
    def GearboxCost(self, rate=8.0) -> float:
        gearbox_cost = self.GearboxMass() * rate

        return gearbox_cost

    @cache
    def CouplingPlusBrakeSystemCost(self, rate=8.0) -> float:
        coupling_brakesystem_cost = self.CouplingPlusBrakeSystemMass() * rate

        return coupling_brakesystem_cost

    @cache
    def GeneratorCost(self, rate=8.0) -> float:
        generator_cost = self.GeneratorMass() * rate

        return generator_cost

    @cache
    def CoolingCost(self, rate=8.0) -> float:
        cooling_cost = self.CoolingMass() * rate

        return cooling_cost

    @cache
    def PowerConverterCost(self, rate=30.0) -> float:
        powerconverter_cost = self.PowerConverterMass() * rate

        return powerconverter_cost

    @cache
    def ControllerCost(self, rate=50.0) -> float:
        controller_cost = self.ControllerMass() * rate

        return controller_cost

    @cache
    def BedplateCost(self, rate=2.5) -> float:
        bedplate_cost = self.BedplateMass() * rate

        return bedplate_cost

    @cache
    def YawSystemCost(self, rate=6.0) -> float:
        yawsystem_cost = self.YawSystemMass() * rate

        return yawsystem_cost

    @cache
    def CanopyCost(self, rate=10.0) -> float:
        canopy_cost = self.CanopyMass() * rate

        return canopy_cost

    @cache
    def NacellSecondaryEquipmentCost(self, rate=10.0) -> float:
        nacell_secondaryequipment_cost = self.NacellSecondaryEquipmentMass() * rate

        return nacell_secondaryequipment_cost

    @cache
    def TowerStructureCost(self, rate=3.0) -> float:
        tower_structure_cost = self.TowerStructureMass() * rate

        return tower_structure_cost

    @cache
    def TowerInternalsCost(self, rate=8.0) -> float:
        tower_internals_cost = self.TowerInternalsMass() * rate

        return tower_internals_cost

    @cache
    def PowerCablesCost(self, rate=8.0) -> float:
        power_cables_cost = self.PowerCablesMass() * rate

        return power_cables_cost

    @cache
    def MainTransformerCost(self, rate=8.0) -> float:
        main_transformer_cost = self.MainTransformerMass() * rate

        return main_transformer_cost

    @cache
    def TowerSecondaryEquipmentCost(self, rate=10.0) -> float:
        tower_secondaryequipment_cost = self.TowerSecondaryEquipmentMass() * rate

        return tower_secondaryequipment_cost

    @cache
    def HubTotalCost(self) -> float:
        return (
            self.HubStructureCost()
            + self.PitchBearingsCost()
            + self.PitchActuatorSystemCost()
            + self.HubComputerCost()
            + self.HubSecondaryEquipmentCost()
            + self.SpinnerCost()
        )

    @cache
    def NacelleTotalCost(self) -> float:
        return (
            self.MainShaftCost()
            + self.MainBearingsCost()
            + self.MainBearingHousingCost()
            + self.GearboxCost()
            + self.CouplingPlusBrakeSystemCost()
            + self.GeneratorCost()
            + self.CoolingCost()
            + self.PowerConverterCost()
            + self.ControllerCost()
            + self.BedplateCost()
            + self.YawSystemCost()
            + self.CanopyCost()
            + self.NacellSecondaryEquipmentCost()
        )

    @cache
    def TowerTotalCost(self) -> float:
        return (
            self.TowerStructureCost()
            + self.TowerInternalsCost()
            + self.PowerCablesCost()
            + self.MainTransformerCost()
            + self.TowerSecondaryEquipmentCost()
        )

    @cache
    def BOMTotalCost(self) -> float:
        return (
            self.BladeTotalCost()
            + self.HubTotalCost()
            + self.NacelleTotalCost()
            + self.TowerTotalCost()
        )

    @cache
    def MaterialOverheadCost(self, cost_coeff=0.03) -> float:
        material_overhead_cost = self.BOMTotalCost() * cost_coeff

        return material_overhead_cost

    @cache
    def DirectLaborCost(self, cost_coeff=0.10) -> float:
        direct_labor_cost = self.BOMTotalCost() * cost_coeff

        return direct_labor_cost

    @cache
    def DirectProductionCost(self) -> float:
        return (
            self.BOMTotalCost() + self.MaterialOverheadCost() + self.DirectLaborCost()
        )

    @cache
    def OverheadCost(self, cost_coeff=0.05) -> float:
        material_overhead_cost = self.DirectProductionCost() * cost_coeff

        return material_overhead_cost

    @cache
    def R_and_D(self, cost_coeff=0.025) -> float:
        RD = self.DirectProductionCost() * cost_coeff

        return RD

    @cache
    def SG_and_A(self, cost_coeff=0.05) -> float:
        SGA = self.DirectProductionCost() * cost_coeff

        return SGA

    @cache
    def TotalProductionCost(self) -> float:
        return (
            self.DirectProductionCost()
            + self.OverheadCost()
            + self.R_and_D()
            + self.SG_and_A()
        )

    @cache
    def WarrantyAccrualsCost(self, cost_coeff=0.03) -> float:
        return self.TotalProductionCost() * cost_coeff

    @cache
    def FinancingCost(self, cost_coeff=0.017778) -> float:
        return self.TotalProductionCost() * cost_coeff

    @cache
    def TransportCost(
        self, cost_coeff=0.2, cost_intercept=10000.0, user_exp=1.0
    ) -> float:
        return cost_coeff * (self.BOMTotalMass() ** user_exp) + cost_intercept

    @cache
    def HarborStorageAssyCost(
        self, cost_coeff=0.0, cost_intercept=0.0, user_exp=1.0
    ) -> float:
        return cost_coeff * (self.rated_power**user_exp) + cost_intercept

    @cache
    def InstallationCommissCost(
        self, cost_coeff=0.0, cost_intercept=0.0, user_exp=1.0
    ) -> float:
        return cost_coeff * (self.rated_power**user_exp) + cost_intercept

    @cache
    def TotalAdditionalCost(self) -> float:
        return (
            self.WarrantyAccrualsCost()
            + self.FinancingCost()
            + self.TransportCost()
            + self.HarborStorageAssyCost()
            + self.InstallationCommissCost()
        )

    @cache
    def TotalCostCalculation(self) -> float:
        return self.TotalAdditionalCost() + self.TotalProductionCost()

    @cache
    def ProfitCalculation(self) -> float:
        return -(1 - 1 / (1 - self.profit)) * self.TotalCostCalculation()

    @cache
    def SalesPriceCalculation(self) -> float:
        return self.TotalCostCalculation() + self.ProfitCalculation()

    @cache
    def TotalBladeShareofSale(self) -> float:
        return self.BladeTotalCost() / self.SalesPriceCalculation()

    @cache
    def HubStructureShareofSale(self) -> float:
        return self.HubStructureCost() / self.SalesPriceCalculation()

    @cache
    def HubComputerShareofSale(self) -> float:
        return self.HubComputerCost() / self.SalesPriceCalculation()

    @cache
    def PitchBearingsShareofSale(self) -> float:
        return self.PitchBearingsCost() / self.SalesPriceCalculation()

    @cache
    def PitchActuatorSystemShareofSale(self) -> float:
        return self.PitchActuatorSystemCost() / self.SalesPriceCalculation()

    @cache
    def HubSecondaryEquipmentShareofSale(self) -> float:
        return self.HubSecondaryEquipmentCost() / self.SalesPriceCalculation()

    @cache
    def SpinnerShareofSale(self) -> float:
        return self.SpinnerCost() / self.SalesPriceCalculation()

    @cache
    def MainShaftShareofSale(self) -> float:
        return self.MainShaftCost() / self.SalesPriceCalculation()

    @cache
    def MainBearingsShareofSale(self) -> float:
        return self.MainBearingsCost() / self.SalesPriceCalculation()

    @cache
    def MainBearingHousingShareofSale(self) -> float:
        return self.MainBearingHousingCost() / self.SalesPriceCalculation()

    @cache
    def GearboxShareofSale(self) -> float:
        return self.GearboxCost() / self.SalesPriceCalculation()

    @cache
    def CouplingPlusBrakeSystemShareofSale(self) -> float:
        return self.CouplingPlusBrakeSystemCost() / self.SalesPriceCalculation()

    @cache
    def GeneratorShareofSale(self) -> float:
        return self.GeneratorCost() / self.SalesPriceCalculation()

    @cache
    def CoolingShareofSale(self) -> float:
        return self.CoolingCost() / self.SalesPriceCalculation()

    @cache
    def PowerConverterShareofSale(self) -> float:
        return self.PowerConverterCost() / self.SalesPriceCalculation()

    @cache
    def ControllerShareofSale(self) -> float:
        return self.ControllerCost() / self.SalesPriceCalculation()

    @cache
    def BedplateShareofSale(self) -> float:
        return self.BedplateCost() / self.SalesPriceCalculation()

    @cache
    def YawSystemShareofSale(self) -> float:
        return self.YawSystemCost() / self.SalesPriceCalculation()

    @cache
    def CanopyShareofSale(self) -> float:
        return self.CanopyCost() / self.SalesPriceCalculation()

    @cache
    def NacellSecondaryEquipmentShareofSale(self) -> float:
        return self.NacellSecondaryEquipmentCost() / self.SalesPriceCalculation()

    @cache
    def TowerStructureShareofSale(self) -> float:
        return self.TowerStructureCost() / self.SalesPriceCalculation()

    @cache
    def TowerInternalsShareofSale(self) -> float:
        return self.TowerInternalsCost() / self.SalesPriceCalculation()

    @cache
    def PowerCablesShareofSale(self) -> float:
        return self.PowerCablesCost() / self.SalesPriceCalculation()

    @cache
    def MainTransformerShareofSale(self) -> float:
        return self.MainTransformerCost() / self.SalesPriceCalculation()

    @cache
    def TowerSecondaryEquipmentShareofSale(self) -> float:
        return self.TowerSecondaryEquipmentCost() / self.SalesPriceCalculation()

    @cache
    def HubTotalShareofSale(self) -> float:
        return self.HubTotalCost() / self.SalesPriceCalculation()

    @cache
    def NacelleTotalShareofSale(self) -> float:
        return self.NacelleTotalCost() / self.SalesPriceCalculation()

    @cache
    def TowerTotalShareofSale(self) -> float:
        return self.TowerTotalCost() / self.SalesPriceCalculation()

    @cache
    def BOMTotalShareofSale(self) -> float:
        return self.BOMTotalCost() / self.SalesPriceCalculation()

    @cache
    def MaterialOverheadShareofSale(self) -> float:
        return self.MaterialOverheadCost() / self.SalesPriceCalculation()

    @cache
    def DirectLaborShareofSale(self) -> float:
        return self.DirectLaborCost() / self.SalesPriceCalculation()

    @cache
    def DirectProductionShareofSale(self) -> float:
        return self.DirectProductionCost() / self.SalesPriceCalculation()

    @cache
    def OverheadShareofSale(self) -> float:
        return self.OverheadCost() / self.SalesPriceCalculation()

    @cache
    def R_and_DShareofSale(self) -> float:
        return self.R_and_D() / self.SalesPriceCalculation()

    @cache
    def SG_and_AShareofSale(self) -> float:
        return self.SG_and_A() / self.SalesPriceCalculation()

    @cache
    def TotalProductionShareofSale(self) -> float:
        return self.TotalProductionCost() / self.SalesPriceCalculation()

    @cache
    def WarrantyAccrualsShareofSale(self) -> float:
        return self.WarrantyAccrualsCost() / self.SalesPriceCalculation()

    @cache
    def FinancingShareofSale(self) -> float:
        return self.FinancingCost() / self.SalesPriceCalculation()

    @cache
    def TransportShareofSale(self) -> float:
        return self.TransportCost() / self.SalesPriceCalculation()

    @cache
    def HarborStorageAssyShareofSale(self) -> float:
        return self.HarborStorageAssyCost() / self.SalesPriceCalculation()

    @cache
    def InstallationCommissShareofSale(self) -> float:
        return self.InstallationCommissCost() / self.SalesPriceCalculation()

    @cache
    def TotalShareofSale(self) -> float:
        return self.TotalCostCalculation() / self.SalesPriceCalculation()

    @cache
    def ProfitShareofSale(self) -> float:
        return self.ProfitCalculation() / self.SalesPriceCalculation()

    @cache
    def SalesShareofSale(self) -> float:
        return self.SalesPriceCalculation() / self.SalesPriceCalculation()

    @cache
    def TotalBladeShareofTPC(self) -> float:
        return self.BladeTotalCost() / self.TotalProductionCost()

    @cache
    def HubStructureShareofTPC(self) -> float:
        return self.HubStructureCost() / self.TotalProductionCost()

    @cache
    def HubComputerShareofTPC(self) -> float:
        return self.HubComputerCost() / self.TotalProductionCost()

    @cache
    def PitchBearingsShareofTPC(self) -> float:
        return self.PitchBearingsCost() / self.TotalProductionCost()

    @cache
    def PitchActuatorSystemShareofTPC(self) -> float:
        return self.PitchActuatorSystemCost() / self.TotalProductionCost()

    @cache
    def HubSecondaryEquipmentShareofTPC(self) -> float:
        return self.HubSecondaryEquipmentCost() / self.TotalProductionCost()

    @cache
    def SpinnerShareofTPC(self) -> float:
        return self.SpinnerCost() / self.TotalProductionCost()

    @cache
    def MainShaftShareofTPC(self) -> float:
        return self.MainShaftCost() / self.TotalProductionCost()

    @cache
    def MainBearingsShareofTPC(self) -> float:
        return self.MainBearingsCost() / self.TotalProductionCost()

    @cache
    def MainBearingHousingShareofTPC(self) -> float:
        return self.MainBearingHousingCost() / self.TotalProductionCost()

    @cache
    def GearboxShareofTPC(self) -> float:
        return self.GearboxCost() / self.TotalProductionCost()

    @cache
    def CouplingPlusBrakeSystemShareofTPC(self) -> float:
        return self.CouplingPlusBrakeSystemCost() / self.TotalProductionCost()

    @cache
    def GeneratorShareofTPC(self) -> float:
        return self.GeneratorCost() / self.TotalProductionCost()

    @cache
    def CoolingShareofTPC(self) -> float:
        return self.CoolingCost() / self.TotalProductionCost()

    @cache
    def PowerConverterShareofTPC(self) -> float:
        return self.PowerConverterCost() / self.TotalProductionCost()

    @cache
    def ControllerShareofTPC(self) -> float:
        return self.ControllerCost() / self.TotalProductionCost()

    @cache
    def BedplateShareofTPC(self) -> float:
        return self.BedplateCost() / self.TotalProductionCost()

    @cache
    def YawSystemShareofTPC(self) -> float:
        return self.YawSystemCost() / self.TotalProductionCost()

    @cache
    def CanopyShareofTPC(self) -> float:
        return self.CanopyCost() / self.TotalProductionCost()

    @cache
    def NacellSecondaryEquipmentShareofTPC(self) -> float:
        return self.NacellSecondaryEquipmentCost() / self.TotalProductionCost()

    @cache
    def TowerStructureShareofTPC(self) -> float:
        return self.TowerStructureCost() / self.TotalProductionCost()

    @cache
    def TowerInternalsShareofTPC(self) -> float:
        return self.TowerInternalsCost() / self.TotalProductionCost()

    @cache
    def PowerCablesShareofTPC(self) -> float:
        return self.PowerCablesCost() / self.TotalProductionCost()

    @cache
    def MainTransformerShareofTPC(self) -> float:
        return self.MainTransformerCost() / self.TotalProductionCost()

    @cache
    def TowerSecondaryEquipmentShareofTPC(self) -> float:
        return self.TowerSecondaryEquipmentCost() / self.TotalProductionCost()

    @cache
    def HubTotalShareofTPC(self) -> float:
        return self.HubTotalCost() / self.TotalProductionCost()

    @cache
    def NacelleTotalShareofTPC(self) -> float:
        return self.NacelleTotalCost() / self.TotalProductionCost()

    @cache
    def TowerTotalShareofTPC(self) -> float:
        return self.TowerTotalCost() / self.TotalProductionCost()

    @cache
    def BOMTotalShareofTPC(self) -> float:
        return self.BOMTotalCost() / self.TotalProductionCost()

    @cache
    def MaterialOverheadShareofTPC(self) -> float:
        return self.MaterialOverheadCost() / self.TotalProductionCost()

    @cache
    def DirectLaborShareofTPC(self) -> float:
        return self.DirectLaborCost() / self.TotalProductionCost()

    @cache
    def DirectProductionShareofTPC(self) -> float:
        return self.DirectProductionCost() / self.TotalProductionCost()

    @cache
    def OverheadShareofTPC(self) -> float:
        return self.OverheadCost() / self.TotalProductionCost()

    @cache
    def R_and_DShareofTPC(self) -> float:
        return self.R_and_D() / self.TotalProductionCost()

    @cache
    def SG_and_AShareofTPC(self) -> float:
        return self.SG_and_A() / self.TotalProductionCost()

    @cache
    def TotalProductionShareofTPC(self) -> float:
        return self.TotalProductionCost() / self.TotalProductionCost()

    @cache
    def BladeCo2Emission(
        self, emissionfactor=4.00
    ) -> float:  # emissionFactor  is in kg CO2/kg
        return emissionfactor * self.TotalBladeMass()

    @cache
    def HubStructureCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.HubStructureMass()

    @cache
    def HubComputerCo2Emission(self, emissionfactor=3.00) -> float:
        return emissionfactor * self.HubComputerMass()

    @cache
    def PitchBearingsCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.PitchBearingsMass()

    @cache
    def PitchActuatorSystemCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.PitchActuatorSystemMass()

    @cache
    def HubSecondaryEquipmentCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.HubSecondaryEquipmentMass()

    @cache
    def SpinnerCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.SpinnerMass()

    @cache
    def MainShaftCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.MainShaftMass()

    @cache
    def MainBearingsCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.MainBearingsMass()

    @cache
    def MainBearingHousingCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.MainBearingHousingMass()

    @cache
    def GearboxCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.GearboxMass()

    @cache
    def CouplingPlusBrakeSystemCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.CouplingPlusBrakeSystemMass()

    @cache
    def GeneratorCo2Emission(self, emissionfactor=6.00) -> float:
        return emissionfactor * self.GeneratorMass()

    @cache
    def CoolingCo2Emission(self, emissionfactor=2.00) -> float:
        return emissionfactor * self.CoolingMass()

    @cache
    def PowerConverterCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.PowerConverterMass()

    @cache
    def ControllerCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.ControllerMass()

    @cache
    def BedplateCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.BedplateMass()

    @cache
    def YawSystemCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.YawSystemMass()

    @cache
    def CanopyCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.CanopyMass()

    @cache
    def NacellSecondaryEquipmentCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.NacellSecondaryEquipmentMass()

    @cache
    def TowerStructureCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.TowerStructureMass()

    @cache
    def TowerInternalsCo2Emission(self, emissionfactor=2.00) -> float:
        return emissionfactor * self.TowerInternalsMass()

    @cache
    def PowerCablesCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.PowerCablesMass()

    @cache
    def MainTransformerCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.MainTransformerMass()

    @cache
    def TowerSecondaryEquipmentCo2Emission(self, emissionfactor=2.00) -> float:
        return emissionfactor * self.TowerSecondaryEquipmentMass()

    @cache
    def Total_Co2Emission(self) -> float:
        return (
            self.BladeCo2Emission()
            + self.HubStructureCo2Emission()
            + self.HubComputerCo2Emission()
            + self.PitchBearingsCo2Emission()
            + self.PitchActuatorSystemCo2Emission()
            + self.HubSecondaryEquipmentCo2Emission()
            + self.SpinnerCo2Emission()
            + self.MainShaftCo2Emission()
            + self.MainBearingsCo2Emission()
            + self.MainBearingHousingCo2Emission()
            + self.GearboxCo2Emission()
            + self.CouplingPlusBrakeSystemCo2Emission()
            + self.GeneratorCo2Emission()
            + self.CoolingCo2Emission()
            + self.PowerConverterCo2Emission()
            + self.ControllerCo2Emission()
            + self.BedplateCo2Emission()
            + self.YawSystemCo2Emission()
            + self.CanopyCo2Emission()
            + self.NacellSecondaryEquipmentCo2Emission()
            + self.TowerStructureCo2Emission()
            + self.TowerInternalsCo2Emission()
            + self.PowerCablesCo2Emission()
            + self.MainTransformerCo2Emission()
            + self.TowerSecondaryEquipmentCo2Emission()
        )

    @cache
    def BladeCo2EmissionShare(self) -> float:  # emissionFactor  is in kg CO2/kg
        return self.BladeCo2Emission() / self.Total_Co2Emission()

    @cache
    def HubStructureCo2EmissionShare(self) -> float:
        return self.HubStructureCo2Emission() / self.Total_Co2Emission()

    @cache
    def HubComputerCo2EmissionShare(self) -> float:
        return self.HubComputerCo2Emission() / self.Total_Co2Emission()

    @cache
    def PitchBearingsCo2EmissionShare(self) -> float:
        return self.PitchBearingsCo2Emission() / self.Total_Co2Emission()

    @cache
    def PitchActuatorSystemCo2EmissionShare(self) -> float:
        return self.PitchActuatorSystemCo2Emission() / self.Total_Co2Emission()

    @cache
    def HubSecondaryEquipmentCo2EmissionShare(self) -> float:
        return self.HubSecondaryEquipmentCo2Emission() / self.Total_Co2Emission()

    @cache
    def SpinnerCo2EmissionShare(self) -> float:
        return self.SpinnerCo2Emission() / self.Total_Co2Emission()

    @cache
    def MainShaftCo2EmissionShare(self) -> float:
        return self.MainShaftCo2Emission() / self.Total_Co2Emission()

    @cache
    def MainBearingsCo2EmissionShare(self) -> float:
        return self.MainBearingsCo2Emission() / self.Total_Co2Emission()

    @cache
    def MainBearingHousingCo2EmissionShare(self) -> float:
        return self.MainBearingHousingCo2Emission() / self.Total_Co2Emission()

    @cache
    def GearboxCo2EmissionShare(self) -> float:
        return self.GearboxCo2Emission() / self.Total_Co2Emission()

    @cache
    def CouplingPlusBrakeSystemCo2EmissionShare(self) -> float:
        return self.CouplingPlusBrakeSystemCo2Emission() / self.Total_Co2Emission()

    @cache
    def GeneratorCo2EmissionShare(self) -> float:
        return self.GeneratorCo2Emission() / self.Total_Co2Emission()

    @cache
    def CoolingCo2EmissionShare(self) -> float:
        return self.CoolingCo2Emission() / self.Total_Co2Emission()

    @cache
    def PowerConverterCo2EmissionShare(self) -> float:
        return self.PowerConverterCo2Emission() / self.Total_Co2Emission()

    @cache
    def ControllerCo2EmissionShare(self) -> float:
        return self.ControllerCo2Emission() / self.Total_Co2Emission()

    @cache
    def BedplateCo2EmissionShare(self) -> float:
        return self.BedplateCo2Emission() / self.Total_Co2Emission()

    @cache
    def YawSystemCo2EmissionShare(self) -> float:
        return self.YawSystemCo2Emission() / self.Total_Co2Emission()

    @cache
    def CanopyCo2EmissionShare(self) -> float:
        return self.CanopyCo2Emission() / self.Total_Co2Emission()

    @cache
    def NacellSecondaryEquipmentCo2EmissionShare(self) -> float:
        return self.NacellSecondaryEquipmentCo2Emission() / self.Total_Co2Emission()

    @cache
    def TowerStructureCo2EmissionShare(self) -> float:
        return self.TowerStructureCo2Emission() / self.Total_Co2Emission()

    @cache
    def TowerInternalsCo2EmissionShare(self) -> float:
        return self.TowerInternalsCo2Emission() / self.Total_Co2Emission()

    @cache
    def PowerCablesCo2EmissionShare(self) -> float:
        return self.PowerCablesCo2Emission() / self.Total_Co2Emission()

    @cache
    def MainTransformerCo2EmissionShare(self) -> float:
        return self.MainTransformerCo2Emission() / self.Total_Co2Emission()

    @cache
    def TowerSecondaryEquipmentCo2EmissionShare(self) -> float:
        return self.TowerSecondaryEquipmentCo2Emission() / self.Total_Co2Emission()

    @cache
    def Total_Co2EmissionShare(self) -> float:
        return self.Total_Co2Emission() / self.Total_Co2Emission()

    def convert_currency(self, foundation_cost):
        """
        Convert the foundation cost to the specified currency.
        """
        rates = {
            "DKK": 1,  # DKK to DKK is just 1
            "EURO": 1 / self.eur_to_dkk,  # Convert to EURO
            "DKK/KW": 1 / 1000,  # Convert to DKK per KW
            "EURO/KW": 1 / (self.eur_to_dkk * 1000),  # Convert to EURO per KW
        }

        # Ensure that a valid currency is provided; if not, assume no conversion (1)
        return foundation_cost * rates.get(self.currency)

    def CalculateFoundationCost(self):
        """
        Calculate the foundation cost based on the water depth and foundation option,
        and convert it to the specified currency.
        """
        # Calculate the foundation cost based on foundation option and water depth
        costs = {
            0: 0.0,
            1: 1000 * (self.water_depth**2) + 100000 * self.water_depth + 1500000,
            2: 6000 * (self.water_depth**2) - 100000 * self.water_depth + 1500000,
            3: 4000 * (self.water_depth**2) - 25000 * self.water_depth + 3000000,
            4: 1250 * self.eur_to_dkk * 1000,
        }
        foundation_cost = costs.get(
            self.foundation_option.value,
            1000 * (self.water_depth**2) + 100000 * self.water_depth + 1500000,
        )
        return self.convert_currency(foundation_cost)

    @cache
    def BOPCost(self):
        """
        Calculate the foundation cost for each water depth and convert to the selected currency.
        If a single water depth is passed, return a single value, otherwise return a list of costs.
        """
        return self.CalculateFoundationCost() + 1000 * self.electrical_cost

    @cache
    def RealWACC(self) -> float:
        return (1 + self.wacc) / (1 + self.inflation) - 1

    @cache
    def devexTotal(self) -> float:
        return np.sum(self.devex * self.rated_power * 1000)

    @cache
    def CAPEXTurbineTower(self) -> float:
        return self.SalesPriceCalculation() / self.rated_power / 1000

    @cache
    def CAPEXBOP(self) -> float:
        return self.BOPCost()

    @cache
    def CAPEXWT(self) -> float:
        return (self.CAPEXBOP() + self.CAPEXTurbineTower()) * self.rated_power * 1000

    @cache
    def CAPEXTotal(self) -> float:
        return np.sum(self.CAPEXWT())

    @cache
    def opexTotal(self) -> float:
        return np.sum(self.opex * self.rated_power * 1000)

    @cache
    def abexTotal(self) -> float:
        return np.sum(self.abex * self.rated_power * 1000)

    @cache
    def AEP_WindFarm(self) -> float:
        # Ensure either AEP or capacity_factor is provided and AEP is not NaN
        if np.isnan(self.aep).any() and np.isnan(self.capacity_factor).any():
            raise ValueError(
                "Either Capacity Factor (capacity_factor) or AEP must be provided."
            )

        if not np.isnan(self.aep).any():
            AEP_farm = np.sum(self.aep)
        elif not np.isnan(self.capacity_factor).any():
            AEP_farm = np.sum(self.capacity_factor * self.rated_power * (365 * 24))

        return AEP_farm

    @cache
    def DiscountFactor_WACC_r(self) -> list:

        discount_factor = []
        # Calculate the discount factors based on project lifetime and WACC
        for year in range(-2, self.lifetime):
            discount_factor.append(1 / (1 + self.RealWACC()) ** year)

        return discount_factor

    @cache
    def DiscountFactor_WACC_n(self) -> list:

        discount_factor = []
        # Calculate the discount factors based on project lifetime and WACC
        for year in range(-2, self.lifetime):
            discount_factor.append(1 / (1 + self.wacc) ** year)

        return discount_factor

    @cache
    def AEPNet(self) -> float:
        AEP_net = []
        for year in range(self.lifetime):
            AEP_ = self.AEP_WindFarm() * ((1 + self.decline_factor) ** year)
            AEP_net.append(AEP_)

        self.aep_net = np.sum(np.array(AEP_net))
        return self.aep_net

    @cache
    def AEPDiscount(self) -> float:
        AEP_discount = []
        for year in range(self.lifetime):
            AEP_d = (self.AEP_WindFarm() * (1 + self.decline_factor) ** year) * (
                1 / (1 + self.RealWACC()) ** year
            )
            AEP_discount.append(AEP_d)

        self.aep_discount = np.sum((np.array(AEP_discount)))
        return self.aep_discount

    @cache
    def devexNet(self) -> float:

        project_start = 0
        devex = []
        for year in range(-2, project_start):
            devex_ = self.devexTotal() / 2
            devex.append(devex_)

        self.devex = np.sum(np.array(devex))
        return self.devex

    @cache
    def devexDiscount(self) -> float:

        project_start = 0
        devex_discount = []
        discount_factors = self.DiscountFactor_WACC_n()

        for indx, year in enumerate(range(-2, project_start)):
            devex_d = (self.devexTotal() / 2) * discount_factors[indx]
            devex_discount.append(devex_d)

        self.devex_discount = np.sum((np.array(devex_discount)))

        return self.devex_discount

    @cache
    def CAPEXNet(self) -> float:
        return self.CAPEXTotal()

    @cache
    def CAPEXDiscount(
        self,
    ) -> float:  # in the excel sheet this is also called Total CAPEX
        base_yaer_indx = 1  # year = -1
        discount_factors = self.DiscountFactor_WACC_n()
        # self.CAPEX_discount = self.CAPEXTotal()*discount_factors[base_yaer_indx]

        return self.CAPEXTotal() * discount_factors[base_yaer_indx]

    @cache
    def TotalCAPEX(self) -> float:  # in the excel sheet this is also called CAPEX total
        return self.CAPEXDiscount()

    @cache
    def opexNET(self) -> float:

        opex_net = []
        for year in range(self.lifetime):
            opex_ = self.opexTotal() * ((1 + self.inflation) ** year)
            opex_net.append(opex_)

        self.opex_net = np.sum(np.array(opex_net))
        return self.opex_net

    @cache
    def opexDiscount(self) -> float:

        base_yaer_indx = 2  # year = 0
        discount_factors = self.DiscountFactor_WACC_n()
        opex_d = []
        for indx, year in enumerate(range(self.lifetime)):

            opex_ = (
                self.opexTotal() * (1 + self.inflation) ** year
            ) * discount_factors[indx + base_yaer_indx]
            opex_d.append(opex_)

        self.opex_d = np.sum(np.array(opex_d))
        return self.opex_d

    @cache
    def abexNET(self) -> float:
        return 0.0

    @cache
    def abexDiscount(self) -> float:
        return 0.0

    @cache
    def LCOENumerator(self):
        return (
            self.devexDiscount()
            + self.CAPEXDiscount()
            + self.opexDiscount()
            + self.abexDiscount()
        )

    @cache
    def LCOEDenominator(self):
        return self.AEPDiscount()

    @cache
    def LCOE(self):
        return self.LCOENumerator() / self.AEPDiscount()

    @cache
    def NVP_devex(self):
        return self.devexDiscount() / self.LCOENumerator()

    @cache
    def NVP_WT_CAPEX(self):
        Turbine_incl_tower = (
            self.SalesPriceCalculation() / np.array(self.rated_power) / 1000
        )
        return (
            self.CAPEXDiscount()
            * Turbine_incl_tower
            / (self.BOPCost() + Turbine_incl_tower)
            / self.LCOENumerator()
        )

    @cache
    def NVP_BOP_CAPEX(self):
        Turbine_incl_tower = (
            self.SalesPriceCalculation() / np.array(self.rated_power) / 1000
        )
        return (
            (self.CAPEXDiscount() * self.BOPCost())
            / (self.BOPCost() + Turbine_incl_tower)
            / self.LCOENumerator()
        )

    @cache
    def NVP_opex(self):
        return self.opexDiscount() / self.LCOENumerator()

    @cache
    def NVP_abex(self):
        return self.abexDiscount() / self.LCOENumerator()

    def reformat_input(self, **kwargs):
        nwt = kwargs["nwt"]

        for k, v in kwargs.items():
            vmag = v.m if isinstance(v, Quant) else v
            if isinstance(v, Quant) and str(v.u) == "%":
                vmag /= 100
            if (
                k
                in (
                    "rated_power",
                    "rotor_speed",
                    "rotor_diameter",
                    "hub_height",
                    "water_depth",
                )
                and np.size(v) == 1
            ):
                setattr(self, k, np.tile(vmag, nwt))
                continue
            if k in ("nwt", "lifetime"):
                setattr(self, k, int(vmag))
                continue
            if k == "decline_factor":
                vmag *= -1
            setattr(self, k, vmag)

    def _run(self):
        self.reformat_input(**self._cm_input)
        if self.nwt == 0:
            raise ValueError(
                "Number of turbines (nwt) must be provided for this calculation."
            )

        LCOE = self.LCOE()
        devexDiscount = self.devexDiscount()
        CAPEXDiscount = self.CAPEXDiscount()
        opexDiscount = self.opexDiscount()
        AEPDiscount = self.AEPDiscount()
        devexNet = self.devexNet()
        CAPEXNet = self.CAPEXNet()
        opexNet = self.opexNET()
        AEPNet = self.AEPNet()
        co2_emmisions = self.Total_Co2Emission()
        wt_cost = self.TotalCostCalculation()

        # cashflows = self.cashflows(
        #     self._cm_input["eprice"],
        #     self._cm_input["inflation"],
        #     Quant(CAPEXNet, "EUR"),
        #     Quant(opexNet, "EUR"),
        #     Quant(AEPNet / self.lifetime, "MWh"),
        #     self.lifetime,
        # )

        # clear cache for the next call; workaround to reuse
        # the many calls made to the same functions during
        # the evaluation of the model : ) ~20x faster with caching;
        # if not cleared the input for next call will be disregarded
        for f in self.__class__.__dict__.values():
            if hasattr(f, "cache_clear"):
                f.cache_clear()

        return {
            "production_net": Quant(AEPNet, "MWh"),
            "production_discount": Quant(AEPDiscount, "MWh"),
            "aep_net": Quant(AEPNet / self.lifetime, "MWh"),
            "aep_discount": Quant(AEPDiscount / self.lifetime, "MWh"),
            "devex_net": Quant(devexNet, "EUR").to("MEUR"),
            "devex_discount": Quant(devexDiscount, "EUR").to("MEUR"),
            "capex_discount": Quant(CAPEXDiscount, "EUR").to("MEUR"),
            "opex_discount": Quant(opexDiscount, "EUR").to("MEUR"),
            "co2_emission_per_wt": co2_emmisions,
            "cost_per_wt": Quant(wt_cost, "EUR").to("MEUR"),
            "lcoe": Quant(LCOE, "EUR/MWh"),
            "capex": Quant(CAPEXNet, "EUR").to("MEUR"),
            "opex": Quant(opexNet, "EUR").to("MEUR"),
            # "npv": self.npv(Quant(self.RealWACC() * 100, "%"), cashflows),
            # "irr": self.irr(cashflows),
        }

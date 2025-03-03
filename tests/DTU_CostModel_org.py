import numpy as np


class DTUOffshoreCostModel:

    def __init__(
        self,
        rated_power=7.0,
        rotor_speed=7.0,
        rotor_diameter=154.0,
        hub_height=106.7,
        foundation_option=1,
        water_depth=18,
        currency="EURO/KW",
        Dkk_per_Euro=7.54,
        wacc=0.08,
        devex=10,
        decline_factor=-0.015,
        inflation=0.03,
        project_lifetime=25,
        opex=30,
        abex=0,
        capacity_factor=0.424,
        profit=0.04,
        AEP=None,
        nwt=0,
        electrical_cost=0,
    ):
        """
        Parameters:
        rated_power: Rated power of the wind turbine.
        rotor_speed: Speed of the rotor.
        rotor_diameter: Diameter of the rotor.
        hub_height: Height of the tower.
        foundation_option: Option for foundation (0: none, 1: monopile, 2: gravity, 3: jacket, 4: floating mockup)
        water_depth: Depth of water for offshore installation.
        currency: Currency for financial calculations ('DKK', 'EURO', 'DKK/KW', 'EURO/KW')
        Dkk_per_Euro: Rate of change of Dkk to Euro
        wacc: Weighted Average Cost of Capital in nominal terms.
        devex (float or None): Development expenditures.
        decline_factor (float): Annual Energy Production decline factor.
        inflation (float): Inflation rate.
        Project_lifetime (int): Project lifespan in years.
        opex (float or None): Operational expenditures.
        abex (float or None): Asset-based expenditures.
        capacity_factor (float or None): Capacity factor.
        profit: Profit margin.
        AEP (float, array): Annual Energy Production, from Pywake.
        nwt (int or None): Number of wind turbines.
        electrical_cost (int): Electrical infrastructure cost in MEURO/MW
        """
        self.annual_hours = 8760
        # Convert all inputs to numpy arrays for consistent array operations
        self.set_inputs(
            rated_power=rated_power,
            rotor_speed=rotor_speed,
            rotor_diameter=rotor_diameter,
            hub_height=hub_height,
            foundation_option=foundation_option,
            water_depth=water_depth,
            currency=currency,
            Dkk_per_Euro=Dkk_per_Euro,
            wacc=wacc,
            devex=devex,
            decline_factor=decline_factor,
            inflation=inflation,
            project_lifetime=project_lifetime,
            opex=opex,
            abex=abex,
            capacity_factor=capacity_factor,
            profit=profit,
            AEP=AEP,
            nwt=nwt,
            electrical_cost=electrical_cost,
        )

    def set_inputs(self, **kwargs):
        if "nwt" in kwargs:
            nwt = int(kwargs["nwt"])
        else:
            nwt = self.nwt
        for k, v in kwargs.items():
            if k in [
                "rated_power",
                "rotor_speed",
                "rotor_diameter",
                "hub_height",
                "water_depth",
            ]:
                if np.size(v) == 1:
                    setattr(self, k, np.tile(v, nwt))
                else:
                    setattr(self, k, v)
            else:
                if k in ["nwt", "project_lifetime"]:
                    setattr(self, k, int(v))
                else:
                    setattr(self, k, v)

    def RotorTorque(self):
        """
        Calculate and return rotor torque in Mega Newton-meters (MNm).
        Returns:
            np.ndarray or float: Rotor torque in MNm.
        """
        rotor_torque = 1.1 * 60 * self.rated_power / (2 * np.pi * self.rotor_speed)
        return rotor_torque

    def RotorArea(self):
        """
        Calculate and return rotor area in square meters (m²).
        Returns:
            np.ndarray or float: Rotor area in m².
        """
        rotor_area = np.pi * (self.rotor_diameter / 2) ** 2
        return rotor_area

    def SpecificPower(self):
        """
        Calculate and return specific power in W/m².
        Returns:
            np.ndarray or float: Specific power in W/m².
        """
        rotor_area = self.RotorArea()
        specific_power = 1_000_000 * self.rated_power / rotor_area
        return specific_power

    def TipSpeed(self):
        """
        Calculate and return the tip speed in meters per second (m/s).
        Returns:
            np.ndarray or float: Tip speed in m/s.
        """
        tip_speed = (self.rotor_speed / 60) * 2 * np.pi * (self.rotor_diameter / 2)
        return tip_speed

    def TotalBladeMass(
        self, mass_coeff=1.65, mass_intercept=0.0, user_exp=2.5
    ) -> float:
        """Calculate the total blade mass."""
        blade_mass = (
            mass_coeff * ((self.rotor_diameter / 2) ** user_exp) + mass_intercept
        )
        return blade_mass

    def HubStructureMass(
        self, mass_coeff=0.5, mass_intercept=6000.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the hub structure."""
        hubstructure_mass = (
            mass_coeff * ((self.rotor_diameter / 2) ** user_exp) + mass_intercept
        )
        return hubstructure_mass

    def HubComputerMass(
        self, mass_coeff=0.0, mass_intercept=200.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the hub computer."""
        hubcomputer_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return hubcomputer_mass

    def PitchBearingsMass(
        self, mass_coeff=0.4, mass_intercept=500.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the pitch bearings."""
        pitchbearing_mass = (
            mass_coeff * ((self.rotor_diameter / 2) ** user_exp) + mass_intercept
        )
        return pitchbearing_mass

    def PitchActuatorSystemMass(
        self, mass_coeff=0.15, mass_intercept=500.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the pitch actuator system."""
        pitch_actuatorsystem_mass = (
            mass_coeff * ((self.rotor_diameter / 2) ** user_exp) + mass_intercept
        )
        return pitch_actuatorsystem_mass

    def HubSecondaryEquipmentMass(
        self, mass_coeff=5, mass_intercept=500.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of secondary equipment in the hub."""
        hub_secondary_equipment_mass = (
            mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        )
        return hub_secondary_equipment_mass

    def SpinnerMass(self, mass_coeff=10.0, mass_intercept=0.0, user_exp=1.0) -> float:
        """Calculate the mass of the spinner."""
        spinner_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return spinner_mass

    def MainShaftMass(self, mass_coeff=0.02, mass_intercept=0.0, user_exp=2.8) -> float:
        """Calculate the mass of the main shaft."""
        mainshaft_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return mainshaft_mass

    def MainBearingsMass(
        self, mass_coeff=0.02, mass_intercept=0.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the main bearings."""
        main_bearingsmass = (
            mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        )
        return main_bearingsmass

    def MainBearingHousingMass(
        self, mass_coeff=0.03, mass_intercept=0.0, user_exp=2.5
    ) -> float:
        """Calculate the mass of the main bearing housing."""
        main_bearinghousing_mass = (
            mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        )
        return main_bearinghousing_mass

    def GearboxMass(
        self, mass_coeff=12500.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the gearbox based on torque."""
        gearbox_mass = mass_coeff * (self.RotorTorque() ** user_exp) + mass_intercept
        return gearbox_mass

    def CouplingPlusBrakeSystemMass(
        self, mass_coeff=500.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the coupling plus brake system."""
        coupling_brakesystem_mass = (
            mass_coeff * (self.rated_power**user_exp) + mass_intercept
        )
        return coupling_brakesystem_mass

    def GeneratorMass(
        self, mass_coeff=1800.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the generator."""
        generator_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return generator_mass

    def CoolingMass(self, mass_coeff=500.0, mass_intercept=0.0, user_exp=1.0) -> float:
        """Calculate the mass of the cooling system."""
        cooling_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return cooling_mass

    def PowerConverterMass(
        self, mass_coeff=1000.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the power converter."""
        powerconverter_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return powerconverter_mass

    def ControllerMass(
        self, mass_coeff=100.0, mass_intercept=200.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the controller."""
        controller_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return controller_mass

    def BedplateMass(self, mass_coeff=1.2, mass_intercept=0.0, user_exp=2.0) -> float:
        """Calculate the mass of the bedplate."""
        bedplate_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return bedplate_mass

    def YawSystemMass(self, mass_coeff=0.1, mass_intercept=0.0, user_exp=2.5) -> float:
        """Calculate the mass of the yaw system."""
        yawsystem_mass = mass_coeff * (self.rotor_diameter**user_exp) + mass_intercept
        return yawsystem_mass

    def CanopyMass(
        self, mass_coeff=1500.0, mass_intercept=1000.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the canopy."""
        canopy_mass = mass_coeff * (self.rated_power**user_exp) + mass_intercept
        return canopy_mass

    def NacellSecondaryEquipmentMass(
        self, mass_coeff=1000.0, mass_intercept=1000.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of nacelle secondary equipment."""
        nacell_secondaryequipment_mass = (
            mass_coeff * (self.rated_power**user_exp) + mass_intercept
        )
        return nacell_secondaryequipment_mass

    def TowerStructureMass(
        self, mass_coeff=0.25, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the tower structure."""
        tower_structure_mass = (
            mass_coeff * (self.hub_height * self.RotorArea()) ** user_exp
            + mass_intercept
        )
        return tower_structure_mass

    def TowerInternalsMass(
        self, mass_coeff=100.0, mass_intercept=1000.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of tower internals."""
        tower_internals_mass = mass_coeff * (self.hub_height**user_exp) + mass_intercept
        return tower_internals_mass

    def PowerCablesMass(
        self, mass_coeff=25.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the power cables."""
        power_cables_mass = (
            mass_coeff * ((self.rated_power * self.hub_height) ** user_exp)
            + mass_intercept
        )
        return power_cables_mass

    def MainTransformerMass(
        self, mass_coeff=2500.0, mass_intercept=0.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of the main transformer."""
        main_transformer_mass = (
            mass_coeff * (self.rated_power**user_exp) + mass_intercept
        )
        return main_transformer_mass

    def TowerSecondaryEquipmentMass(
        self, mass_coeff=500.0, mass_intercept=1000.0, user_exp=1.0
    ) -> float:
        """Calculate the mass of secondary equipment in the tower."""
        tower_secondaryequipment_mass = (
            mass_coeff * (self.rated_power**user_exp) + mass_intercept
        )
        return tower_secondaryequipment_mass

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

    def TowerTotalMass(self) -> float:
        """Calculate the total mass of the tower by summing its components."""
        return (
            self.TowerStructureMass()
            + self.TowerInternalsMass()
            + self.PowerCablesMass()
            + self.MainTransformerMass()
            + self.TowerSecondaryEquipmentMass()
        )

    def BOMTotalMass(self) -> float:
        """Calculate the total Bill of Materials (BOM) mass by summing all component masses."""
        return (
            self.TotalBladeMass()
            + self.HubTotalMass()
            + self.NacelleTotalMass()
            + self.TowerTotalMass()
        )

    def BladeTotalCost(self, rate=15.0) -> float:
        blade_cost = self.TotalBladeMass() * rate

        return blade_cost

    def HubStructureCost(self, rate=2.5) -> float:
        hubstructure_cost = self.HubStructureMass() * rate

        return hubstructure_cost

    def HubComputerCost(self, rate=50.0) -> float:
        hubcomputer_cost = self.HubComputerMass() * rate

        return hubcomputer_cost

    def PitchBearingsCost(self, rate=8.0) -> float:
        pitchbearing_cost = self.PitchBearingsMass() * rate

        return pitchbearing_cost

    def PitchActuatorSystemCost(self, rate=8.0) -> float:
        pitch_actuatorsystem_cost = self.PitchActuatorSystemMass() * rate

        return pitch_actuatorsystem_cost

    def HubSecondaryEquipmentCost(self, rate=8.0) -> float:
        hub_secondary_equipment_cost = self.HubSecondaryEquipmentMass() * rate

        return hub_secondary_equipment_cost

    def SpinnerCost(self, rate=10.0) -> float:
        spinner_cost = self.SpinnerMass() * rate

        return spinner_cost

    def MainShaftCost(self, rate=5.0) -> float:
        mainshaft_cost = self.MainShaftMass() * rate

        return mainshaft_cost

    def MainBearingsCost(self, rate=15.0) -> float:
        main_bearings_cost = self.MainBearingsMass() * rate

        return main_bearings_cost

    def MainBearingHousingCost(self, rate=2.5) -> float:
        main_bearinghousing_cost = self.MainBearingHousingMass() * rate

        return main_bearinghousing_cost

    def GearboxCost(self, rate=8.0) -> float:
        gearbox_cost = self.GearboxMass() * rate

        return gearbox_cost

    def CouplingPlusBrakeSystemCost(self, rate=8.0) -> float:
        coupling_brakesystem_cost = self.CouplingPlusBrakeSystemMass() * rate

        return coupling_brakesystem_cost

    def GeneratorCost(self, rate=8.0) -> float:
        generator_cost = self.GeneratorMass() * rate

        return generator_cost

    def CoolingCost(self, rate=8.0) -> float:
        cooling_cost = self.CoolingMass() * rate

        return cooling_cost

    def PowerConverterCost(self, rate=30.0) -> float:
        powerconverter_cost = self.PowerConverterMass() * rate

        return powerconverter_cost

    def ControllerCost(self, rate=50.0) -> float:
        controller_cost = self.ControllerMass() * rate

        return controller_cost

    def BedplateCost(self, rate=2.5) -> float:
        bedplate_cost = self.BedplateMass() * rate

        return bedplate_cost

    def YawSystemCost(self, rate=6.0) -> float:
        yawsystem_cost = self.YawSystemMass() * rate

        return yawsystem_cost

    def CanopyCost(self, rate=10.0) -> float:
        canopy_cost = self.CanopyMass() * rate

        return canopy_cost

    def NacellSecondaryEquipmentCost(self, rate=10.0) -> float:
        nacell_secondaryequipment_cost = self.NacellSecondaryEquipmentMass() * rate

        return nacell_secondaryequipment_cost

    def TowerStructureCost(self, rate=3.0) -> float:
        tower_structure_cost = self.TowerStructureMass() * rate

        return tower_structure_cost

    def TowerInternalsCost(self, rate=8.0) -> float:
        tower_internals_cost = self.TowerInternalsMass() * rate

        return tower_internals_cost

    def PowerCablesCost(self, rate=8.0) -> float:
        power_cables_cost = self.PowerCablesMass() * rate

        return power_cables_cost

    def MainTransformerCost(self, rate=8.0) -> float:
        main_transformer_cost = self.MainTransformerMass() * rate

        return main_transformer_cost

    def TowerSecondaryEquipmentCost(self, rate=10.0) -> float:
        tower_secondaryequipment_cost = self.TowerSecondaryEquipmentMass() * rate

        return tower_secondaryequipment_cost

    def HubTotalCost(self) -> float:
        return (
            self.HubStructureCost()
            + self.PitchBearingsCost()
            + self.PitchActuatorSystemCost()
            + self.HubComputerCost()
            + self.HubSecondaryEquipmentCost()
            + self.SpinnerCost()
        )

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

    def TowerTotalCost(self) -> float:
        return (
            self.TowerStructureCost()
            + self.TowerInternalsCost()
            + self.PowerCablesCost()
            + self.MainTransformerCost()
            + self.TowerSecondaryEquipmentCost()
        )

    def BOMTotalCost(self) -> float:
        return (
            self.BladeTotalCost()
            + self.HubTotalCost()
            + self.NacelleTotalCost()
            + self.TowerTotalCost()
        )

    def MaterialOverheadCost(self, cost_coeff=0.03) -> float:
        material_overhead_cost = self.BOMTotalCost() * cost_coeff

        return material_overhead_cost

    def DirectLaborCost(self, cost_coeff=0.10) -> float:
        direct_labor_cost = self.BOMTotalCost() * cost_coeff

        return direct_labor_cost

    def DirectProductionCost(self) -> float:
        return (
            self.BOMTotalCost() + self.MaterialOverheadCost() + self.DirectLaborCost()
        )

    def OverheadCost(self, cost_coeff=0.05) -> float:
        material_overhead_cost = self.DirectProductionCost() * cost_coeff

        return material_overhead_cost

    def R_and_D(self, cost_coeff=0.025) -> float:
        RD = self.DirectProductionCost() * cost_coeff

        return RD

    def SG_and_A(self, cost_coeff=0.05) -> float:
        SGA = self.DirectProductionCost() * cost_coeff

        return SGA

    def TotalProductionCost(self) -> float:
        return (
            self.DirectProductionCost()
            + self.OverheadCost()
            + self.R_and_D()
            + self.SG_and_A()
        )

    def WarrantyAccrualsCost(self, cost_coeff=0.03) -> float:
        return self.TotalProductionCost() * cost_coeff

    def FinancingCost(self, cost_coeff=0.017778) -> float:
        return self.TotalProductionCost() * cost_coeff

    def TransportCost(
        self, cost_coeff=0.2, cost_intercept=10000.0, user_exp=1.0
    ) -> float:
        return cost_coeff * (self.BOMTotalMass() ** user_exp) + cost_intercept

    def HarborStorageAssyCost(
        self, cost_coeff=0.0, cost_intercept=0.0, user_exp=1.0
    ) -> float:
        return cost_coeff * (self.rated_power**user_exp) + cost_intercept

    def InstallationCommissCost(
        self, cost_coeff=0.0, cost_intercept=0.0, user_exp=1.0
    ) -> float:
        return cost_coeff * (self.rated_power**user_exp) + cost_intercept

    def TotalAdditionalCost(self) -> float:
        return (
            self.WarrantyAccrualsCost()
            + self.FinancingCost()
            + self.TransportCost()
            + self.HarborStorageAssyCost()
            + self.InstallationCommissCost()
        )

    def TotalCostCalculation(self) -> float:
        return self.TotalAdditionalCost() + self.TotalProductionCost()

    def ProfitCalculation(self) -> float:
        return -(1 - 1 / (1 - self.profit)) * self.TotalCostCalculation()

    def SalesPriceCalculation(self) -> float:
        return self.TotalCostCalculation() + self.ProfitCalculation()

    def TotalBladeShareofSale(self) -> float:
        return self.BladeTotalCost() / self.SalesPriceCalculation()

    def HubStructureShareofSale(self) -> float:
        return self.HubStructureCost() / self.SalesPriceCalculation()

    def HubComputerShareofSale(self) -> float:
        return self.HubComputerCost() / self.SalesPriceCalculation()

    def PitchBearingsShareofSale(self) -> float:
        return self.PitchBearingsCost() / self.SalesPriceCalculation()

    def PitchActuatorSystemShareofSale(self) -> float:
        return self.PitchActuatorSystemCost() / self.SalesPriceCalculation()

    def HubSecondaryEquipmentShareofSale(self) -> float:
        return self.HubSecondaryEquipmentCost() / self.SalesPriceCalculation()

    def SpinnerShareofSale(self) -> float:
        return self.SpinnerCost() / self.SalesPriceCalculation()

    def MainShaftShareofSale(self) -> float:
        return self.MainShaftCost() / self.SalesPriceCalculation()

    def MainBearingsShareofSale(self) -> float:
        return self.MainBearingsCost() / self.SalesPriceCalculation()

    def MainBearingHousingShareofSale(self) -> float:
        return self.MainBearingHousingCost() / self.SalesPriceCalculation()

    def GearboxShareofSale(self) -> float:
        return self.GearboxCost() / self.SalesPriceCalculation()

    def CouplingPlusBrakeSystemShareofSale(self) -> float:
        return self.CouplingPlusBrakeSystemCost() / self.SalesPriceCalculation()

    def GeneratorShareofSale(self) -> float:
        return self.GeneratorCost() / self.SalesPriceCalculation()

    def CoolingShareofSale(self) -> float:
        return self.CoolingCost() / self.SalesPriceCalculation()

    def PowerConverterShareofSale(self) -> float:
        return self.PowerConverterCost() / self.SalesPriceCalculation()

    def ControllerShareofSale(self) -> float:
        return self.ControllerCost() / self.SalesPriceCalculation()

    def BedplateShareofSale(self) -> float:
        return self.BedplateCost() / self.SalesPriceCalculation()

    def YawSystemShareofSale(self) -> float:
        return self.YawSystemCost() / self.SalesPriceCalculation()

    def CanopyShareofSale(self) -> float:
        return self.CanopyCost() / self.SalesPriceCalculation()

    def NacellSecondaryEquipmentShareofSale(self) -> float:
        return self.NacellSecondaryEquipmentCost() / self.SalesPriceCalculation()

    def TowerStructureShareofSale(self) -> float:
        return self.TowerStructureCost() / self.SalesPriceCalculation()

    def TowerInternalsShareofSale(self) -> float:
        return self.TowerInternalsCost() / self.SalesPriceCalculation()

    def PowerCablesShareofSale(self) -> float:
        return self.PowerCablesCost() / self.SalesPriceCalculation()

    def MainTransformerShareofSale(self) -> float:
        return self.MainTransformerCost() / self.SalesPriceCalculation()

    def TowerSecondaryEquipmentShareofSale(self) -> float:
        return self.TowerSecondaryEquipmentCost() / self.SalesPriceCalculation()

    def HubTotalShareofSale(self) -> float:
        return self.HubTotalCost() / self.SalesPriceCalculation()

    def NacelleTotalShareofSale(self) -> float:
        return self.NacelleTotalCost() / self.SalesPriceCalculation()

    def TowerTotalShareofSale(self) -> float:
        return self.TowerTotalCost() / self.SalesPriceCalculation()

    def BOMTotalShareofSale(self) -> float:
        return self.BOMTotalCost() / self.SalesPriceCalculation()

    def MaterialOverheadShareofSale(self) -> float:
        return self.MaterialOverheadCost() / self.SalesPriceCalculation()

    def DirectLaborShareofSale(self) -> float:
        return self.DirectLaborCost() / self.SalesPriceCalculation()

    def DirectProductionShareofSale(self) -> float:
        return self.DirectProductionCost() / self.SalesPriceCalculation()

    def OverheadShareofSale(self) -> float:
        return self.OverheadCost() / self.SalesPriceCalculation()

    def R_and_DShareofSale(self) -> float:
        return self.R_and_D() / self.SalesPriceCalculation()

    def SG_and_AShareofSale(self) -> float:
        return self.SG_and_A() / self.SalesPriceCalculation()

    def TotalProductionShareofSale(self) -> float:
        return self.TotalProductionCost() / self.SalesPriceCalculation()

    def WarrantyAccrualsShareofSale(self) -> float:
        return self.WarrantyAccrualsCost() / self.SalesPriceCalculation()

    def FinancingShareofSale(self) -> float:
        return self.FinancingCost() / self.SalesPriceCalculation()

    def TransportShareofSale(self) -> float:
        return self.TransportCost() / self.SalesPriceCalculation()

    def HarborStorageAssyShareofSale(self) -> float:
        return self.HarborStorageAssyCost() / self.SalesPriceCalculation()

    def InstallationCommissShareofSale(self) -> float:
        return self.InstallationCommissCost() / self.SalesPriceCalculation()

    def TotalShareofSale(self) -> float:
        return self.TotalCostCalculation() / self.SalesPriceCalculation()

    def ProfitShareofSale(self) -> float:
        return self.ProfitCalculation() / self.SalesPriceCalculation()

    def SalesShareofSale(self) -> float:
        return self.SalesPriceCalculation() / self.SalesPriceCalculation()

    def TotalBladeShareofTPC(self) -> float:
        return self.BladeTotalCost() / self.TotalProductionCost()

    def HubStructureShareofTPC(self) -> float:
        return self.HubStructureCost() / self.TotalProductionCost()

    def HubComputerShareofTPC(self) -> float:
        return self.HubComputerCost() / self.TotalProductionCost()

    def PitchBearingsShareofTPC(self) -> float:
        return self.PitchBearingsCost() / self.TotalProductionCost()

    def PitchActuatorSystemShareofTPC(self) -> float:
        return self.PitchActuatorSystemCost() / self.TotalProductionCost()

    def HubSecondaryEquipmentShareofTPC(self) -> float:
        return self.HubSecondaryEquipmentCost() / self.TotalProductionCost()

    def SpinnerShareofTPC(self) -> float:
        return self.SpinnerCost() / self.TotalProductionCost()

    def MainShaftShareofTPC(self) -> float:
        return self.MainShaftCost() / self.TotalProductionCost()

    def MainBearingsShareofTPC(self) -> float:
        return self.MainBearingsCost() / self.TotalProductionCost()

    def MainBearingHousingShareofTPC(self) -> float:
        return self.MainBearingHousingCost() / self.TotalProductionCost()

    def GearboxShareofTPC(self) -> float:
        return self.GearboxCost() / self.TotalProductionCost()

    def CouplingPlusBrakeSystemShareofTPC(self) -> float:
        return self.CouplingPlusBrakeSystemCost() / self.TotalProductionCost()

    def GeneratorShareofTPC(self) -> float:
        return self.GeneratorCost() / self.TotalProductionCost()

    def CoolingShareofTPC(self) -> float:
        return self.CoolingCost() / self.TotalProductionCost()

    def PowerConverterShareofTPC(self) -> float:
        return self.PowerConverterCost() / self.TotalProductionCost()

    def ControllerShareofTPC(self) -> float:
        return self.ControllerCost() / self.TotalProductionCost()

    def BedplateShareofTPC(self) -> float:
        return self.BedplateCost() / self.TotalProductionCost()

    def YawSystemShareofTPC(self) -> float:
        return self.YawSystemCost() / self.TotalProductionCost()

    def CanopyShareofTPC(self) -> float:
        return self.CanopyCost() / self.TotalProductionCost()

    def NacellSecondaryEquipmentShareofTPC(self) -> float:
        return self.NacellSecondaryEquipmentCost() / self.TotalProductionCost()

    def TowerStructureShareofTPC(self) -> float:
        return self.TowerStructureCost() / self.TotalProductionCost()

    def TowerInternalsShareofTPC(self) -> float:
        return self.TowerInternalsCost() / self.TotalProductionCost()

    def PowerCablesShareofTPC(self) -> float:
        return self.PowerCablesCost() / self.TotalProductionCost()

    def MainTransformerShareofTPC(self) -> float:
        return self.MainTransformerCost() / self.TotalProductionCost()

    def TowerSecondaryEquipmentShareofTPC(self) -> float:
        return self.TowerSecondaryEquipmentCost() / self.TotalProductionCost()

    def HubTotalShareofTPC(self) -> float:
        return self.HubTotalCost() / self.TotalProductionCost()

    def NacelleTotalShareofTPC(self) -> float:
        return self.NacelleTotalCost() / self.TotalProductionCost()

    def TowerTotalShareofTPC(self) -> float:
        return self.TowerTotalCost() / self.TotalProductionCost()

    def BOMTotalShareofTPC(self) -> float:
        return self.BOMTotalCost() / self.TotalProductionCost()

    def MaterialOverheadShareofTPC(self) -> float:
        return self.MaterialOverheadCost() / self.TotalProductionCost()

    def DirectLaborShareofTPC(self) -> float:
        return self.DirectLaborCost() / self.TotalProductionCost()

    def DirectProductionShareofTPC(self) -> float:
        return self.DirectProductionCost() / self.TotalProductionCost()

    def OverheadShareofTPC(self) -> float:
        return self.OverheadCost() / self.TotalProductionCost()

    def R_and_DShareofTPC(self) -> float:
        return self.R_and_D() / self.TotalProductionCost()

    def SG_and_AShareofTPC(self) -> float:
        return self.SG_and_A() / self.TotalProductionCost()

    def TotalProductionShareofTPC(self) -> float:
        return self.TotalProductionCost() / self.TotalProductionCost()

    def BladeCo2Emission(
        self, emissionfactor=4.00
    ) -> float:  # emissionFactor  is in kg CO2/kg
        return emissionfactor * self.TotalBladeMass()

    def HubStructureCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.HubStructureMass()

    def HubComputerCo2Emission(self, emissionfactor=3.00) -> float:
        return emissionfactor * self.HubComputerMass()

    def PitchBearingsCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.PitchBearingsMass()

    def PitchActuatorSystemCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.PitchActuatorSystemMass()

    def HubSecondaryEquipmentCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.HubSecondaryEquipmentMass()

    def SpinnerCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.SpinnerMass()

    def MainShaftCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.MainShaftMass()

    def MainBearingsCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.MainBearingsMass()

    def MainBearingHousingCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.MainBearingHousingMass()

    def GearboxCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.GearboxMass()

    def CouplingPlusBrakeSystemCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.CouplingPlusBrakeSystemMass()

    def GeneratorCo2Emission(self, emissionfactor=6.00) -> float:
        return emissionfactor * self.GeneratorMass()

    def CoolingCo2Emission(self, emissionfactor=2.00) -> float:
        return emissionfactor * self.CoolingMass()

    def PowerConverterCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.PowerConverterMass()

    def ControllerCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.ControllerMass()

    def BedplateCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.BedplateMass()

    def YawSystemCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.YawSystemMass()

    def CanopyCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.CanopyMass()

    def NacellSecondaryEquipmentCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.NacellSecondaryEquipmentMass()

    def TowerStructureCo2Emission(self, emissionfactor=1.83) -> float:
        return emissionfactor * self.TowerStructureMass()

    def TowerInternalsCo2Emission(self, emissionfactor=2.00) -> float:
        return emissionfactor * self.TowerInternalsMass()

    def PowerCablesCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.PowerCablesMass()

    def MainTransformerCo2Emission(self, emissionfactor=4.00) -> float:
        return emissionfactor * self.MainTransformerMass()

    def TowerSecondaryEquipmentCo2Emission(self, emissionfactor=2.00) -> float:
        return emissionfactor * self.TowerSecondaryEquipmentMass()

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

    def BladeCo2EmissionShare(self) -> float:  # emissionFactor  is in kg CO2/kg
        return self.BladeCo2Emission() / self.Total_Co2Emission()

    def HubStructureCo2EmissionShare(self) -> float:
        return self.HubStructureCo2Emission() / self.Total_Co2Emission()

    def HubComputerCo2EmissionShare(self) -> float:
        return self.HubComputerCo2Emission() / self.Total_Co2Emission()

    def PitchBearingsCo2EmissionShare(self) -> float:
        return self.PitchBearingsCo2Emission() / self.Total_Co2Emission()

    def PitchActuatorSystemCo2EmissionShare(self) -> float:
        return self.PitchActuatorSystemCo2Emission() / self.Total_Co2Emission()

    def HubSecondaryEquipmentCo2EmissionShare(self) -> float:
        return self.HubSecondaryEquipmentCo2Emission() / self.Total_Co2Emission()

    def SpinnerCo2EmissionShare(self) -> float:
        return self.SpinnerCo2Emission() / self.Total_Co2Emission()

    def MainShaftCo2EmissionShare(self) -> float:
        return self.MainShaftCo2Emission() / self.Total_Co2Emission()

    def MainBearingsCo2EmissionShare(self) -> float:
        return self.MainBearingsCo2Emission() / self.Total_Co2Emission()

    def MainBearingHousingCo2EmissionShare(self) -> float:
        return self.MainBearingHousingCo2Emission() / self.Total_Co2Emission()

    def GearboxCo2EmissionShare(self) -> float:
        return self.GearboxCo2Emission() / self.Total_Co2Emission()

    def CouplingPlusBrakeSystemCo2EmissionShare(self) -> float:
        return self.CouplingPlusBrakeSystemCo2Emission() / self.Total_Co2Emission()

    def GeneratorCo2EmissionShare(self) -> float:
        return self.GeneratorCo2Emission() / self.Total_Co2Emission()

    def CoolingCo2EmissionShare(self) -> float:
        return self.CoolingCo2Emission() / self.Total_Co2Emission()

    def PowerConverterCo2EmissionShare(self) -> float:
        return self.PowerConverterCo2Emission() / self.Total_Co2Emission()

    def ControllerCo2EmissionShare(self) -> float:
        return self.ControllerCo2Emission() / self.Total_Co2Emission()

    def BedplateCo2EmissionShare(self) -> float:
        return self.BedplateCo2Emission() / self.Total_Co2Emission()

    def YawSystemCo2EmissionShare(self) -> float:
        return self.YawSystemCo2Emission() / self.Total_Co2Emission()

    def CanopyCo2EmissionShare(self) -> float:
        return self.CanopyCo2Emission() / self.Total_Co2Emission()

    def NacellSecondaryEquipmentCo2EmissionShare(self) -> float:
        return self.NacellSecondaryEquipmentCo2Emission() / self.Total_Co2Emission()

    def TowerStructureCo2EmissionShare(self) -> float:
        return self.TowerStructureCo2Emission() / self.Total_Co2Emission()

    def TowerInternalsCo2EmissionShare(self) -> float:
        return self.TowerInternalsCo2Emission() / self.Total_Co2Emission()

    def PowerCablesCo2EmissionShare(self) -> float:
        return self.PowerCablesCo2Emission() / self.Total_Co2Emission()

    def MainTransformerCo2EmissionShare(self) -> float:
        return self.MainTransformerCo2Emission() / self.Total_Co2Emission()

    def TowerSecondaryEquipmentCo2EmissionShare(self) -> float:
        return self.TowerSecondaryEquipmentCo2Emission() / self.Total_Co2Emission()

    def Total_Co2EmissionShare(self) -> float:
        return self.Total_Co2Emission() / self.Total_Co2Emission()

    def convert_currency(self, foundation_cost):
        """
        Convert the foundation cost to the specified currency.
        """
        rates = {
            "DKK": 1,  # DKK to DKK is just 1
            "EURO": 1 / self.Dkk_per_Euro,  # Convert to EURO
            "DKK/KW": 1 / 1000,  # Convert to DKK per KW
            "EURO/KW": 1 / (self.Dkk_per_Euro * 1000),  # Convert to EURO per KW
        }

        # Ensure that a valid currency is provided; if not, assume no conversion (1)
        return foundation_cost * rates.get(self.currency, 1)

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
            4: 1250 * self.Dkk_per_Euro * 1000,
        }
        foundation_cost = costs.get(
            self.foundation_option,
            1000 * (self.water_depth**2) + 100000 * self.water_depth + 1500000,
        )
        return self.convert_currency(foundation_cost)

    def BOPCost(self):
        """
        Calculate the foundation cost for each water depth and convert to the selected currency.
        If a single water depth is passed, return a single value, otherwise return a list of costs.
        """
        return self.CalculateFoundationCost() + 1000 * self.electrical_cost

    def RealWACC(self) -> float:
        return (1 + self.wacc) / (1 + self.inflation) - 1

    def devexTotal(self) -> float:
        return np.sum(self.devex * self.rated_power * 1000)

    def CAPEXTurbineTower(self) -> float:
        return self.SalesPriceCalculation() / self.rated_power / 1000

    def CAPEXBOP(self) -> float:
        return self.BOPCost()

    def CAPEXWT(self) -> float:
        return (self.CAPEXBOP() + self.CAPEXTurbineTower()) * self.rated_power * 1000

    def CAPEXTotal(self) -> float:
        return np.sum(self.CAPEXWT())

    def opexTotal(self) -> float:
        return np.sum(self.opex * self.rated_power * 1000)

    def abexTotal(self) -> float:
        return np.sum(self.abex * self.rated_power * 1000)

    def AEP_WindFarm(self) -> float:
        # Ensure either AEP or capacity_factor is provided and AEP is not NaN
        if self.AEP is None and self.capacity_factor is None:
            raise ValueError(
                "Either Capacity Factor (capacity_factor) or AEP must be provided."
            )

        if self.AEP is not None:
            if self.nwt is None:
                raise ValueError(
                    "Number of turbines (nwt) must be provided for this calculation."
                )
            AEP_farm = np.sum(self.AEP)

        elif self.capacity_factor is not None:
            if self.nwt is None:
                raise ValueError(
                    "Number of turbines (nwt) must be provided for this calculation."
                )
            # Calculate AEP_farm using capacity_factor and rated_power if AEP is not available
            AEP_farm = np.sum(
                self.capacity_factor * self.rated_power * self.annual_hours
            )

        return AEP_farm

    def DiscountFactor_WACC_r(self) -> list:

        discount_factor = []
        # Calculate the discount factors based on project lifetime and WACC
        for year in range(-2, self.project_lifetime):
            discount_factor.append(1 / (1 + self.RealWACC()) ** year)

        return discount_factor

    def DiscountFactor_WACC_n(self) -> list:

        discount_factor = []
        # Calculate the discount factors based on project lifetime and WACC
        for year in range(-2, self.project_lifetime):
            discount_factor.append(1 / (1 + self.wacc) ** year)

        return discount_factor

    def AEPNet(self) -> float:
        AEP_net = []
        for year in range(self.project_lifetime):
            AEP_ = self.AEP_WindFarm() * ((1 + self.decline_factor) ** year)
            AEP_net.append(AEP_)

        self.AEP_net = np.sum(np.array(AEP_net))
        return self.AEP_net

    def AEPDiscount(self) -> float:

        AEP_discount = []
        for year in range(self.project_lifetime):
            AEP_d = (self.AEP_WindFarm() * (1 + self.decline_factor) ** year) * (
                1 / (1 + self.RealWACC()) ** year
            )
            AEP_discount.append(AEP_d)

        self.AEP_discount = np.sum((np.array(AEP_discount)))
        return self.AEP_discount

    def devexNet(self) -> float:

        project_start = 0
        devex = []
        for year in range(-2, project_start):
            devex_ = self.devexTotal() / 2
            devex.append(devex_)

        self.devex = np.sum(np.array(devex))
        return self.devex

    def devexDiscount(self) -> float:

        project_start = 0
        devex_discount = []
        discount_factors = self.DiscountFactor_WACC_n()

        for indx, year in enumerate(range(-2, project_start)):
            devex_d = (self.devexTotal() / 2) * discount_factors[indx]
            devex_discount.append(devex_d)

        self.devex_discount = np.sum((np.array(devex_discount)))

        return self.devex_discount

    def CAPEXNet(self) -> float:
        return self.CAPEXTotal()

    def CAPEXDiscount(
        self,
    ) -> float:  # in the excel sheet this is also called Total CAPEX
        base_yaer_indx = 1  # year = -1
        discount_factors = self.DiscountFactor_WACC_n()
        # self.CAPEX_discount = self.CAPEXTotal()*discount_factors[base_yaer_indx]

        return self.CAPEXTotal() * discount_factors[base_yaer_indx]

    def TotalCAPEX(self) -> float:  # in the excel sheet this is also called CAPEX total
        return self.CAPEXDiscount()

    def opexNET(self) -> float:

        opex_net = []
        for year in range(self.project_lifetime):
            opex_ = self.opexTotal() * ((1 + self.inflation) ** year)
            opex_net.append(opex_)

        self.opex_net = np.sum(np.array(opex_net))
        return self.opex_net

    def opexDiscount(self) -> float:

        base_yaer_indx = 2  # year = 0
        discount_factors = self.DiscountFactor_WACC_n()
        opex_d = []
        for indx, year in enumerate(range(self.project_lifetime)):

            opex_ = (
                self.opexTotal() * (1 + self.inflation) ** year
            ) * discount_factors[indx + base_yaer_indx]
            opex_d.append(opex_)

        self.opex_d = np.sum(np.array(opex_d))
        return self.opex_d

    def abexNET(self) -> float:
        return 0.0

    def abexDiscount(self) -> float:
        return 0.0

    def LCOENumerator(self):
        return (
            self.devexDiscount()
            + self.CAPEXDiscount()
            + self.opexDiscount()
            + self.abexDiscount()
        )

    def LCOEDenominator(self):
        return self.AEPDiscount()

    def LCOE(self):
        return self.LCOENumerator() / self.AEPDiscount()

    def NVP_devex(self):
        return self.devexDiscount() / self.LCOENumerator()

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

    def NVP_BOP_CAPEX(self):
        Turbine_incl_tower = (
            self.SalesPriceCalculation() / np.array(self.rated_power) / 1000
        )
        return (
            (self.CAPEXDiscount() * self.BOPCost())
            / (self.BOPCost() + Turbine_incl_tower)
            / self.LCOENumerator()
        )

    def NVP_opex(self):
        return self.opexDiscount() / self.LCOENumerator()

    def NVP_abex(self):
        return self.abexDiscount() / self.LCOENumerator()

    def run(self, **kwargs):

        self.set_inputs(**kwargs)
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

        Total_Co2Emission = self.Total_Co2Emission()

        turbine_cost = self.TotalCostCalculation()

        return {
            "Production net (MWh)": AEPNet,
            "Production discount (MWh)": AEPDiscount,
            "AEP net (MWh)": AEPNet / self.project_lifetime,
            "AEP discount (MWh)": AEPDiscount / self.project_lifetime,
            "DEVEX net (EURO)": devexNet,
            "DEVEX discount (EURO)": devexDiscount,
            "CAPEX net (EURO)": CAPEXNet,
            "CAPEX discount (EURO)": CAPEXDiscount,
            "OPEX net (EURO)": opexNet,
            "OPEX discount (EURO)": opexDiscount,
            "LCOE (EURO/MWh)": LCOE,
            "Total Co2 emission per turbine (kg CO2 eq)": Total_Co2Emission,
            "Turbine cost (EURO)": turbine_cost,
        }

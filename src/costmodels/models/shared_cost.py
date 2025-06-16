import numpy as np

from costmodels.base import CostModel


class SharedCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {
            "area": np.nan,  # km*km
            "grid_capacity": np.nan,  # MW
            "hpp_BOS_soft_cost": 119940,  # EUR/MW
            "hpp_grid_connection_cost": 50000,  # EUR/MW
            "land_cost": 300000,  # EUR/km**2
        }

    def _run(self) -> dict:
        CAPEX = (
            self.hpp_BOS_soft_cost + self.hpp_grid_connection_cost
        ) * self.grid_capacity + self.land_cost * self.area
        return {"capex": CAPEX / 1e6}

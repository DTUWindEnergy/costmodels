import numpy as np

from costmodels.base import CostModel


class SharedCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {
            "area": np.nan,
            "grid_capacity": np.nan,
            "hpp_BOS_soft_cost": 119940,
            "hpp_grid_connection_cost": 50000,
            "land_cost": 300000,
        }

    def _run(self) -> dict:
        CAPEX = (
            self.hpp_BOS_soft_cost + self.hpp_grid_connection_cost
        ) * self.grid_capacity + self.land_cost * self.area
        return {"capex": CAPEX / 1e6}


if __name__ == "__main__":
    SCM = SharedCostModel()
    res = SCM.run(area=127, grid_capacity=300)
    print(res)

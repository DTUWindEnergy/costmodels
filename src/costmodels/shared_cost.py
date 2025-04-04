import numpy as np

from costmodels.base import CostModel
from costmodels.units import Quant


class SharedCostModel(CostModel):
    @property
    def _cm_input_def(self):
        return {
            "area": Quant(np.nan, "km*km"),
            "grid_capacity": Quant(np.nan, "MW"),
            "hpp_BOS_soft_cost": Quant(119940, "EUR/MW"),
            "hpp_grid_connection_cost": Quant(50000, "EUR/MW"),
            "land_cost": Quant(300000, "EUR/km/km"),
        }

    def _run(self) -> dict:
        CAPEX = (
            self.hpp_BOS_soft_cost + self.hpp_grid_connection_cost
        ) * self.grid_capacity + self.land_cost * self.area
        return {"capex": CAPEX.to("MEUR")}


if __name__ == "__main__":
    SCM = SharedCostModel()
    res = SCM.run(area=Quant(127, "km*km"), grid_capacity=Quant(300, "MW"))
    print(res)

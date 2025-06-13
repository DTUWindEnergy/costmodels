import numpy as np

from costmodels.models import SharedCostModel
from costmodels.units import Quant


def test_run_shared_model():
    SCM = SharedCostModel()
    res = SCM.run(area=Quant(127, "km*km"), grid_capacity=Quant(300, "MW"))
    np.testing.assert_allclose(res["capex"].to_base_units().magnitude, 89082000)

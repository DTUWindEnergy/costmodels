import numpy as np

from costmodels.models import SharedCostModel


def test_run_shared_model():
    SCM = SharedCostModel()
    res = SCM.run(area=127, grid_capacity=300)
    np.testing.assert_allclose(res.capex, 89.082)
    np.testing.assert_allclose(res.opex, 0.0)

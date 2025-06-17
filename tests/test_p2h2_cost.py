import numpy as np

from costmodels.models.p2h2_cost import PowerToHydrogenCostModel


def test_run_power_to_hydrogen_model():
    electrolyzer_capacity = 800
    hydrogen_storage_capacity = 5000
    mean_hydrogen_offtake = 2343
    PTHCM = PowerToHydrogenCostModel()
    res = PTHCM.run(
        electrolyzer_capacity=electrolyzer_capacity,
        hydrogen_storage_capacity=hydrogen_storage_capacity,
        mean_hydrogen_offtake=mean_hydrogen_offtake,
    )
    np.testing.assert_allclose(res.capex, 641.5)
    # convert MEUR/year to EUR/s
    np.testing.assert_allclose(
        res.opex * 1e6 / (365.25 * 24 * 60 * 60),
        0.4427647207645701,
    )

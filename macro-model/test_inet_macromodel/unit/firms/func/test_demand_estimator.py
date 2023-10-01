import numpy as np

from inet_macromodel.firms.func.demand_estimator import DefaultDemandEstimator


class TestDemandEstimator:
    def test__compute_estimated_demand(self):
        assert np.allclose(
            DefaultDemandEstimator().compute_estimated_demand(
                previous_demand=np.array([1.0, 2.0]),
                estimated_growth_by_firm=np.array([0.0, 0.1]),
                estimated_sectoral_growth=np.array([0.1, 0.0]),
                firm_industry=np.array([0, 1]),
            ),
            np.array([1.1, 2.2]),
        )

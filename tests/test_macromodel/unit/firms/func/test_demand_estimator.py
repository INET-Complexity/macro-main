import numpy as np

from macromodel.agents.firms.func.demand_estimator import DefaultDemandEstimator


class TestDemandEstimator:
    def test__compute_estimated_demand(self):
        assert np.allclose(
            DefaultDemandEstimator(
                sectoral_growth_adjustment_speed=1.0,
                firm_growth_adjustment_speed=0.0,
            ).compute_estimated_demand(
                previous_demand=np.array([1.0, 2.0]),
                estimated_growth_by_firm=np.array([0.0, 0.1]),
                current_estimated_growth=0.0,
            ),
            np.array([1.0, 2.0]),
        )

import numpy as np

from inet_macromodel.households.func.consumption import DefaultHouseholdConsumption


class TestHouseholdConsumption:
    def test__compute_desired_consumption(self):
        assert np.allclose(
            DefaultHouseholdConsumption(
                consumption_smoothing_fraction=0.5,
                consumption_smoothing_window=4,
            ).compute_target_consumption(
                target_consumption_before_ce=np.full((18, 18), 3.0),
                target_consumption_ce=np.full(18, 1.0),
                target_consumption_expansion_loans=np.zeros(18),
                received_consumption_expansion_loans=np.zeros(18),
                consumption_weights=np.full(18, 1.0 / 18.0),
                income=np.ones(18),
                consumption_weights_by_income=np.array([]),
                take_consumption_weights_by_income_quantile=False,
            ),
            np.full((18, 18), 3.0555555),
        )

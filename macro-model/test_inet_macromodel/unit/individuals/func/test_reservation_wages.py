import numpy as np

from inet_macromodel.individuals.individual_properties import ActivityStatus
from inet_macromodel.individuals.func.reservation_wages import DefaultReservationWageSetter


class TestReservationWageSetter:
    def test__compute_reservation_wages(self):
        assert np.allclose(
            DefaultReservationWageSetter(unemployed_reservation_wage_timespan=12)
            .compute_reservation_wages(
                historic_wages=np.array([[10.0, 5.0, 2.0]]),
                current_individuals_activity=np.array(
                    [
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.UNEMPLOYED,
                    ]
                ),
                current_unemployment_benefits_by_individual=2.0,
            )
            .astype(float),
            np.array([10, 5.0, 2.0]),
        )

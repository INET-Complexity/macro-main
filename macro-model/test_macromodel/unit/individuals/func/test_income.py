import numpy as np

from macromodel.individuals.func.income import DefaultIncomeSetter
from macromodel.individuals.individual_properties import ActivityStatus


class TestIncomeSetter:
    def test__compute_income(self):
        assert np.allclose(
            DefaultIncomeSetter()
            .compute_income(
                current_individual_activity_status=np.array(
                    [
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.UNEMPLOYED,
                    ]
                ),
                current_wage=np.array(
                    [
                        10.0,
                        20.0,
                        5.0,
                    ]
                ),
                individual_social_benefits=np.array(
                    [
                        2.0,
                        2.0,
                        2.0,
                    ]
                ),
            )
            .astype(float),
            np.array([12.0, 22.0, 2.0]),
        )

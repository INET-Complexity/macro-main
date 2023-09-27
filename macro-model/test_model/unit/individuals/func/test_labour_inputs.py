import numpy as np

from model.individuals.func.labour_inputs import (
    ConstantIndividualsLabourInputsSetter,
    ScaledIndividualsProductivitySetter,
)
from model.individuals.individual_properties import ActivityStatus


class TestIndividualLabourInputsSetter:
    def test__update_labour_inputs(self):
        assert np.allclose(
            ConstantIndividualsLabourInputsSetter().update_labour_inputs(
                previous_individuals_labour_inputs=np.array([1.0, 2.0, 3.0]),
                current_individuals_activity=np.array(
                    [
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.UNEMPLOYED,
                    ]
                ),
            ),
            np.array([1.0, 2.0, 3.0]),
        )
        assert np.allclose(
            ScaledIndividualsProductivitySetter(
                increase_employed=0.05,
                decrease_unemployed=0.05,
            ).update_labour_inputs(
                previous_individuals_labour_inputs=np.array([1.0, 2.0, 3.0]),
                current_individuals_activity=np.array(
                    [
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.UNEMPLOYED,
                    ]
                ),
            ),
            np.array([1.05, 2.1, 2.85714286]),
        )

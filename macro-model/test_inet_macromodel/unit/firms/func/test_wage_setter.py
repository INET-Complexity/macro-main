import numpy as np

from inet_macromodel.firms.func.wage_setter import DefaultFirmWageSetter


class TestFirmWageSetter:
    def test__set_wages(self):
        assert np.allclose(
            DefaultFirmWageSetter(
                labour_market_tightness_markup_scale=0.1,
                markup_time_span=12,
            ).set_wages(
                firm_employments=[np.array([0, 1])],
                current_individual_labour_inputs=np.array([1.0, 2.0]),
                previous_employee_income=np.array([10.0, 20.0]),
                historic_desired_labour_inputs=[np.array([3.0, 4.0])],
                historic_realised_labour_inputs=[np.array([2.0, 3.0])],
            ),
            np.array([10.0, 20.0]),
        )

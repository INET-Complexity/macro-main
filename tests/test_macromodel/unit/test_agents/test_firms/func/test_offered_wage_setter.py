import numpy as np

from macromodel.agents.firms.func.offered_wage_setter import DefaultOfferedWageSetter


class TestOfferedWageSetter:
    def test__get_offered_wage_given_labour_inputs(self):
        offered_wage_function = DefaultOfferedWageSetter(
            labour_market_tightness_markup_scale=0.1,
            markup_time_span=12,
        ).get_offered_wage_given_labour_inputs_function(
            firm_employments=[np.array([0, 1])],
            current_individual_labour_inputs=np.array([1.0, 1.0]),
            previous_employee_income=np.array([5.0, 6.0]),
            historic_desired_labour_inputs=[np.array(3.0)],
            historic_realised_labour_inputs=[np.array(2.0)],
            unemployment_benefits_by_individual=1.0,
        )
        assert offered_wage_function(0, 1.0) == 5.5
        assert offered_wage_function(0, 2.0) == 11.0

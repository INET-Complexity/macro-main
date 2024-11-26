import numpy as np

from macromodel.firms.func.desired_labour import DefaultDesiredLabourSetter


class TestDesiredLabourSetter:
    def test__compute_desired_labour(self):
        assert np.allclose(
            DefaultDesiredLabourSetter(
                consider_intermediate_inputs=False,
                consider_capital_inputs=False,
            ).compute_desired_labour(
                current_target_production=np.array([1.0, 2.0]),
                current_limiting_intermediate_inputs=np.array([1.0, 2.0]),
                current_limiting_capital_inputs=np.array([1.0, 2.0]),
            ),
            np.array([1.0, 2.0]),
        )

import numpy as np

from macromodel.firms.func.desired_labour import DefaultDesiredLabourSetter


class TestDesiredLabourSetter:
    def test__compute_desired_labour(self):
        assert np.allclose(
            DefaultDesiredLabourSetter().compute_desired_labour(current_desired_production=np.array([1.0, 2.0])),
            np.array([1.0, 2.0]),
        )

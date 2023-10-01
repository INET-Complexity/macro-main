import numpy as np

from inet_macromodel.rest_of_the_world.func.exports import (
    DefaultRoWExportsSetter,
)


class TestRoWExportsSetter:
    def test__compute_exports(self):
        assert np.allclose(
            DefaultRoWExportsSetter().compute_exports(
                previous_desired_exports=np.array([1.0, 2.0]),
                model=None,
            ),
            np.array([1.0, 2.0]),
        )

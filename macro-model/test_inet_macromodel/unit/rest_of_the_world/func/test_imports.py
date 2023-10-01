import numpy as np

from inet_macromodel.rest_of_the_world.func.imports import DefaultRoWImportsSetter


class TestRoWImportsSetter:
    def test__compute_imports(self):
        assert np.allclose(
            DefaultRoWImportsSetter().compute_imports(
                previous_desired_imports=np.array([1.0, 2.0]),
                model=None,
            ),
            np.array([1.0, 2.0]),
        )

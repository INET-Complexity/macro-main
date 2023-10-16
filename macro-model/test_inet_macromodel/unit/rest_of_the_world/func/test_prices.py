import numpy as np

from inet_macromodel.rest_of_the_world.func.prices import (
    ConstantRoWPriceSetter,
    InflationRoWPriceSetter,
)


class TestRoWPriceSetter:
    def test__compute_price(self):
        assert np.allclose(
            ConstantRoWPriceSetter().compute_price(previous_price=np.array([1.0, 2.0]), previous_row_inflation=0.01),
            np.array([1.0, 2.0]),
        )
        assert np.allclose(
            InflationRoWPriceSetter().compute_price(previous_price=np.array([1.0, 2.0]), previous_row_inflation=0.01),
            np.array([1.01, 2.02]),
        )

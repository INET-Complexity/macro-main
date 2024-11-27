import numpy as np

from macromodel.rest_of_the_world.func.prices import (
    InflationRoWPriceSetter,
)


class TestRoWPriceSetter:
    def test__compute_price(self):
        assert np.allclose(
            InflationRoWPriceSetter().compute_price(
                initial_price=np.array([1.0, 2.0]),
                aggregate_country_price_index=1.01,
                adjustment_speed=1.0,
            ),
            np.array([1.01, 2.02]),
        )

import pathlib

from data.processing.synthetic_housing_market.default_synthetic_housing_market import (
    DefaultSyntheticHousingMarket,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticHousingMarket:
    def test__create(
        self,
        readers,
    ):
        housing_market = DefaultSyntheticHousingMarket(
            country_name="FRA",
            year=2014,
        )
        housing_market.create()

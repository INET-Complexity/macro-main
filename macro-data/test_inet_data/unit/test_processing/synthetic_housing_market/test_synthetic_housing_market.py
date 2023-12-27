import pathlib

from inet_data.processing.synthetic_housing_market.default_synthetic_housing_market import (
    DefaultSyntheticHousingMarket,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticHousingMarket:
    def test__create(
        self,
        readers,
    ):
        # TODO: this is a dataclass; what we should check is that the matching makes sense
        ...

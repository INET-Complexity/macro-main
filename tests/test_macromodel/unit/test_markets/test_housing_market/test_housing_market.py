from types import SimpleNamespace

import numpy as np
import pandas as pd

from macromodel.configurations import HousingMarketConfiguration
from macromodel.markets.housing_market.housing_market import HousingMarket


def _housing_data():
    return pd.DataFrame(
        {
            "House ID": [10, 11],
            "Value": [100.0, 200.0],
            "Rent": [1.0, 2.0],
            "Corresponding Inhabitant Household ID": [3.0, np.nan],
            "Corresponding Owner Household ID": [5.0, 6.0],
            "Is Owner-Occupied": [1.0, 0.0],
        }
    )


def test_from_data_preserves_household_and_property_ids(test_config):
    """Regression: HousingMarket.from_data must preserve property and household IDs."""
    market = HousingMarket.from_data(
        country_name="FRA",
        scale=1,
        data=_housing_data(),
        config=test_config["FRA"]["housing_market"],
    )

    assert market.states["House ID"].tolist() == [10, 11]
    assert market.states["Corresponding Owner Household ID"].tolist() == [5, 6]
    assert market.states["Corresponding Inhabitant Household ID"].tolist() == [3, -1]
    assert market.states["Is Owner-Occupied"].tolist() == [1, 0]


def test_from_pickled_market_preserves_household_and_property_ids():
    """Regression: pickled housing-market data keeps existing property IDs."""
    market = HousingMarket.from_pickled_market(
        synthetic_housing_market=SimpleNamespace(housing_market_data=_housing_data()),
        housing_market_configuration=HousingMarketConfiguration(),
        scale=1,
        country_name="FRA",
    )

    properties = market.states["properties"]
    assert properties["House ID"].tolist() == [10, 11]
    assert properties["Corresponding Owner Household ID"].tolist() == [5, 6]
    assert properties["Corresponding Inhabitant Household ID"].tolist() == [3, -1]
    assert properties["Is Owner-Occupied"].tolist() == [1, 0]

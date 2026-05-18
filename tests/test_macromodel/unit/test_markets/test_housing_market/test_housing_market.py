from types import SimpleNamespace

import numpy as np
import pandas as pd

from macromodel.configurations import HousingMarketConfiguration
from macromodel.markets.housing_market.func.clearing import (
    AutomaticHousingMarketClearer,
    DefaultHousingMarketClearer,
)
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


def test_property_sold_this_period_is_not_also_rented():
    """Regression: one property cannot clear in both sale and rental markets."""
    housing_data = pd.DataFrame(
        {
            "House ID": [0],
            "Value": [100.0],
            "Rent": [1.0],
            "Sale Price": [100.0],
            "Corresponding Inhabitant Household ID": [-1],
            "Corresponding Owner Household ID": [0],
            "Is Owner-Occupied": [0],
            "Temporarily for Sale": [True],
            "Up for Rent": [True],
        }
    )
    max_price_willing_to_pay = np.array([np.nan, 100.0, np.nan])
    max_rent_willing_to_pay = np.array([np.nan, np.nan, 10.0])

    for clearer in [
        DefaultHousingMarketClearer(random_assignment_shock_variance=0.0),
        AutomaticHousingMarketClearer(random_assignment_shock_variance=0.0),
    ]:
        current_sales = clearer.clear(
            housing_data=housing_data.copy(),
            household_main_residence_tenure_status=np.array([1, 1, 3]),
            max_price_willing_to_pay=max_price_willing_to_pay,
            max_rent_willing_to_pay=max_rent_willing_to_pay,
        )

        assert current_sales["sales_types"].tolist() == ["Sell"]
        assert current_sales["property_id"].tolist() == [0]

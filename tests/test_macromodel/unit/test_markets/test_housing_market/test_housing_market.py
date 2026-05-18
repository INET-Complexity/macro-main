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


def test_automatic_clearer_returns_empty_transactions_without_matches():
    """Regression: automatic matching handles empty demand and empty supply."""
    clearer = AutomaticHousingMarketClearer(random_assignment_shock_variance=0.0)

    housing_data = pd.DataFrame(
        {
            "House ID": [0],
            "Value": [100.0],
            "Rent": [1.0],
            "Sale Price": [100.0],
            "Corresponding Inhabitant Household ID": [-1],
            "Corresponding Owner Household ID": [0],
            "Is Owner-Occupied": [0],
            "Temporarily for Sale": [False],
            "Up for Rent": [False],
        }
    )

    current_sales = clearer.clear(
        housing_data=housing_data,
        household_main_residence_tenure_status=np.array([1]),
        max_price_willing_to_pay=np.array([np.nan]),
        max_rent_willing_to_pay=np.array([np.nan]),
    )

    assert current_sales.empty
    assert current_sales.columns.tolist() == [
        "sales_types",
        "property_id",
        "property_value",
        "price_or_rent",
        "seller_id",
        "buyer_id",
    ]


def test_sale_does_not_displace_inhabitant_without_completed_move():
    """Regression: a sale cannot overwrite an inhabitant who has not moved."""
    market = object.__new__(HousingMarket)
    market.states = {
        "properties": pd.DataFrame(
            {
                "House ID": [0, 1],
                "Value": [100.0, 200.0],
                "Rent": [1.0, 2.0],
                "Corresponding Inhabitant Household ID": [0, 1],
                "Corresponding Owner Household ID": [0, 1],
                "Is Owner-Occupied": [1, 1],
            }
        ),
        "current_sales": pd.DataFrame(
            {
                "sales_types": ["Sell"],
                "property_id": [0],
                "seller_id": [0],
                "buyer_id": [1],
                "property_value": [100.0],
                "price_or_rent": [100.0],
            }
        ),
    }
    market.ts = SimpleNamespace(
        total_number_of_houses_rented=[],
        total_number_of_houses_owner_occupied=[],
        total_number_of_houses_unoccupied=[],
        total_number_of_bought_houses=[],
        total_number_of_newly_rented_houses=[],
    )
    household_states = {
        "Corresponding Inhabited House ID": np.array([0, 1]),
        "Corresponding Property Owner": np.array([0, 1]),
        "Tenure Status of the Main Residence": np.array([1, 1]),
        "corr_renters": [[], []],
    }

    market.process_housing_market_clearing(
        household_states=household_states,
        household_received_mortgages=np.array([0.0, 100.0]),
        household_financial_wealth=np.array([0.0, 0.0]),
    )

    properties = market.states["properties"]
    assert market.states["current_sales"].empty
    assert properties["Corresponding Owner Household ID"].tolist() == [0, 1]
    assert properties["Corresponding Inhabitant Household ID"].tolist() == [0, 1]
    assert household_states["Corresponding Inhabited House ID"].tolist() == [0, 1]
    assert market.ts.total_number_of_bought_houses == [[0]]


def test_swap_sales_do_not_clear_new_inhabitant_from_previous_residence():
    """Regression: clearing old residences must respect transaction order."""
    market = object.__new__(HousingMarket)
    market.states = {
        "properties": pd.DataFrame(
            {
                "House ID": [0, 1],
                "Value": [100.0, 200.0],
                "Rent": [1.0, 2.0],
                "Corresponding Inhabitant Household ID": [0, 1],
                "Corresponding Owner Household ID": [0, 1],
                "Is Owner-Occupied": [1, 1],
            }
        ),
        "current_sales": pd.DataFrame(
            {
                "sales_types": ["Sell", "Sell"],
                "property_id": [1, 0],
                "seller_id": [1, 0],
                "buyer_id": [0, 1],
                "property_value": [200.0, 100.0],
                "price_or_rent": [200.0, 100.0],
            }
        ),
    }
    market.ts = SimpleNamespace(
        total_number_of_houses_rented=[],
        total_number_of_houses_owner_occupied=[],
        total_number_of_houses_unoccupied=[],
        total_number_of_bought_houses=[],
        total_number_of_newly_rented_houses=[],
    )
    household_states = {
        "Corresponding Inhabited House ID": np.array([0, 1]),
        "Corresponding Property Owner": np.array([0, 1]),
        "Tenure Status of the Main Residence": np.array([1, 1]),
        "corr_renters": [[], []],
    }

    market.process_housing_market_clearing(
        household_states=household_states,
        household_received_mortgages=np.array([200.0, 100.0]),
        household_financial_wealth=np.array([0.0, 0.0]),
    )

    properties = market.states["properties"]
    assert properties["Corresponding Inhabitant Household ID"].tolist() == [1, 0]
    assert properties["Corresponding Owner Household ID"].tolist() == [1, 0]
    assert household_states["Corresponding Inhabited House ID"].tolist() == [1, 0]
    assert market.ts.total_number_of_houses_unoccupied == [[0]]

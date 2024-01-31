from abc import ABC

import pandas as pd


class SyntheticHousingMarket(ABC):
    """
    Represents a synthetic housing market for a specific country and year.

    The housing market data is stored in a pandas DataFrame with the following columns:
        - House ID: The ID of the house.
        - Is Owner-Occupied: Whether the house is owner-occupied.
        - Corresponding Owner Household ID: The ID of the household that owns the house.
        - Corresponding Inhabitant Household ID: The ID of the household that inhabits the house.
        - Value: The value of the house.
        - Rent: The rent of the house.
        - Up for Rent: Whether the house is up for rent.

    Attributes:
        country_name (str): The name of the country.
        housing_market_data (pd.DataFrame): The housing market data.
    """

    def __init__(
        self,
        country_name: str,
        housing_market_data: pd.DataFrame,
    ):
        self.country_name = country_name

        # Housing market data
        self.housing_market_data = housing_market_data

"""Consumer carbon tax policy for the macroeconomic model.

This module calculates the tax that households pay on emissions
"""

from dataclasses import dataclass
from importlib.resources import files
from io import BytesIO

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country


@dataclass
class CarbonPrice:  ### consider renaming class to ConsumerCarbonPrice??? (now that it isn't the universal price for all carbon taxes)
    """Consumer carbon tax policy class.

    This class calculates the tax that households pay on emissions

    Attributes:
        country_name (Country): Country with the consumer carbon tax policy
        price (np.ndarray): Tax rate
        df (pd.DataFrame): Imported consumer carbon tax data
        current_t (int): Current index for price
        current_year (int) = Current year
    """

    country_name: Country
    price: np.ndarray
    df: pd.DataFrame
    current_t: int = 0
    current_year: int = 2014

    def __init__(self, country_name, policy_data=None):
        """Initialize a new consumer carbon tax.

        This class calculates the tax that households pay on emissions

        Attributes:
            country_name (Country): Country with the consumer carbon tax policy
            price (np.ndarray): Tax rate
            df (pd.DataFrame): Imported consumer carbon tax data
            current_t (int): Current index for price
            current_year (int) = Current year
        """

        self.country_name = country_name

        ### Fetch tax rates (PRELIMINARY - NEEDS TO BE REDONE)
        #       https://www.canada.ca/en/environment-climate-change/services/climate-change/pricing-pollution-how-it-will-work.html
        # QC cap-and-trade reserve price (or price floor) is used to set consumer tax rate from 2014-2019
        #       https://onclimatechangepolicydotorg.wordpress.com/carbon-pricing/price-floors-and-ceilings/
        # Apply mapping from id to sector name
        # CARBON_PRICE_BYTES = files("macromodel.policy").joinpath("consumer_carbon_price_rates.csv").read_bytes()
        self.df = policy_data.carbon_prices_data["carbon_price_bytes"]  # pd.read_csv(BytesIO(CARBON_PRICE_BYTES))

        self.price = np.zeros(len(self.df))
        df_sub = self.df[["Date", self.country_name]]
        for t in range(len(self.df)):
            time = t + 2014
            df_row = df_sub[df_sub["Date"] == time]
            df_item = df_row[self.country_name].values
            self.price[t] = df_item[0]

    def update(self):
        """Advance the timestep by the increment."""
        self.current_t += 1
        self.current_year += 1

    def get_price(self):
        """Retrieve the current price

        Returns:
            float: Current tax rate
        """
        return self.price[self.current_t]

    def get_inflation_price(self, cpi: float):
        """Retrieve the current price accounting for inflation

        Returns:
            float: Current tax rate with inflation
        """
        return self.price[self.current_t] * cpi

    def reset(self):
        """Resets the time variables"""
        self.current_t = 0
        self.current_year = 2014

from dataclasses import dataclass

import pandas as pd


@dataclass
class SyntheticCreditMarket:
    """
    Represents a synthetic credit market for a specific country and year.

    Attributes:
        country_name (str): The name of the country.
        year (int): The year of the credit market data.
        credit_market_data (pd.DataFrame): The credit market data for the country and year (contains information on loans
                                            including value, interest rate and maturity).
    """

    country_name: str
    year: int
    credit_market_data: pd.DataFrame

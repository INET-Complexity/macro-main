from abc import ABC, abstractmethod

import pandas as pd


class SyntheticCentralBank(ABC):
    """
    Represents a synthetic central bank for a specific country and year.

    The central bank data is stored in a pandas DataFrame with a single column:
        - Policy Rate: The policy rate.

    Attributes:
        country_name (str): The name of the country.
        year (int): The year of the central bank data.
        central_bank_data (pd.DataFrame): The central bank data.
    """

    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
        central_bank_data: pd.DataFrame,
    ):
        self.country_name = country_name
        self.year = year

        # Bank data
        self.central_bank_data = central_bank_data

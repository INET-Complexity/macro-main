"""
This module provides functionality for reading and processing World Bank exchange rate data.
It supports various currency conversions between USD, EUR, and local currency units (LCU)
for different countries and time periods.

Key Features:
- Read exchange rate data from CSV files
- Convert between USD, EUR, and local currencies
- Support for annual exchange rates
- Handle special cases (e.g., ROW as USA)
- Prune historical data by date

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader

    # Initialize reader with exchange rate data
    reader = ExchangeRatesReader.from_csv(
        path=Path("path/to/exchange_rates.csv")
    )

    # Convert EUR to local currency for Japan in 2020
    jpy_rate = reader.from_eur_to_lcu("JPN", 2020)

    # Get all exchange rates for 2020
    rates_2020 = reader.exchange_rates_dict(2020)

    # Convert between USD and local currency
    to_usd = reader.to_usd("GBR", 2020)
    from_usd = reader.from_usd("GBR", 2020)
    ```

Note:
    Exchange rates are stored relative to USD (i.e., LCU per USD).
    ROW (Rest of World) is treated as USA for exchange rate purposes.
"""

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from macro_data.configuration.region import Region
from macro_data.readers.util.prune_util import prune_index


class ExchangeRatesReader:
    """
    A class for reading and manipulating World Bank exchange rate data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the exchange rate data.

    Methods:
        from_csv(path: Path | str) -> "ExchangeRatesReader":
            Creates a WorldBankRatesReader instance from a CSV file.

        exchange_rates_dict(year: int) -> dict[str, float]:
            Returns a dictionary of exchange rates for a specific year.

        to_usd(country: str, year: int) -> float:
            Converts a currency value from a specific country to USD.

        from_usd(country: str, year: int) -> float:
            Converts a currency value from USD to a specific country.

        from_eur_to_lcu(country: str, year: int) -> float:
            Converts a currency value from EUR to local currency unit (LCU).

        from_usd_to_lcu(country: str, year: int) -> float:
            Converts a currency value from USD to local currency unit (LCU).

        prune(prune_date: date):
            Prunes the exchange rate data based on a specified date.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ExchangeRatesReader.

        Args:
            df (pd.DataFrame): DataFrame containing exchange rate data with countries as index
                             and years as columns.
        """
        self.df = df

    @classmethod
    def from_csv(cls, path: Path | str) -> "ExchangeRatesReader":
        """
        Create an ExchangeRatesReader instance from a CSV file.

        Args:
            path (Path | str): Path to CSV file containing exchange rate data.

        Returns:
            ExchangeRatesReader: Initialized reader with loaded exchange rate data.

        Note:
            CSV should have countries as rows and years as columns.
        """
        df = pd.read_csv(path, index_col=0)
        df.columns.name = "Year"
        return cls(df)

    def exchange_rates_dict(self, year: int) -> dict[str, float]:
        """
        Get all exchange rates for a specific year.

        Args:
            year (int): Year to get exchange rates for.

        Returns:
            dict[str, float]: Dictionary mapping country codes to exchange rates (LCU per USD).
        """
        return self.df[str(year)].to_dict()

    def to_usd(self, country: str, year: int) -> float:
        """
        Convert from local currency to USD.

        Args:
            country (str): Country code (e.g., 'GBR', 'JPN').
            year (int): Year of exchange rate.

        Returns:
            float: Exchange rate for converting from local currency to USD.

        Note:
            ROW (Rest of World) is treated as USA.
        """
        if isinstance(country, Region):
            country = country.parent_country
        if country == "ROW":
            country = "USA"
        return 1 / self.df.loc[country, str(year)]

    def from_usd(self, country: str, year: int) -> float:
        """
        Convert from USD to local currency.

        Args:
            country (str): Country code (e.g., 'GBR', 'JPN').
            year (int): Year of exchange rate.

        Returns:
            float: Exchange rate for converting from USD to local currency.

        Note:
            ROW (Rest of World) is treated as USA. The default currency is USD.
        """
        if isinstance(country, Region):
            country = country.parent_country
        if country == "ROW":
            country = "USA"
        return self.df.loc[country, str(year)]

    def from_eur_to_lcu(self, country: str, year: int) -> float:
        """
        Convert from EUR to local currency unit (LCU).

        This method performs a two-step conversion:
        1. EUR to USD (using Germany as EUR proxy)
        2. USD to target currency

        Args:
            country (str): Country code (e.g., 'GBR', 'JPN').
            year (int): Year of exchange rate.

        Returns:
            float: Exchange rate for converting from EUR to local currency.

        Note:
            Uses Germany (DEU) as proxy for EUR.
            ROW (Rest of World) is treated as USA.
        """
        if isinstance(country, Region):
            country = country.parent_country
        if country == "ROW":
            country = "USA"
        return self.to_usd("DEU", year) * self.from_usd(country, year)

    def from_usd_to_lcu(self, country: str, year: int) -> float:
        """
        Convert from USD to local currency unit (LCU).

        This is an alias for from_usd() for consistency with from_eur_to_lcu().

        Args:
            country (str): Country code (e.g., 'GBR', 'JPN').
            year (int): Year of exchange rate.

        Returns:
            float: Exchange rate for converting from USD to local currency.

        Note:
            ROW (Rest of World) is treated as USA.
        """
        if isinstance(country, Region):
            country = country.parent_country
        if country == "ROW":
            country = "USA"
        return self.from_usd(country, year)

    def prune(self, prune_date: date):
        """
        Prune exchange rate data to remove entries after a specified date.

        Args:
            prune_date (date): Date after which to remove data.

        Note:
            Modifies the df attribute in place.
            Uses the prune_index utility function.
        """
        mask = prune_index(self.df.columns, prune_date)
        self.df = self.df.loc[:, mask]

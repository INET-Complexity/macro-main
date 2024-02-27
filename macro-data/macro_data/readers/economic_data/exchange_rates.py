from datetime import date, datetime
from pathlib import Path

import pandas as pd

from macro_data.readers.util.prune_util import prune_index


class WorldBankRatesReader:
    """
    A class for reading and manipulating World Bank exchange rate data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the exchange rate data.

    Methods:
        from_csv(path: Path | str) -> "WorldBankRatesReader":
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

        prune(prune_date: str | int | pd.Timestamp, prune_date_format="%Y-%m-%d"):
            Prunes the exchange rate data based on a specified date.
    """

    def __init__(self, df):
        self.df = df

    @classmethod
    def from_csv(cls, path: Path | str) -> "WorldBankRatesReader":
        df = pd.read_csv(path, index_col=0)
        df.columns.name = "Year"
        return cls(df)

    def exchange_rates_dict(self, year: int) -> dict[str, float]:
        return self.df[str(year)].to_dict()

    def to_usd(self, country: str, year: int) -> float:
        return 1 / self.df.loc[country, str(year)]

    def from_usd(self, country: str, year: int) -> float:
        return self.df.loc[country, str(year)]

    def from_eur_to_lcu(self, country: str, year: int) -> float:
        return self.to_usd("DEU", year) * self.from_usd(country, year)

    def from_usd_to_lcu(self, country: str, year: int) -> float:
        return self.from_usd(country, year)

    def prune(self, prune_date: date):
        """
        Prunes the exchange rate data based on a specified date.

        Args:
            prune_date (datetime): The date to prune the data.
        """
        # WB exchange rates
        mask = prune_index(self.df.columns, prune_date)
        self.df = self.df.loc[:, mask]

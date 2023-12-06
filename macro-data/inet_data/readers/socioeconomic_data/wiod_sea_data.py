import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from inet_data.readers.economic_data.exchange_rates import WorldBankRatesReader
from inet_data.readers.util.prune_util import prune_index


class WIODSEAReader:
    """
    A class for reading and manipulating socioeconomic data from the WIOD-SEA dataset.

    Args:
        df (pd.DataFrame): The DataFrame containing the socioeconomic data.
        year (int): The year of the data.
        industries (list[str]): The list of industries to include in the analysis.
        exchange_rates (WorldBankRatesReader): An instance of the WorldBankRatesReader class for exchange rate data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the socioeconomic data.
        year (int): The year of the data.
        industries (list[str]): The list of industries to include in the analysis.
        exchange_rates (WorldBankRatesReader): An instance of the WorldBankRatesReader class for exchange rate data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        year: int,
        industries: list[str],
        exchange_rates: WorldBankRatesReader,
    ):
        self.df = df
        self.year = year
        self.industries = industries
        self.exchange_rates = exchange_rates

        self.clean_sea()

    @classmethod
    def agg_from_csv(
        cls,
        path: Path | str,
        aggregation_path: Path,
        year: int,
        country_names: list[str],
        industries: list,
        exchange_rates: WorldBankRatesReader,
    ):
        """
        Aggregate socioeconomic data from a CSV file. Aggregation is done using a JSON file that maps sectors to aggregated sectors.

        Args:
            path (Path | str): The path to the CSV file.
            aggregation_path (Path): The path to the aggregation JSON file.
            year (int): The year of the data.
            country_names (list[str]): The list of country names to include in the aggregation.
            industries (list): The list of industries to include in the aggregation.
            exchange_rates (WorldBankRatesReader): The exchange rates reader.

        Returns:
            WIOD_SEA_Data: An instance of the WIOD_SEA_Data class containing the aggregated data.
        """
        # Aggregate industries
        raw_df = pd.read_csv(path, thousands=",", index_col=[0, 1, 2, 3])
        aggregation = json.load(open(aggregation_path))
        agg_dict_full = {}
        for key, values in aggregation.items():
            for value in values:
                agg_dict_full[value] = key
        stacked = raw_df[str(year)].reset_index()
        stacked.rename(columns={str(year): "Value"}, inplace=True)

        # Don't include indices or employment info
        stacked = stacked[stacked["variable"].isin(["VA", "LAB", "CAP", "K"])]

        # Convert to USD
        stacked["Value"] /= stacked["country"].map(exchange_rates.exchange_rates_dict(year))
        stacked["Value"] *= 1e6

        # Aggregate
        stacked["new_code"] = stacked["code"].map(agg_dict_full)

        # Unstack things
        sea = stacked.groupby(["country", "new_code", "variable"])["Value"].sum().unstack()

        # Cosmetics
        sea = sea.loc[sea.index.get_level_values(0).isin(country_names)]
        sea = sea.loc[sea.index.get_level_values(1).isin(industries)]
        sea.index.names = ["Country", "Industry"]
        sea.columns.name = "Field"
        sea.rename(
            {
                "VA": "Value Added",
                "LAB": "Labour Compensation",
                "CAP": "Capital Compensation",
                "K": "Capital Stock",
            },
            axis=1,
            inplace=True,
        )

        return cls(
            df=sea,
            year=year,
            industries=industries,
            exchange_rates=exchange_rates,
        )

    def clean_sea(self) -> None:
        """
        Clean the socioeconomic data by overwriting negative capital compensation with zero.
        """
        self.df.loc[:, "Capital Compensation"] = np.maximum(0.0, self.df.loc[:, "Capital Compensation"])

    def get_values_in_usd(self, country: str, field: str) -> np.ndarray:
        """
        Get the values of a specific field in USD for a given country and industry.

        Args:
            country (str): The name of the country.
            field (str): The name of the field.

        Returns:
            np.ndarray: An array of values in USD.
        """
        return self.df.loc[country].loc[self.industries, field].values

    def get_values_in_lcu(self, country: str, field: str) -> np.ndarray:
        """
        Get the values of a specific field in local currency units (LCU) for a given country and industry.

        Args:
            country (str): The name of the country.
            field (str): The name of the field.

        Returns:
            np.ndarray: An array of values in LCU.
        """
        return self.get_values_in_usd(country, field) * self.exchange_rates.from_usd_to_lcu(country, self.year)

    def prune(self, prune_date: int | datetime | str, date_format: str = "%Y-%m-%d"):
        """
        Prune the exchange rate data based on a given date.

        Args:
            prune_date (int | datetime | str): The date to prune the exchange rate data.
            date_format (str, optional): The format of the prune_date if it is a string. Defaults to "%Y-%m-%d".
        """
        # WIOD_SEA
        mask = prune_index(self.exchange_rates.df.columns, prune_date, "WIOD_SEA", date_format=date_format)
        self.exchange_rates.df = self.exchange_rates.df.loc[:, mask]

from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader
from macro_data.readers.io_tables.mappings import WIOD_AGGREGATE, WIOD_ALL
from macro_data.readers.util.prune_util import prune_index


class WIODSEAReader:
    """
    A class for reading and manipulating socioeconomic data from the WIOD-SEA dataset.

    Args:
        df (pd.DataFrame): The DataFrame containing the socioeconomic data.
        year (int): The year of the data.
        industries (list[str]): The list of industries to include in the analysis.
        exchange_rates (ExchangeRatesReader): An instance of the WorldBankRatesReader class for exchange rate data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the socioeconomic data.
        year (int): The year of the data.
        industries (list[str]): The list of industries to include in the analysis.
        exchange_rates (ExchangeRatesReader): An instance of the WorldBankRatesReader class for exchange rate data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        year: int,
        industries: list[str],
        exchange_rates: ExchangeRatesReader,
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
        aggregation_type: Literal["All", "Aggregate"],
        year: int,
        country_names: list[str],
        industries: list,
        exchange_rates: ExchangeRatesReader,
        value_added_dict: dict[str, pd.Series],
        regions_dict: Optional[dict[Country, list[Region]]] = None,
    ) -> "WIODSEAReader":
        """
        Aggregate socioeconomic data from a CSV file. Aggregation is done using a JSON file that maps sectors
        to aggregated sectors.

        Args:
            path (Path | str): The path to the CSV file.
            aggregation_type (Literal["All", "Aggregate"]): The industry level aggregation.
            year (int): The year of the data.
            country_names (list[str]): The list of country names to include in the aggregation.
            industries (list): The list of industries to include in the aggregation.
            exchange_rates (ExchangeRatesReader): The exchange rates reader.
            value_added_dict (dict[str, np.ndarray]): A dictionary containing the value added data.
            regions_dict (Optional[dict[Country, list[Region]]]): A dictionary containing the regions for each country.

        Returns:
            WIOD_SEA_Data: An instance of the WIOD_SEA_Data class containing the aggregated data.
        """
        # Aggregate industries
        raw_df = pd.read_csv(path, thousands=",", index_col=[0, 1, 2, 3])
        # aggregation = json.load(open(aggregation_path))
        aggregation = WIOD_AGGREGATE if aggregation_type == "Aggregate" else WIOD_ALL
        agg_dict_full = {}
        for key, values in aggregation.items():
            for value in values:
                agg_dict_full[value] = key
        stacked = raw_df[str(year)].reset_index()
        stacked.rename(columns={str(year): "Value"}, inplace=True)

        # Don't include indices or employment info
        stacked = stacked[stacked["variable"].isin(["VA", "COMP", "CAP", "K"])]

        # Convert to USD
        stacked["Value"] = np.maximum(1.0, stacked["Value"])  # minimum value
        stacked["Value"] /= stacked["country"].map(exchange_rates.exchange_rates_dict(year))
        stacked["Value"] *= 1e6

        # Aggregate
        stacked["new_code"] = stacked["code"].map(agg_dict_full)

        # Unstack things
        sea = stacked.groupby(["country", "new_code", "variable"])["Value"].sum().unstack()

        # Cosmetics
        sea = sea.loc[sea.index.get_level_values(0).isin(country_names)]

        sea_industries = sea.index.get_level_values(1).unique()
        not_present_industries = [industry for industry in sea_industries if industry not in industries]

        industry_to_fix = [
            any([ind.startswith(sea_ind[0]) for ind in industries]) for sea_ind in not_present_industries
        ]

        for sea_industry, should_fix in zip(not_present_industries, industry_to_fix):
            sub_industries = [ind for ind in industries if ind.startswith(sea_industry[0])]
            for country in country_names:
                factors = value_added_dict[country].loc[sub_industries].values
                factors /= factors.sum()
                for sub_industry, factor in zip(sub_industries, factors):
                    sea.loc[(country, sub_industry), :] = sea.loc[(country, sea_industry), :] * factor

        sea = sea.loc[sea.index.get_level_values(1).isin(industries)]

        sea.index.names = ["Country", "Industry"]
        sea.columns.name = "Field"
        sea.rename(
            {
                "VA": "Value Added",
                "COMP": "Labour Compensation",
                "CAP": "Capital Compensation",
                "K": "Capital Stock",
            },
            axis=1,
            inplace=True,
        )

        # rescale
        for country in country_names:
            scale = value_added_dict[country] / sea.loc[country, "Value Added"]
            scale = np.copy(scale.values)
            for field in ["Value Added", "Labour Compensation", "Capital Compensation", "Capital Stock"]:
                sea.loc[country, field] = (sea.loc[country, field] * scale).values

        if regions_dict is not None:
            for country, regions in regions_dict.items():
                for region in regions:
                    ratios = (value_added_dict[region] / value_added_dict[country]).values
                    region._va_ratio = value_added_dict[region].sum() / value_added_dict[country].sum()
                    new = (sea.loc[country].copy().T * ratios).T
                    new = pd.concat([new], keys=[region])
                    new.index.names = sea.index.names
                    new.columns.names = sea.columns.names
                    # add the new rows to the sea dataframe
                    new = new.fillna(0)
                    sea = pd.concat([sea, new])
                sea.drop(country, inplace=True)

        sea = sea.fillna(0)

        sea.sort_index(inplace=True)

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

    def set_values_in_usd(self, country: str, field: str, values: np.ndarray) -> None:
        """
        Set the values of a specific field in USD for a given country and industry.

        Args:
            country (str): The name of the country.
            field (str): The name of the field.
            values (np.ndarray): An array of values in USD.
        """
        self.df.loc[country].loc[self.industries, field] = values

    #
    # def get_values_in_lcu(self, country: str, field: str) -> np.ndarray:
    #     """
    #     Get the values of a specific field in local currency units (LCU) for a given country and industry.
    #
    #     Args:
    #         country (str): The name of the country.
    #         field (str): The name of the field.
    #
    #     Returns:
    #         np.ndarray: An array of values in LCU.
    #     """
    #     return self.get_values_in_usd(country, field) * self.exchange_rates.from_usd_to_lcu(country, self.year)

    def prune(self, prune_date: date):
        """
        Prune the exchange rate data based on a given date.

        Args:
            prune_date (datetime): The date to prune the exchange rate data.
        """
        # WIOD_SEA
        mask = prune_index(self.exchange_rates.df.columns, prune_date)
        self.exchange_rates.df = self.exchange_rates.df.loc[:, mask]

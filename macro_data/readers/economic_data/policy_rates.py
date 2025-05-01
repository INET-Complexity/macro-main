from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers.util.prune_util import prune_index

default_rates = {"BRN": 0.055, "KAZ": 0.15, "TUN": 0.06}


def default_rate_df(dates: list[pd.Timestamp], rate):
    """
    Create a DataFrame with default policy rates for a given set of dates.

    Args:
        dates (list[pd.Timestamp]): List of dates for the rates
        rate (float): Default policy rate value

    Returns:
        pd.DataFrame: DataFrame with dates as index and constant rate values
    """
    return pd.DataFrame(
        index=pd.to_datetime(dates),
        data={"Policy Rate": np.full(len(dates), rate)},
    )


class PolicyRatesReader:
    """
    Reader class for central bank policy rates data.

    This class handles reading and processing of policy rates from various central banks.
    It supports special cases for certain countries and provides default rates where
    necessary.

    Args:
        path (Path | str): Path to the main policy rates CSV file
        country_code_path (Path | str): Path to the country code mappings CSV file

    Attributes:
        path (Path): Path to the policy rates data
        df (pd.DataFrame): Main policy rates DataFrame
        c_map (pd.DataFrame): Country code mapping DataFrame
    """

    def __init__(self, path: Path | str, country_code_path: Path | str):
        """Initialize the PolicyRatesReader with data and country code paths."""
        self.path = path
        self.df = pd.read_csv(path, engine="pyarrow")
        self.c_map = pd.read_csv(country_code_path)
        self.c_map.loc[len(self.c_map)] = [
            "Euro Area",
            "XM",
            "XM",
            "",
            "",
        ]
        self.df["code"] = self.country_code_switch(self.df["REF_AREA"].values)

    def get_policy_rates(self, country: Country | str | Region) -> pd.DataFrame:
        """
        Get policy rates for a specific country.

        Args:
            country (Country | str): Country to get rates for, either as Country object
                                   or ISO 3-letter code

        Returns:
            pd.DataFrame: DataFrame with dates as index and policy rates as values,
                         rates are in decimal form

        Note:
            - For EU countries, returns ECB rates
            - Special handling for Costa Rica and Singapore
            - Uses default rates for certain countries
            - Returns quarterly data
        """
        if isinstance(country, Region):
            country = country.parent_country
        if isinstance(country, Country):
            is_eu_country = country.is_eu_country
            country = country.value
        else:
            is_eu_country = Country(country).is_eu_country

        dates = self.get_dates()
        if country == "CRI":
            return self.costa_rica_rates()
        elif country == "SGP":
            return self.singapore_rates()
        elif country in default_rates:
            return default_rate_df(dates, default_rates[country])

        if is_eu_country:
            country = "XM"

        df_c = self.df.loc[self.df["code"] == country].copy()
        dt_cols = pd.to_datetime(df_c.columns, format="%Y-%m", errors="coerce")
        df_c.drop(columns=df_c.columns[dt_cols.isna()], inplace=True)  # noqa
        df_c = df_c.T
        df_c.index = pd.to_datetime(df_c.index)
        df_c = df_c.resample("QS").mean()
        # pandas bug
        df_c.index.freq = None
        df_c.columns = ["Policy Rate"]
        return df_c / 100.0

    def singapore_rates(self):
        """
        Get Singapore policy rates from a separate data file.

        Returns:
            pd.DataFrame: DataFrame with dates as index and Singapore policy rates
                         as values, rates are in decimal form

        Note:
            Uses SORA (Singapore Overnight Rate Average) data
        """
        df = pd.read_csv(self.path / "policy_rate_sgp.csv", index_col=[0, 1, 2])
        df = df.reset_index()
        df["SORA Value Date"] = df["SORA Value Date"].ffill()
        df[df == "SORA"] = np.nan
        df["SORA"] = df["SORA"].astype(float)
        df = df.groupby("SORA Value Date")["SORA"].mean()[:-1]
        df.index = pd.to_datetime([pd.Timestamp(int(y), 1, 1) for y in df.index.values])
        df = df.resample("QE").ffill()
        df.index = pd.to_datetime([d + pd.Timedelta(days=1) for d in df.index.values])
        df = pd.DataFrame(df)
        df.columns = ["Policy Rate"]
        return df / 100.0

    def costa_rica_rates(self):
        """
        Get Costa Rica policy rates from a separate data file.

        Returns:
            pd.DataFrame: DataFrame with dates as index and Costa Rica policy rates
                         as values, rates are in decimal form
        """
        df = pd.read_csv(self.path.parent / "policy_rate_cri.csv", engine="pyarrow", index_col=0)
        df = df.mean(axis=0)
        df.index = pd.to_datetime([pd.Timestamp(int(y), 1, 1) for y in df.index.values])
        df = df.resample("QE").ffill()
        df.index = pd.to_datetime([d + pd.Timedelta(days=1) for d in df.index.values])
        df = pd.DataFrame(df)
        df.columns = ["Policy Rate"]
        return df / 100.0

    def country_code_switch(self, codes):
        """
        Convert country codes from Alpha-2 to Alpha-3 format.

        Args:
            codes (list[str]): List of Alpha-2 country codes

        Returns:
            list[str]: List of corresponding Alpha-3 country codes
        """
        return [self.c_map.loc[self.c_map["Alpha-2 code"] == c, "Alpha-3 code"].values[0] for c in codes]

    @staticmethod
    def get_dates() -> list[pd.Timestamp]:
        """
        Generate list of quarterly dates from 2000 to 2024.

        Returns:
            list[pd.Timestamp]: List of first days of each quarter
        """
        dates = []
        for year in range(2000, 2025):
            for month in [1, 4, 7, 10]:
                dates.append(pd.Timestamp(year, month, 1))
        return dates

    def prune(self, prune_date: date):
        """
        Prune policy rate data to only include entries after specified date.

        Args:
            prune_date (date): Date to prune data from

        Note:
            Modifies the data in place
        """
        # WB exchange rates
        mask = prune_index(self.df.columns, prune_date)
        self.df = self.df.loc[:, mask]

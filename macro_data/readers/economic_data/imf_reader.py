"""
This module provides functionality for reading and processing International Monetary Fund (IMF)
data, including banking statistics, national accounts, inflation, and labor market indicators.
It handles data from multiple IMF databases including the Financial Access Survey (FAS) and
International Financial Statistics (IFS).

Key Features:
- Read IMF banking sector demographics
- Process national accounts growth rates
- Calculate inflation rates from CPI and PPI
- Access labor market statistics
- Scale data for model compatibility
- Handle missing data and country codes

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.economic_data.imf_reader import IMFReader
    from macro_data.configuration.countries import Country

    # Initialize reader with data directory and scaling factors
    scale_dict = {Country.FRANCE: 1000, Country.GERMANY: 1000}
    reader = IMFReader.from_data(
        data_path=Path("path/to/imf/data"),
        scale_dict=scale_dict
    )

    # Get banking sector statistics
    n_banks = reader.number_of_commercial_banks(2020, "FRA")
    deposits = reader.total_commercial_deposits(2020, "FRA")

    # Get macroeconomic indicators
    inflation = reader.get_inflation("DEU")
    growth = reader.get_na_growth_rates("DEU")
    labor = reader.get_labour_stats("DEU")
    ```

Note:
    Monetary values are in domestic currency units.
    Growth rates and ratios are returned as decimals.
"""

import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers.util.prune_util import prune_index


class IMFReader:
    """
    A class for reading and processing IMF data across multiple databases.

    This class handles:
    1. Banking sector statistics (FAS database)
    2. National accounts and growth rates (IFS)
    3. Price indices and inflation rates
    4. Labor market indicators

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Dictionary containing loaded IMF data by category
    scale_dict : dict[Country, int]
        Dictionary mapping countries to scaling factors

    Attributes
    ----------
    scale_dict : dict[Country, int]
        Scaling factors for each country
    data : dict[str, pd.DataFrame]
        Processed IMF data by category

    Notes
    -----
    - Banking data is scaled according to model requirements
    - Growth rates and ratios are converted to decimals
    - Missing data is handled through optional returns
    """

    def __init__(self, data: dict[str, pd.DataFrame], scale_dict: dict[Country, int]):
        self.scale_dict = scale_dict
        self.data = {key: data[key] for key in data.keys()}

    @classmethod
    def from_data(cls, data_path: Path | str, scale_dict: dict[Country, int]) -> "IMFReader":
        """
        Create an IMFReader instance from data files.

        Parameters
        ----------
        data_path : Path | str
            Path to directory containing IMF data files
        scale_dict : dict[Country, int]
            Dictionary mapping countries to scaling factors

        Returns
        -------
        IMFReader
            Initialized reader with loaded IMF data

        Notes
        -----
        - Expects 'imf_fas_bank_demographics.csv' and 'IFS.csv' in data_path
        - Files should use latin-1 encoding
        """
        data = {
            "bank_demography": pd.read_csv(
                data_path / "imf_fas_bank_demographics.csv", encoding="latin-1", engine="pyarrow"
            ),
            "international_financial_statistics": pd.read_csv(
                data_path / "IFS.csv", encoding="latin-1", engine="pyarrow"
            ),
        }
        return cls(data, scale_dict)

    @staticmethod
    def get_files_with_codes() -> dict[str, str]:
        """
        Get mapping of data categories to file names.

        Returns
        -------
        dict[str, str]
            Dictionary mapping data categories to file names
        """
        return {
            "bank_demography": "imf_fas_bank_demographics",
            "international_financial_statistics": "IFS",
        }

    def get_value(self, year: int, country: str, stat: str) -> float:
        """
        Get a specific statistic for a country and year.

        Parameters
        ----------
        year : int
            Year to get data for
        country : str
            Country code
        stat : str
            Statistical indicator to retrieve

        Returns
        -------
        float
            Value of the requested statistic

        Notes
        -----
        - Handles comma-separated number formatting
        """
        df = self.data["bank_demography"]
        mask = (df["STAT"] == stat) & (df["COU"] == country)
        value = df.loc[mask][str(year)].iloc[0]
        return float(value.replace(",", ""))

    def number_of_commercial_banks(self, year: int, country: str | Country) -> float:
        """
        Get number of commercial banks for a country.

        Parameters
        ----------
        year : int
            Year to get data for
        country : str | Country
            Country identifier

        Returns
        -------
        float
            Number of commercial banks (scaled)
        """
        return self.get_value(year, country, "Institutions of commercial banks") / self.scale_dict[country]

    def number_of_commercial_depositors(self, year: int, country: str | Country) -> float:
        """
        Get number of commercial bank depositors for a country.

        Parameters
        ----------
        year : int
            Year to get data for
        country : str | Country
            Country identifier

        Returns
        -------
        float
            Number of depositors (scaled)
        """
        return self.get_value(year, country, "Depositors with commercial banks") / self.scale_dict[country]

    def number_of_commercial_borrowers(self, year: int, country: str | Country) -> float:
        """
        Get number of commercial bank borrowers for a country.

        Parameters
        ----------
        year : int
            Year to get data for
        country : str | Country
            Country identifier

        Returns
        -------
        float
            Number of borrowers (scaled)
        """
        return self.get_value(year, country, "Borrowers from commercial banks") / self.scale_dict[country]

    def total_commercial_deposits(self, year: int, country: str | Country) -> float:
        """
        Get total commercial bank deposits for a country.

        Parameters
        ----------
        year : int
            Year to get data for
        country : str | Country
            Country identifier

        Returns
        -------
        float
            Total deposits in domestic currency

        Notes
        -----
        - Returns value in millions of domestic currency units
        """
        return self.get_value(year, country, "Outstanding deposits with commercial banks") * 1e6

    def total_commercial_loans(self, year: int, country: str | Country) -> float:
        """
        Get total commercial bank loans for a country.

        Parameters
        ----------
        year : int
            Year to get data for
        country : str | Country
            Country identifier

        Returns
        -------
        float
            Total loans in domestic currency

        Notes
        -----
        - Returns value in millions of domestic currency units
        """
        return self.get_value(year, country, "Outstanding loans from commercial banks") * 1e6

    def get_inflation(self, country: Country | str) -> Optional[pd.DataFrame]:
        """
        Get inflation rates from CPI and PPI for a country.

        Parameters
        ----------
        country : Country | str
            Country identifier

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with CPI and PPI inflation rates,
            or None if data not available

        Notes
        -----
        - Returns quarterly data
        - Uses PPI if CPI not available and vice versa
        - Returns None for Argentina (uses central bank data instead)
        - Rates are returned as decimals
        """
        if isinstance(country, Region):
            country = country.parent_country
        if isinstance(country, str):
            country = Country(country)
        if country.value == "ARG":  # using CB data instead
            return None

        country_english = str(country).lower()
        data = self.data["international_financial_statistics"]
        data.rename(columns=data.iloc[0]).drop(data.index[0]).reset_index(drop=True)
        data = data.loc[(data["Attribute"] == "Value") & (data["Country Name"].str.lower() == country_english)]
        data.set_index("Indicator Name", inplace=True)
        if (
            "Prices, Consumer Price Index, All items, Index" not in data.index
            and "Prices, Producer Price Index, All Commodities, Index" not in data.index
        ):
            return None
        elif "Prices, Consumer Price Index, All items, Index" not in data.index:
            data = data.loc[
                [
                    "Prices, Producer Price Index, All Commodities, Index",
                    "Prices, Producer Price Index, All Commodities, Index",
                ]
            ].T.iloc[4:-1]
        elif "Prices, Producer Price Index, All Commodities, Index" not in data.index:
            data = data.loc[
                [
                    "Prices, Consumer Price Index, All items, Index",
                    "Prices, Consumer Price Index, All items, Index",
                ]
            ].T.iloc[4:-1]
        else:
            data = data.loc[
                [
                    "Prices, Consumer Price Index, All items, Index",
                    "Prices, Producer Price Index, All Commodities, Index",
                ]
            ].T.iloc[4:-1]
        data.columns = ["CPI Inflation", "PPI Inflation"]
        data.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[5]) - 2, 1) for ind in data.index]  # noqa
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return data.astype(float).pct_change()

    def get_na_growth_rates(self, country: str | Country) -> pd.DataFrame:
        """
        Get national accounts growth rates for a country.

        This method calculates growth rates for various national accounts components:
        - GDP and its components
        - Consumption (household, NPISH, government)
        - Investment (fixed capital, inventories)
        - Trade (exports and imports of goods and services)

        Parameters
        ----------
        country : str | Country
            Country identifier

        Returns
        -------
        pd.DataFrame
            DataFrame with growth rates for each component

        Notes
        -----
        - Returns quarterly data
        - Uses seasonally adjusted data when available
        - Growth rates are returned as decimals
        - Combines household and NPISH consumption
        """
        if isinstance(country, str):
            country = Country(country)
        country_english = str(country).lower()
        data = self.data["international_financial_statistics"]
        data.rename(columns=data.iloc[0]).drop(data.index[0]).reset_index(drop=True)
        data = data.loc[(data["Attribute"] == "Value") & (data["Country Name"].str.lower() == country_english)]
        data.set_index("Indicator Name", inplace=True)
        data = data.T.iloc[4:-1]

        # Get the data
        fields = {
            "Gross Domestic Product": "GDP",
            "Households Final Consumption Expenditure": "HH Cons",
            "Non-profit Institutions Serving Households (NPISHs) Final Consumption Expenditure": "NPISH Cons",
            "General Government Final Consumption Expenditure": "Gov Cons",
            "Gross Fixed Capital Formation": "Gross Fixed Capital Formation",
            "Changes in Inventories": "Changes in Inventories",
            "Exports of Goods": "Exports of Goods",
            "Exports of Services": "Exports of Services",
            "Exports of Goods and Services": "Exports of Goods and Services",
            "Imports of Goods": "Imports of Goods",
            "Imports of Services": "Imports of Services",
            "Imports of Goods and Services": "Imports of Goods and Services",
        }
        gdp_field = "Gross Domestic Product"
        data_ls = {}

        for field in fields.keys():
            if field + ", Nominal, Seasonally Adjusted, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[field + ", Nominal, Seasonally Adjusted, Domestic Currency"].values
            elif field + ", Nominal, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[field + ", Nominal, Domestic Currency"].values
            elif field + ", Nominal, Unadjusted, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[field + ", Nominal, Unadjusted, Domestic Currency"].values
            elif gdp_field + ", Nominal, Seasonally Adjusted, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[gdp_field + ", Nominal, Seasonally Adjusted, Domestic Currency"].values
            elif gdp_field + ", Nominal, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[gdp_field + ", Nominal, Domestic Currency"].values
            elif gdp_field + ", Nominal, Unadjusted, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[gdp_field + ", Nominal, Unadjusted, Domestic Currency"].values
            else:
                raise ValueError(f"No suitable data found for {country} {field}")

        data = pd.DataFrame(
            data=data_ls,
            index=[pd.Timestamp(int(ind[0:4]), 3 * int(ind[5]) - 2, 1) for ind in data.index],
        ).iloc[0:-1]
        data = data.astype(float)
        data["HH Cons"] = data["HH Cons"] + data["NPISH Cons"]
        data = (data / data.shift(1) - 1.0).iloc[1:]
        data["Gross Output"] = data["GDP"].values
        data["Intermediate Consumption"] = data["GDP"].values
        data = data.loc[:, data.columns != "NPISH Cons"]

        return data

    def get_labour_stats(self, country: str | Country) -> Optional[pd.DataFrame]:
        """
        Get labor market statistics for a country.

        Parameters
        ----------
        country : str | Country
            Country identifier

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with labor market indicators:
            - Labor force size
            - Employment numbers
            - Unemployment numbers and rate
            Returns None if data not available

        Notes
        -----
        - Returns quarterly data
        - Unemployment rate is converted to decimal
        """
        if isinstance(country, str):
            country = Country(country)
        country_english = str(country).lower()
        data = self.data["international_financial_statistics"]
        data.rename(columns=data.iloc[0]).drop(data.index[0]).reset_index(drop=True)
        data = data.loc[(data["Attribute"] == "Value") & (data["Country Name"].str.lower() == country_english)]
        data.set_index("Indicator Name", inplace=True)
        if (
            "Labor Force, Persons, Number of" not in data.index
            or "Employment, Persons, Number of" not in data.index
            or "Unemployment, Persons, Number of" not in data.columns
            or "Labor Markets, Unemployment Rate, Percent" not in data.columns
        ):
            return None
        data = data.loc[
            [
                "Labor Force, Persons, Number of",
                "Employment, Persons, Number of",
                "Unemployment, Persons, Number of",
                "Labor Markets, Unemployment Rate, Percent",
            ]
        ].T.iloc[4:-1]
        data.columns = [
            "Labour Force",
            "Employment Number",
            "Unemployment Number",
            "Unemployment Rate",
        ]
        data.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[5]) - 2, 1) for ind in data.index]  # noqa
        data = data.astype(float)
        data["Unemployment Rate"] /= 100.0
        return data

    def prune(self, prune_date: date):
        """
        Prune data to remove entries after a specified date.

        Parameters
        ----------
        prune_date : date
            Date after which to remove data

        Notes
        -----
        - Modifies bank_demography data in place
        - Uses prune_index utility function
        """
        mask = prune_index(self.data["bank_demography"].columns, prune_date)
        self.data["bank_demography"] = self.data["bank_demography"].loc[:, mask]

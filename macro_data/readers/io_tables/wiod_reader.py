"""
This module provides functionality for reading and processing World Input-Output Database (WIOD) data.
The WIOD is a comprehensive database containing input-output tables for multiple countries and industries,
enabling analysis of international trade flows, global value chains, and economic relationships.

Key Features:
- Read and parse WIOD CSV files with multi-level headers
- Aggregate data across countries and industries
- Calculate various economic metrics:
  * Capital formation and weights
  * Household and government consumption
  * Intermediate inputs and trade flows
  * Import and export totals

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.io_tables.wiod_reader import WIODReader

    # Read WIOD data from CSV
    wiod = WIODReader.from_csv("path/to/wiod_data.csv")

    # Get household consumption for a specific country
    country = "USA"
    hh_cons = wiod.hh_consumption(country)
    hh_weights = wiod.hh_consumption_weights(country)

    # Analyze trade flows
    exports = wiod.total_exports(country)
    imports = wiod.total_imports(country)
    ```

Note:
    All monetary values are in current USD (multiplied by 1e6 after reading).
"""

import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from macro_data.readers.io_tables.util import aggregate_df


# all in current USD
class WIODReader:
    """
    A class for reading and manipulating World Input-Output Database (WIOD) data.

    The WIODReader processes WIOD tables that contain detailed information about economic
    transactions between industries and countries. It provides methods to analyze various
    aspects of these transactions, including consumption patterns, trade flows, and
    input-output relationships.

    Parameters
    ----------
    df : pd.DataFrame
        The input-output table as a pandas DataFrame with multi-level indices for
        countries and industries.
    considered_countries : list[str]
        List of countries to include in the analysis. Countries not in this list
        are aggregated into a "ROW" (Rest of World) category.
    industries : list[str]
        List of industry codes used in the input-output table.

    Attributes
    ----------
    df : pd.DataFrame
        The processed input-output table.
    industries : list[str]
        List of industry codes.
    considered_countries : list[str]
        List of countries included in the analysis.

    Notes
    -----
    - All monetary values are in current USD
    - The data structure uses multi-level indices for both rows and columns
    - Countries not in `considered_countries` are aggregated into "ROW"
    """

    def __init__(self, df: pd.DataFrame, considered_countries: list[str], industries: list[str]):
        self.df = df
        self.industries = industries
        self.considered_countries = considered_countries

    @classmethod
    def from_csv(cls, path: Path | str) -> "WIODReader":
        """
        Create a WIODReader instance from a WIOD CSV file.

        Parameters
        ----------
        path : Path | str
            Path to the WIOD CSV file.

        Returns
        -------
        WIODReader
            A new WIODReader instance initialized with the data from the CSV file.

        Notes
        -----
        - Automatically identifies unique countries and industries from the data
        - Excludes "ROW" and "TOT" from considered countries
        """
        df = cls.read_csv(path)
        countries = df.columns.get_level_values(0).unique()
        considered_countries = [c for c in countries if c not in ["ROW", "TOT"]]
        industries = list(df.loc[considered_countries[0]].index)
        return cls(df, considered_countries=considered_countries, industries=industries)

    @staticmethod
    def read_csv(path: Path | str) -> pd.DataFrame:
        """
        Read and process a WIOD CSV file.

        Parameters
        ----------
        path : Path | str
            Path to the WIOD CSV file.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with proper multi-level indices and columns.

        Notes
        -----
        - Skips the first row and uses multiple header rows
        - Removes unnecessary levels and adjusts index/column names
        - Drops "PURR" and "PURNR" rows
        - Swaps levels to maintain consistent structure
        """
        df = pd.read_csv(path, skiprows=1, header=[1, 2, 3, 4], index_col=[0, 1, 2, 3], thousands=",")
        df.index = df.index.droplevel(1)
        df.index = df.index.droplevel(2)
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.droplevel(2)
        df.drop(index=["PURR", "PURNR"], level=0, inplace=True)
        df = df.swaplevel(0, 1, axis=1)
        df = df.swaplevel(0, 1, axis=0)
        df.index.names = ["CountryInd", "industryInd"]
        df.columns.names = ["CountryCol", "industryCol"]
        return df

    @classmethod
    def agg_from_csv(
        cls,
        path: Path | str,
        considered_countries: list[str],
        aggregation_path: Path,
    ) -> "WIODReader":
        """
        Create a WIODReader instance with aggregated data from a CSV file.

        This method reads a WIOD CSV file and aggregates the data according to
        the specified country list and aggregation rules defined in the
        aggregation JSON file.

        Parameters
        ----------
        path : Path | str
            Path to the WIOD CSV file.
        considered_countries : list[str]
            List of countries to include in the analysis.
        aggregation_path : Path
            Path to the JSON file containing industry aggregation rules.

        Returns
        -------
        WIODReader
            A new WIODReader instance with aggregated data.

        Notes
        -----
        - Countries not in `considered_countries` are aggregated into "ROW"
        - Industry aggregation follows rules in the JSON file
        """
        df = cls.read_csv(path)
        aggregation = json.load(open(aggregation_path))
        df = cls.aggregate_io(considered_countries, df, aggregation)
        industries = list(df.loc[considered_countries[0]].index)
        return cls(df, considered_countries, industries)

    @staticmethod
    def aggregate_io(
        considered_countries: list[str],
        df: pd.DataFrame,
        aggregation: dict[str, list[str]],
    ) -> pd.DataFrame:
        """
        Aggregate an input-output table by countries and industries.

        This method performs two types of aggregation:
        1. Country aggregation: Countries not in `considered_countries` are
           combined into a "ROW" (Rest of World) category
        2. Industry aggregation: Industries are combined according to the
           mapping provided in the aggregation dictionary

        Parameters
        ----------
        considered_countries : list[str]
            List of countries to keep separate (not aggregated into ROW)
        df : pd.DataFrame
            Input-output table to aggregate
        aggregation : dict[str, list[str]]
            Dictionary mapping aggregate industry codes to lists of detailed
            industry codes (e.g., {'A': ['A01', 'A02', 'A03']})

        Returns
        -------
        pd.DataFrame
            Aggregated input-output table with values in USD (multiplied by 1e6)

        Notes
        -----
        - Drops NA values and "TOT" columns after aggregation
        - Converts values to USD by multiplying by 1e6
        """
        col_level_0 = df.columns.get_level_values(0).unique()
        # tracks the countries we want to keep
        keep_level_0 = considered_countries + ["ROW", "TOT"]
        # countries we want to discard
        discard_level_0 = [c for c in col_level_0 if c not in keep_level_0]
        # build the country aggregation dictionary,
        # ie mapping AUS to ROW if Australia is not in considered countries
        country_agg_dict = {c: "ROW" for c in discard_level_0}
        for c in keep_level_0:
            country_agg_dict[c] = c
        aggregated = aggregate_df(aggregation, country_agg_dict, df)
        aggregated.dropna(axis=1, inplace=True)
        aggregated.drop(columns="TOT", level=0, inplace=True)
        aggregated *= 1e6  # units in USD
        return aggregated

    def column_allc(self, country: str, symbol: str) -> pd.Series:
        """
        Calculate the sum of values for a specific symbol across all countries.

        Parameters
        ----------
        country : str
            Target country for the calculation
        symbol : str
            Symbol to sum (e.g., 'GFCF', 'CONS_h')

        Returns
        -------
        pd.Series
            Sum of values for the specified symbol across all countries
        """
        considered_countries_row = self.considered_countries + ["ROW"]
        all_cols = [self.df.loc[col, (country, symbol)].loc[self.industries] for col in considered_countries_row]
        return reduce(lambda a, b: a + b, all_cols)

    def capital_formation(self, country: str) -> np.ndarray:
        """
        Get gross fixed capital formation values for a country.

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            Array of capital formation values by industry
        """
        return self.column_allc(country, "GFCF").values

    def capital_weights(self, country: str) -> np.ndarray:
        """
        Calculate normalized weights of capital formation by industry.

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            Array of normalized capital formation weights
        """
        cap_form = self.capital_formation(country)
        return cap_form / cap_form.sum()

    def hh_consumption(self, country: str) -> np.ndarray:
        """
        Calculate total household consumption (including non-profit).

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            Array of household consumption values by industry
        """
        return self.column_allc(country, "CONS_h") + self.column_allc(country, "CONS_np")

    def hh_consumption_weights(self, country: str) -> np.ndarray:
        """
        Calculate normalized weights of household consumption by industry.

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            Array of normalized household consumption weights
        """
        hh_cons = self.hh_consumption(country)
        return hh_cons / hh_cons.sum()

    def govt_consumption(self, country: str) -> np.ndarray:
        """
        Get government consumption values for a country.

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            Array of government consumption values by industry
        """
        return self.column_allc(country, "CONS_g").values

    def govt_cons_weights(self, country: str) -> np.ndarray:
        """
        Calculate normalized weights of government consumption by industry.

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            Array of normalized government consumption weights
        """
        gov_cons = self.govt_consumption(country)
        return gov_cons / gov_cons.sum()

    def intermediate_inputs(self, country: str) -> np.ndarray:
        """
        Calculate the matrix of intermediate inputs between industries.

        This method computes the flow of goods between industries within a country,
        including imports from other countries.

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            2D array representing the intermediate input flows between industries
        """
        return reduce(
            lambda a, b: a + b,
            [
                self.df.loc[c_prime, country].loc[self.industries, self.industries]
                # flow of goods is row->columns,
                # so we count all incoming goods
                for c_prime in self.considered_countries + ["ROW"]
            ],
        ).values

    def intermediate_input_weights(self, country: str) -> np.ndarray:
        """
        Calculate normalized weights of intermediate inputs between industries.

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            2D array of normalized intermediate input weights
        """
        ii_g = self.intermediate_inputs(country)
        # normalise so that columns sum to 1
        return ii_g / ii_g.sum(axis=0)

    def total_exports(self, country: str) -> np.ndarray:
        """
        Calculate total exports by industry for a country.

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            Array of total export values by industry
        """
        considered_countries_row = self.considered_countries + ["ROW"]
        exports = reduce(
            lambda a, b: a + b,
            (
                self.df.loc[country, c2].loc[self.industries].sum(axis=1).values
                for c2 in considered_countries_row
                if c2 != country
            ),
        )
        return exports

    def total_imports(self, country: str) -> np.ndarray:
        """
        Calculate total imports by industry for a country.

        Parameters
        ----------
        country : str
            Country to analyze

        Returns
        -------
        np.ndarray
            Array of total import values by industry
        """
        considered_countries_row = self.considered_countries + ["ROW"]
        imports = reduce(
            lambda a, b: a + b,
            (
                self.df.loc[c2, country].loc[self.industries].sum(axis=0)
                for c2 in considered_countries_row
                if c2 != country
            ),
        )
        return imports

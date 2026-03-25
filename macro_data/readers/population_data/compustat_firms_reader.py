"""
This module provides functionality for reading and processing Compustat firm-level financial data.
It handles both annual and quarterly data, with support for multiple countries and
automatic currency conversion.

Key Features:
- Read and merge annual and quarterly Compustat data
- Handle multiple countries and currencies
- Automatic missing value imputation
- Support for proxy country data
- Currency conversion capabilities

The module processes various financial metrics including:
- Employment data
- Balance sheet items (assets, liabilities, equity)
- Income statement items (revenue, profits)
- Operational data (inventory)

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.population_data.compustat_firms_reader import CompustatFirmsReader
    from macro_data.configuration.countries import Country

    # Initialize reader with raw data
    reader = CompustatFirmsReader.from_raw_data(
        year=2020,
        quarter=4,
        raw_annual_path=Path("path/to/annual.csv"),
        raw_quarterly_path=Path("path/to/quarterly.csv"),
        countries=["USA", "GBR", Country.FRANCE]
    )

    # Get firm data for a specific country
    usa_firms = reader.get_firm_data("USA")

    # Get proxy data with currency conversion
    proxy_firms = reader.get_proxied_firm_data(
        proxy_country="GBR",
        exchange_rate=1.25
    )
    ```

Note:
    Missing values are imputed using scikit-learn's IterativeImputer.
"""

from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region

# Mapping of Compustat variable codes to descriptive names
var_mapping = {
    "curcdq": "Currency Code",  # Currency identifier
    "emp": "Number of Employees",  # Total employees
    "atq": "Assets",  # Total assets
    "ceqq": "Equity",  # Common equity
    "dlttq": "Debt",  # Long-term debt
    "dptbq": "Deposits",  # Bank deposits
    "invtq": "Inventory",  # Total inventory
    "ltq": "Liabilities",  # Total liabilities
    "revtq": "Revenue",  # Total revenue
    "gpq": "Profits",  # Gross profits
    "gsector": "Sector",  # Industry sector
    "loc": "Country",  # Country location
}

# List of variables containing monetary values
var_numerical = [
    "Assets",  # Total assets
    "Equity",  # Common equity
    "Debt",  # Long-term debt
    "Deposits",  # Bank deposits
    "Inventory",  # Total inventory
    "Liabilities",  # Total liabilities
    "Revenue",  # Total revenue
    "Profits",  # Gross profits
]

# Complete list of variables to keep in processed data
var_keeping = [
    "Number of Employees",  # Workforce size
    "Assets",  # Total assets
    "Equity",  # Common equity
    "Debt",  # Long-term debt
    "Deposits",  # Bank deposits
    "Inventory",  # Total inventory
    "Liabilities",  # Total liabilities
    "Revenue",  # Total revenue
    "Profits",  # Gross profits
    "Sector",  # Industry sector
    "Country",  # Country location
    "Currency Code",  # Currency identifier
]

# Suppress convergence warnings from imputer
simplefilter("ignore", category=ConvergenceWarning)


class CompustatFirmsReader:
    """
    A class for reading and processing Compustat firm-level financial data.

    This class handles the reading and processing of Compustat data, including:
    - Merging annual and quarterly data
    - Filtering by country and time period
    - Imputing missing values
    - Currency conversion for international comparisons

    Parameters
    ----------
    data : pd.DataFrame
        Processed Compustat data with standardized columns

    Attributes
    ----------
    data : pd.DataFrame
        Processed firm-level data indexed by country
    numerical_columns : list[str]
        List of columns containing monetary values

    Notes
    -----
    - Missing values are imputed using scikit-learn's IterativeImputer
    - All monetary values are in their original currencies
    """

    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self.data = data

    @classmethod
    def from_raw_data(
        cls,
        year: int,
        quarter: int,
        raw_annual_path: Path | str,
        raw_quarterly_path: Path | str,
        countries: list[str | Country],
    ):
        """
        Create a CompustatFirmsReader instance from raw Compustat files.

        This method:
        1. Reads annual and quarterly data
        2. Filters for specific time period
        3. Merges the datasets
        4. Processes and cleans the data
        5. Imputes missing values

        Parameters
        ----------
        year : int
            Year to filter data for
        quarter : int
            Quarter to filter data for (1-4)
        raw_annual_path : Path | str
            Path to annual Compustat data file
        raw_quarterly_path : Path | str
            Path to quarterly Compustat data file
        countries : list[str | Country]
            List of countries to include in the data

        Returns
        -------
        CompustatFirmsReader
            Initialized reader with processed data

        Notes
        -----
        - Data is filtered to match the specified year and quarter
        - Countries can be specified as strings or Country enum values
        - Missing values are imputed across all numeric columns
        """
        raw_annual_data = pd.read_csv(
            raw_annual_path,
            encoding="unicode_escape",
            engine="pyarrow",
        )
        raw_quarterly_data = pd.read_csv(
            raw_quarterly_path,
            encoding="unicode_escape",
            engine="pyarrow",
        )
        raw_quarterly_data = raw_quarterly_data[
            np.logical_and(
                raw_quarterly_data["fyearq"] == year,
                raw_quarterly_data["fqtr"] == quarter,
            )
        ]

        # Clean and filter data
        annual_data = raw_annual_data.dropna(axis=0, how="all").dropna(axis=1, how="all")
        quarterly_data = raw_quarterly_data.dropna(axis=0, how="all").dropna(axis=1, how="all")

        # Filter for specified countries
        annual_data = annual_data[annual_data["loc"].isin(countries)]
        quarterly_data = quarterly_data[quarterly_data["loc"].isin(countries)]

        # Merge annual and quarterly data
        data = pd.merge(
            quarterly_data,
            annual_data,
            on="conm",
        )

        # Clean up merged data
        data.drop(columns=["loc_y"], inplace=True)
        data.rename(columns={"loc_x": "loc"}, inplace=True)
        data = data.rename(columns=var_mapping)
        data = data[var_keeping]

        # Impute missing values
        column_selection = [col for col in data.columns if col not in ["Country", "Currency Code"]]
        data.loc[:, column_selection] = IterativeImputer().fit_transform(data[column_selection].values)

        data.set_index("Country", inplace=True)

        return cls(data)

    @property
    def numerical_columns(self):
        """
        Get the list of columns containing monetary values.

        Returns
        -------
        list[str]
            Names of columns containing monetary values
        """
        return var_numerical

    def get_firm_data(self, country: str | Country | Region) -> pd.DataFrame:
        """
        Get firm-level data for a specific country.

        Parameters
        ----------
        country : str | Country
            Country to get data for (string or Country enum)

        Returns
        -------
        pd.DataFrame
            Firm-level data for the specified country
        """
        if isinstance(country, Region):
            country = country.parent_country

        if isinstance(country, Country):
            country = country.value
        return self.data.loc[country]

    def get_proxied_firm_data(self, proxy_country: str | Country, exchange_rate: float) -> pd.DataFrame:
        """
        Get firm-level data from a proxy country with currency conversion.

        This method is useful when direct data for a country is not available
        and data from another country needs to be used as a proxy.

        Parameters
        ----------
        proxy_country : str | Country
            Country to use as proxy (string or Country enum)
        exchange_rate : float
            Exchange rate to convert monetary values

        Returns
        -------
        pd.DataFrame
            Converted firm-level data from the proxy country

        Notes
        -----
        - Only monetary values are converted
        - Non-monetary fields (e.g., employee counts) are unchanged
        """
        if isinstance(proxy_country, Country):
            proxy_country = proxy_country.value
        proxied = self.data.loc[proxy_country, self.numerical_columns].copy()
        proxied = proxied * exchange_rate
        return proxied

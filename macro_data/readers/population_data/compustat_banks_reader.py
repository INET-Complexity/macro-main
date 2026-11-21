"""
This module provides functionality for reading and processing Compustat bank-level financial data.
It handles quarterly bank financial statements with support for multiple countries and
automatic currency conversion.

Key Features:
- Read and process quarterly Compustat bank data
- Handle multiple countries and currencies
- Automatic missing value imputation
- Support for proxy data using US banks
- Currency conversion capabilities

The module processes various banking metrics including:
- Balance sheet items (assets, liabilities, equity)
- Funding sources (deposits, debt)
- Debt dynamics (issuance, reduction)
- Income data

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.population_data.compustat_banks_reader import CompustatBanksReader
    from macro_data.configuration.countries import Country

    # Initialize reader with raw data
    reader = CompustatBanksReader.from_raw_data(
        year=2020,
        quarter=4,
        raw_quarterly_path=Path("path/to/quarterly.csv"),
        countries=["GBR", Country.FRANCE],
        proxy_with_us=True  # Include US banks for proxying
    )

    # Get bank data for a specific country
    uk_banks = reader.get_country_data(
        country="GBR",
        exchange_rate=1.25
    )

    # Get proxy data using another country's banks
    proxy_banks = reader.get_proxied_country_data(
        proxy_country="USA",
        exchange_rate=1.25
    )
    ```

Note:
    Missing values are imputed using scikit-learn's IterativeImputer,
    with imputation performed separately for each country's banks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa

from macro_data.configuration.countries import Country

# Mapping of Compustat variable codes to descriptive names
var_mapping = {
    "fqtr": "Quarter",  # Fiscal quarter (1-4)
    "fyearq": "Year",  # Fiscal year
    "loc": "Country",  # Bank location
    "curcdq": "Currency Code",  # Currency identifier
    "atq": "Assets",  # Total assets
    "ciq": "Income",  # Total income
    "dlttq": "Debt",  # Long-term debt
    "dptcq": "Deposits",  # Customer deposits
    "ltq": "Liabilities",  # Total liabilities
    "teqq": "Equity",  # Total equity
    "dltisy": "Long-term Debt Issuance",  # New debt issued
    "dltry": "Long-term Debt Reduction",  # Debt repayments
}

# List of variables containing monetary values
var_numerical = [
    "Assets",  # Total assets
    "Income",  # Total income
    "Debt",  # Long-term debt
    "Deposits",  # Customer deposits
    "Liabilities",  # Total liabilities
    "Equity",  # Total equity
    "Long-term Debt Issuance",  # New debt issued
    "Long-term Debt Reduction",  # Debt repayments
]

# Core balance sheet items to keep in processed data
var_keeping = [
    "Assets",  # Total assets
    "Debt",  # Long-term debt
    "Deposits",  # Customer deposits
    "Liabilities",  # Total liabilities
    "Equity",  # Total equity
]


class CompustatBanksReader:
    """
    A class for reading and processing Compustat bank-level financial data.

    This class handles the reading and processing of Compustat bank data, including:
    - Reading quarterly financial statements
    - Filtering by country and time period
    - Imputing missing values
    - Currency conversion for international comparisons

    Parameters
    ----------
    data : pd.DataFrame
        Processed Compustat bank data with standardized columns

    Attributes
    ----------
    data : pd.DataFrame
        Processed bank-level data indexed by country
    numerical_columns : list[str]
        List of columns containing monetary values

    Notes
    -----
    - Missing values are imputed separately for each country's banks
    - All monetary values are in their original currencies unless converted
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
        raw_quarterly_path: Path | str,
        countries: list[str | Country],
        proxy_with_us: bool = True,
    ):
        """
        Create a CompustatBanksReader instance from raw Compustat data.

        This method:
        1. Reads quarterly bank data
        2. Filters for specific time period
        3. Processes and cleans the data
        4. Imputes missing values by country

        Parameters
        ----------
        year : int
            Year to filter data for
        quarter : int
            Quarter to filter data for (1-4)
        raw_quarterly_path : Path | str
            Path to quarterly Compustat data file
        countries : list[str | Country]
            List of countries to include in the data
        proxy_with_us : bool, optional
            Whether to include US banks for proxying (default: True)

        Returns
        -------
        CompustatBanksReader
            Initialized reader with processed data

        Notes
        -----
        - Data is filtered to match the specified year and quarter
        - Countries can be specified as strings or Country enum values
        - Missing values are imputed separately for each country
        - US banks are included if proxy_with_us is True
        """
        raw_data = pd.read_csv(raw_quarterly_path, encoding="unicode_escape", engine="pyarrow")

        # Filter for time period
        data = raw_data[np.logical_and(raw_data["fyearq"] == year, raw_data["fqtr"] == quarter)]

        # Add US banks if needed for proxying
        if proxy_with_us:
            countries += [Country("USA")]

        # Filter for specified countries
        data = data[data["loc"].isin(countries)]

        # Clean and standardize data
        data.rename(columns={"loc": "Country"}, inplace=True)
        data.set_index("Country", inplace=True)
        data = data[[col for col in var_mapping.keys() if col in data.columns]]
        data.rename(columns=var_mapping, inplace=True)
        data = data[var_keeping]

        # Impute missing values by country
        for c in data.index.get_level_values(0).unique():
            data_values = data.loc[c].values
            if len(data_values) == 1:
                data_values = data_values.reshape(1, -1)
            data.loc[c] = IterativeImputer().fit_transform(data_values)

        return cls(data)

    @property
    def numerical_columns(self):
        """
        Get the list of columns containing monetary values.

        Returns
        -------
        list[str]
            Names of columns containing monetary values that are present in the data
        """
        return [col for col in var_numerical if col in self.data.columns]

    def get_country_data(self, country: str, exchange_rate: float) -> pd.DataFrame:
        """
        Get bank-level data for a specific country.

        For non-US countries, this method returns US bank data converted to
        the target country's currency using the provided exchange rate.

        Parameters
        ----------
        country : str
            Country to get data for
        exchange_rate : float
            Exchange rate for currency conversion (if using US proxy)

        Returns
        -------
        pd.DataFrame
            Bank-level data for the specified country

        Notes
        -----
        - Returns actual data for US banks
        - Returns converted US bank data for other countries
        """
        if country == "USA":
            return self.data.loc[country]
        else:
            proxied = self.data.loc["USA"].copy()
            proxied[self.numerical_columns] *= exchange_rate
            return proxied

    def get_proxied_country_data(self, proxy_country: str | Country, exchange_rate: float):
        """
        Get bank-level data from a proxy country with currency conversion.

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
            Converted bank-level data from the proxy country

        Notes
        -----
        - Only monetary values are converted
        - Non-monetary fields are unchanged
        """
        if isinstance(proxy_country, Country):
            proxy_country = proxy_country.value
        proxied = self.data.loc[proxy_country, self.numerical_columns].copy()
        proxied = proxied * exchange_rate
        return proxied

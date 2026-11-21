"""
This module provides functionality for reading and processing European Central Bank (ECB)
interest rate data. It handles various types of lending rates across Eurozone countries,
including rates for firm loans, household consumption loans, and mortgages.

Key Features:
- Read ECB interest rate data from CSV files
- Support for firm and household lending rates
- Automatic country code conversion
- Quarterly data resampling
- Proxy mechanism for missing data

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.economic_data.ecb_reader import ECBReader
    from macro_data.configuration.countries import Country

    # Initialize reader with data directory
    reader = ECBReader(
        path=Path("path/to/ecb/data"),
        proxy_country=Country.GERMANY
    )

    # Get lending rates for France
    firm_rates = reader.get_firm_rates("FRA")
    mortgage_rates = reader.get_household_mortgage_rates("FRA")
    consumption_rates = reader.get_household_consumption_rates("FRA")
    ```

Note:
    All rates are returned as decimals (e.g., 0.05 for 5%) for direct use in calculations.
"""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from macro_data.configuration.countries import Country


def country_code_switch(codes: Iterable[str]) -> list[str]:
    """
    Convert two-letter country codes to three-letter format.

    Parameters
    ----------
    codes : Iterable[str]
        Collection of two-letter country codes (e.g., 'DE', 'FR')

    Returns
    -------
    list[str]
        List of three-letter country codes (e.g., 'DEU', 'FRA')
    """
    return [Country.convert_two_letter_to_three(c) for c in codes]


def preprocess_df(df: pd.DataFrame, freq: str = "QS") -> Optional[pd.Series]:
    """
    Preprocess ECB data by cleaning columns and resampling.

    This function:
    1. Removes unnecessary time period columns
    2. Converts country codes from two to three letters
    3. Resamples data to specified frequency
    4. Handles missing values

    Parameters
    ----------
    df : pd.DataFrame
        Raw ECB data with country columns
    freq : str, optional
        Pandas frequency string for resampling (default: 'QS' for start of quarter)

    Returns
    -------
    Optional[pd.Series]
        Processed time series data, or None if processing fails

    Notes
    -----
    - Assumes country codes are in the last two characters of column names
    - Removes 'U2' column (Euro area aggregate)
    - Returns mean values when resampling
    """
    df.drop(columns="TIME PERIOD", inplace=True)
    df.columns = [c[-26:-24] for c in df.columns]
    df.drop(columns="U2", inplace=True)
    df.columns = country_code_switch(df.columns)
    data = df.resample(freq).mean()
    data.freq = None
    return data


class ECBReader:
    """
    A class for reading and processing European Central Bank interest rate data.

    This class provides access to three types of lending rates:
    1. Firm loan rates
    2. Household consumption loan rates
    3. Household mortgage rates

    Parameters
    ----------
    path : Path | str
        Path to directory containing ECB data files
    proxy_country : Country, optional
        Country to use as proxy when data is missing (default: Germany)

    Attributes
    ----------
    proxy_country : Country
        Country used for proxying missing data
    data : dict
        Dictionary containing processed rate data for each loan type

    Notes
    -----
    - Expected file names: firm_loans.csv, household_loans_for_consumption.csv,
      household_loans_for_mortgages.csv
    - All rates are stored as percentages but returned as decimals
    """

    def __init__(
        self,
        path: Path | str,
        proxy_country: Country = Country("DEU"),
    ):
        # For proxying
        self.proxy_country = proxy_country

        # Load data files
        self.data = {}
        for f in [
            "firm_loans",
            "household_loans_for_consumption",
            "household_loans_for_mortgages",
        ]:
            filepath = path / (f + ".csv")
            self.data[f] = preprocess_df(pd.read_csv(filepath, index_col="DATE", parse_dates=True))

    def get_firm_rates(self, country_name: str) -> Optional[pd.Series]:
        """
        Get firm loan interest rates for a specific country.

        Parameters
        ----------
        country_name : str
            Three-letter country code (e.g., 'DEU', 'FRA')

        Returns
        -------
        Optional[pd.Series]
            Time series of firm loan rates as decimals,
            or None if country not found

        Notes
        -----
        - Returns rates as decimals (e.g., 0.05 for 5%)
        """
        df = self.data["firm_loans"].copy()
        if country_name in df.columns:
            return df[country_name] / 100.0
        else:
            return None

    def get_household_consumption_rates(self, country_name: str) -> Optional[pd.Series]:
        """
        Get household consumption loan rates for a specific country.

        Parameters
        ----------
        country_name : str
            Three-letter country code (e.g., 'DEU', 'FRA')

        Returns
        -------
        Optional[pd.Series]
            Time series of consumption loan rates as decimals,
            or None if country not found

        Notes
        -----
        - Returns rates as decimals (e.g., 0.05 for 5%)
        """
        df = self.data["household_loans_for_consumption"].copy()
        if country_name in df.columns:
            return df[country_name] / 100.0
        else:
            return None

    def get_household_mortgage_rates(self, country_name: str) -> Optional[pd.Series]:
        """
        Get household mortgage rates for a specific country.

        Parameters
        ----------
        country_name : str
            Three-letter country code (e.g., 'DEU', 'FRA')

        Returns
        -------
        Optional[pd.Series]
            Time series of mortgage rates as decimals,
            or None if country not found

        Notes
        -----
        - Returns rates as decimals (e.g., 0.05 for 5%)
        """
        df = self.data["household_loans_for_mortgages"].copy()
        if country_name in df.columns:
            return df[country_name] / 100.0
        else:
            return None

"""
This module provides functionality for reading and processing data from the ECB's Household
Finance and Consumption Survey (HFCS). The HFCS is a detailed survey that collects household-level
data on households' finances and consumption patterns across European countries.

Key Features:
- Read and process HFCS survey data from multiple waves
- Handle both household and individual level data
- Convert monetary values to local currency units
- Map standardized variable names
- Filter and clean survey responses

The module supports reading various types of HFCS data:
1. Individual data (P files): Personal characteristics, income, employment
2. Household data (H files): Assets, liabilities, consumption
3. Derived data (D files): Calculated variables and aggregates

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.population_data.hfcs_reader import HFCSReader
    from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader

    # Initialize exchange rates reader
    exchange_rates = ExchangeRatesReader(...)

    # Read HFCS data for Germany in 2017
    hfcs = HFCSReader.from_csv(
        country_name="Germany",
        country_name_short="DE",
        year=2017,
        hfcs_data_path=Path("path/to/hfcs/data"),
        exchange_rates=exchange_rates
    )

    # Access household and individual data
    households = hfcs.households_df
    individuals = hfcs.individuals_df
    ```

Note:
    All monetary values are converted to local currency units using the provided
    exchange rates.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader

# Mapping of HFCS variable codes to descriptive names
var_mapping = {
    "ID": "ID",  # Unique identifier
    "id": "ID",  # Alternative ID format
    "HID": "Corresponding Household ID",  # Link to household
    "hid": "Corresponding Household ID",  # Alternative household link
    "iid": "Corresponding Individuals ID",  # Individual within household
    "HW0010": "Weight",  # Survey weight
    "DHHTYPE": "Type",  # Household type
    "RA0200": "Gender",  # Gender of individual
    "RA0300": "Age",  # Age of individual
    "PA0200": "Education",  # Education level
    "PE0100a": "Labour Status",  # Employment status
    "PE0400": "Employment Industry",  # Industry of employment
    "PG0110": "Employee Income",  # Income from employment
    "PG0210": "Self-Employment Income",  # Income from self-employment
    "DI1300": "Rental Income from Real Estate",  # Income from property
    "DI1400": "Income from Financial Assets",  # Investment income
    "DI1500": "Income from Pensions",  # Pension income
    "DI1620": "Regular Social Transfers",  # Social benefits
    "DI2000": "Income",  # Total income
    "PG0510": "Income from Unemployment Benefits",  # Unemployment benefits
    "DA1110": "Value of the Main Residence",  # Primary home value
    "DA1120": "Value of other Properties",  # Other real estate
    "DA1130": "Value of Household Vehicles",  # Vehicle assets
    "DA1131": "Value of Household Valuables",  # Valuable items
    "DA1140": "Value of Self-Employment Businesses",  # Business assets
    "DA2101": "Wealth in Deposits",  # Bank deposits
    "DA2102": "Mutual Funds",  # Investment funds
    "DA2103": "Bonds",  # Bond holdings
    "DA2104": "Value of Private Businesses",  # Private equity
    "DA2105": "Shares",  # Stock holdings
    "DA2106": "Managed Accounts",  # Managed investments
    "DA2107": "Money owed to Households",  # Receivables
    "DA2108": "Other Assets",  # Miscellaneous assets
    "DA2109": "Voluntary Pension",  # Private pension
    "DL1110": "Outstanding Balance of HMR Mortgages",  # Home mortgage
    "DL1120": "Outstanding Balance of Mortgages on other Properties",  # Other mortgages
    "DL1210": "Outstanding Balance of Credit Line",  # Credit lines
    "DL1220": "Outstanding Balance of Credit Card Debt",  # Credit card debt
    "DL1230": "Outstanding Balance of other Non-Mortgage Loans",  # Other loans
    "HB0300": "Tenure Status of the Main Residence",  # Housing tenure
    "HB2300": "Rent Paid",  # Rental payments
    "HB2410": "Number of Properties other than Household Main Residence",  # Property count
    "DOCOGOODP": "Consumption of Consumer Goods/Services as a Share of Income",  # Consumption ratio
    "HI0220": "Amount spent on Consumption of Goods and Services",  # Total consumption
}

# List of variables containing monetary values that need currency conversion
var_numerical = [
    "Income",  # Total income
    "Employee Income",  # Employment earnings
    "Self-Employment Income",  # Business income
    "Rental Income from Real Estate",  # Property income
    "Income from Financial Assets",  # Investment returns
    "Income from Pensions",  # Pension payments
    "Regular Social Transfers",  # Social benefits
    "Income from Unemployment Benefits",  # Unemployment support
    "Value of the Main Residence",  # Home value
    "Value of other Properties",  # Other property value
    "Value of Household Vehicles",  # Vehicle worth
    "Value of Household Valuables",  # Valuables worth
    "Value of Self-Employment Businesses",  # Business value
    "Wealth in Deposits",  # Bank balances
    "Mutual Funds",  # Fund investments
    "Bonds",  # Bond investments
    "Value of Private Businesses",  # Private equity
    "Shares",  # Stock investments
    "Managed Accounts",  # Managed portfolios
    "Money owed to Households",  # Receivables
    "Other Assets",  # Other assets
    "Voluntary Pension",  # Private pension
    "Outstanding Balance of HMR Mortgages",  # Home loan
    "Outstanding Balance of Mortgages on other Properties",  # Other mortgages
    "Outstanding Balance of Credit Line",  # Credit line
    "Outstanding Balance of Credit Card Debt",  # Card debt
    "Outstanding Balance of other Non-Mortgage Loans",  # Other debt
    "Rent Paid",  # Rent expense
    "Amount spent on Consumption of Goods and Services",  # Total spending
    "Consumption of Consumer Goods/Services as a Share of Income",  # Spending ratio
]


class HFCSReader:
    """
    A class for reading and processing Household Finance and Consumption Survey (HFCS) data.

    This class handles the reading and initial processing of HFCS data, including:
    - Loading multiple survey waves
    - Converting monetary values to local currency
    - Joining household and derived data
    - Filtering by country
    - Standardizing variable names

    Parameters
    ----------
    country_name_short : str
        Two-letter country code (e.g., "DE" for Germany)
    individuals_df : pd.DataFrame
        DataFrame containing individual-level survey data
    households_df : pd.DataFrame
        DataFrame containing household-level survey data

    Attributes
    ----------
    country_name_short : str
        Two-letter country code
    individuals_df : pd.DataFrame
        Processed individual-level data
    households_df : pd.DataFrame
        Processed household-level data
    """

    def __init__(
        self,
        country_name_short: str,
        individuals_df: pd.DataFrame,
        households_df: pd.DataFrame,
    ):
        self.country_name_short = country_name_short
        self.individuals_df = individuals_df
        self.households_df = households_df

    @classmethod
    def from_csv(
        cls,
        country_name: str,
        country_name_short: str,
        year: int,
        hfcs_data_path: Path,
        exchange_rates: ExchangeRatesReader,
        num_surveys: int = 5,
    ) -> "HFCSReader":
        """
        Create a HFCSReader instance from CSV files.

        This method reads and processes multiple HFCS survey files, including:
        - Individual (P) files: Personal characteristics and income
        - Household (H) files: Household assets and liabilities
        - Derived (D) files: Calculated variables

        Parameters
        ----------
        country_name : str
            Full country name (e.g., "Germany")
        country_name_short : str
            Two-letter country code (e.g., "DE")
        year : int
            Survey year
        hfcs_data_path : Path
            Base path to HFCS data files
        exchange_rates : ExchangeRatesReader
            Exchange rate converter for monetary values
        num_surveys : int, optional
            Number of survey waves to read (default: 5)

        Returns
        -------
        HFCSReader
            Initialized reader with processed survey data

        Notes
        -----
        - Files are expected to be named P1.csv, H1.csv, D1.csv, etc.
        - All monetary values are converted to local currency
        - Derived data is joined with household data
        """
        # Take default paths
        individuals_paths = [hfcs_data_path / str(year) / ("P" + str(i) + ".csv") for i in range(1, num_surveys + 1)]
        households_paths = [hfcs_data_path / str(year) / ("H" + str(i) + ".csv") for i in range(1, num_surveys + 1)]
        derived_paths = [hfcs_data_path / str(year) / ("D" + str(i) + ".csv") for i in range(1, num_surveys + 1)]

        # Read data on individuals
        if len(individuals_paths) > 0:
            individuals_df = pd.concat(
                [
                    cls.read_csv(
                        path=ind_path,
                        country_name=country_name,
                        country_name_short=country_name_short,
                        year=year,
                        exchange_rates=exchange_rates,
                    )
                    for ind_path in individuals_paths
                ],
                axis=0,
            )
        else:
            individuals_df = pd.DataFrame()

        # Read data on households
        if len(households_paths) > 0:
            households_df = pd.concat(
                [
                    cls.read_csv(
                        path=hh_path,
                        country_name=country_name,
                        country_name_short=country_name_short,
                        year=year,
                        exchange_rates=exchange_rates,
                    )
                    for hh_path in households_paths
                ],
                axis=0,
            )
        else:
            households_df = pd.DataFrame()

        # Read derived data
        if len(derived_paths) > 0:
            derived_df = pd.concat(
                [
                    cls.read_csv(
                        path=der_path,
                        country_name=country_name,
                        country_name_short=country_name_short,
                        year=year,
                        exchange_rates=exchange_rates,
                    )
                    for der_path in derived_paths
                ],
                axis=0,
            )
            derived_df.drop("Weight", axis=1, inplace=True)
        else:
            derived_df = pd.DataFrame()

        # Join the derived data with the household data
        households_df = households_df.join(derived_df)

        return cls(
            country_name_short=country_name_short,
            individuals_df=individuals_df,
            households_df=households_df,
        )

    @staticmethod
    def read_csv(
        path: Path | str,
        country_name: str,
        country_name_short: str,
        year: int,
        exchange_rates: ExchangeRatesReader,
    ) -> pd.DataFrame:
        """
        Read and process a single HFCS CSV file.

        This method:
        1. Reads the CSV file
        2. Filters for the specified country
        3. Maps variable names to standardized format
        4. Converts monetary values to local currency
        5. Handles missing values and data types

        Parameters
        ----------
        path : Path | str
            Path to the CSV file
        country_name : str
            Full country name for exchange rate lookup
        country_name_short : str
            Two-letter country code for filtering
        year : int
            Year for exchange rate lookup
        exchange_rates : ExchangeRatesReader
            Exchange rate converter

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with standardized columns and local currency values

        Notes
        -----
        - Missing values ('A', 'M') are converted to NaN
        - Monetary values are converted from EUR to local currency
        - Only variables in var_mapping are kept
        """
        # Load data
        df = pd.read_csv(path, encoding="unicode_escape", engine="pyarrow")

        # Filter for country and keep only mapped variables
        df = df[df["SA0100"] == country_name_short]
        df = df[[col for col in var_mapping.keys() if col in df.columns]]
        df.rename(columns=var_mapping, inplace=True)
        df.set_index("ID", inplace=True)

        # Convert monetary values to local currency
        var_numerical_union = [v for v in var_numerical if v in df.columns]
        df.loc[:, var_numerical_union] = df.loc[:, var_numerical_union].replace(["A", "M"], np.nan).astype(float)
        df.loc[:, var_numerical_union] = exchange_rates.from_eur_to_lcu(
            country=country_name,
            year=year,
        ) * df.loc[
            :, var_numerical_union
        ].apply(pd.to_numeric, errors="coerce")
        return df

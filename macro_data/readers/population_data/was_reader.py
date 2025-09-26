"""
This module provides functionality for reading and processing data from the ONS's Wealth and Assets Survey (WAS). 
The WAS is a detailed survey that collects household-level data on households' finances and consumption patterns 
for Great Britain (England, Wales, and Scotland).

Key Features:
- Read and process WAS survey data from multiple waves (Rounds 1-8)
- Handle both household and individual level data
- Convert monetary values to local currency units (GBP)
- Map standardized variable names from WAS to HFCS-compatible format
- Filter and clean survey responses
- Support for Stata (.dta) file format

The module supports reading various types of WAS data:
1. Person files: Personal characteristics, income, employment
2. Household files: Assets, liabilities, housing characteristics

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.population_data.was_reader import WASReader
    from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader

    # Initialize exchange rates reader
    exchange_rates = ExchangeRatesReader(...)

    # Read WAS data for Great Britain in 2022
    was = WASReader.from_stata(
        country_name="United Kingdom",
        country_name_short="GB",
        year=2022,
        was_data_path=Path("path/to/was/data"),
        exchange_rates=exchange_rates
    )

    # Access household and individual data
    households = was.households_df
    individuals = was.individuals_df
    ```

Note:
    All monetary values are converted to local currency units using the provided
    exchange rates. WAS data is already in GBP, so conversion is typically 1:1.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader

# Base mapping of WAS variable codes to standardized HFCS-compatible names
# This mapping uses placeholders for the round/wave suffix that will be dynamically replaced
# Based on the comprehensive variable mapping provided by the user
base_var_mapping = {
    # Individual Characteristics
    "pidno": "ID",  # Personal identifier from WAS survey
    "person{round}": "iid",  # Individual within household
    "CASE{round}": "HID",  # Household identifier
    "{round}xsperswgt": "Weight",  # Person weight from WAS survey
    "sex{round}": "Gender",  # Sex from WAS survey
    "DVAge17{round}": "Age",  # Age from WAS survey
    "edlevel{round}": "Education",  # Education level from WAS survey
    "wrking{round}": "Labour Status",  # Working status from WAS survey
    "siccode{round}": "Employment Industry",  # Industry SIC codes from WAS survey
    "hholdtype{round}": "Type",  # Household type from WAS survey
    
    # Income Sources
    "dvgrspayannual{round}": "Employee Income",  # Gross annual income from WAS survey
    "dvnetpayannual{round}": "Employee Income Net",  # Net annual income from WAS survey
    "agp_ashe": "Employee Income ASHE",  # Annual gross pay from ASHE survey
    "dvsegrspay{round}": "Self-Employment Income",  # Gross annual self-employed income
    "dvsenetpay{round}": "Self-Employment Income Net",  # Net annual self-employed income
    "DVGISE{round}": "Self-Employment Income Total",  # Total Annual Gross self employed income
    "dvrentincamannual{round}": "Rental Income from Real Estate",  # Annual rental income
    "DVGrsRentAmtAnnual{round}_aggr": "Rental Income Gross",  # Gross rental income
    "DVNetRentAmtAnnual{round}_aggr": "Rental Income Net",  # Net rental income
    "fincv{round}": "Income from Financial Assets",  # Total income from investments
    "DVGIINV{round}_aggr": "Investment Income Gross",  # Gross investment income
    "DVNIINV{round}_aggr": "Investment Income Net",  # Net investment income
    "DVGIPPen{round}_aggr": "Income from Pensions",  # Annual Household Income - Gross occupational or private pension
    "DVBenefitAnnual{round}_aggr": "Regular Social Transfers",  # Total benefits received
    "wageben1{round}": "Working Age Benefits",  # Working Age Benefits
    "disben1{round}": "Disability Benefits",  # Disability Benefits
    "penben1{round}": "Pensioner Benefits",  # Pensioner Benefits
    "Dvtotgir{round}": "Income",  # Total gross regular household annual income
    "unidk{round}": "Unemployment Benefit Indicator",  # Unemployment benefit indicators
    "unifdk{round}": "Unemployment Benefit Indicator 2",  # Unemployment benefit indicators
    
    # Household Assets
    "hvalue{round}": "Value of the Main Residence",  # Current value of main residence
    "DVHValue{round}": "Value of the Main Residence Alt",  # Value of main residence
    "DVOPrVal{round}": "Value of other Properties",  # Total value of other houses
    "uvals{round}": "Value of Second Homes",  # Value of second homes
    "OthPropVal{round}": "Value of Other Property",  # Total value of other property excluding main property
    "DVProperty{round}": "Total Property Value",  # Sum of all property values
    "Dvtotvehval{round}": "Value of Household Vehicles",  # Total value of all vehicles
    "Allgd{round}": "Value of Household Valuables",  # Value of all household goods and collectables
    "bworthb{round}": "Value of Self-Employment Businesses",  # Approximate value of share of business
    "DVSaVal{round}_SUM": "Wealth in Deposits",  # Total value of savings accounts
    "DVFFAssets{round}_SUM": "Formal Financial Assets",  # Total value of formal financial assets (covers mutual funds, bonds, shares, managed accounts)
    "DVFInvOtV{round}": "Other Assets",  # Value of other investments
    "TOTPEN{round}": "Voluntary Pension",  # Total value of individual pension wealth
    
    # Household Liabilities
    "dburdh{round}": "Household Debt Burden",  # Burden of mortgage and other debt on household
    "Totmort{round}": "Outstanding Balance of HMR Mortgages",  # Total mortgage on main residence
    "DVHseDebt{round}_sum": "Other Property Debt",  # Total debt houses not main residence
    "OthMort{round}_sum": "Other Property Mortgage",  # Total property debt excluding main residence
    "TOTCSC{round}_aggr": "Outstanding Balance of Credit Card Debt",  # Hhold total outstanding credit/store/charge card balance
    "dburd{round}": "Non-Mortgage Debt Burden",  # Burden from non-mortgage debt
    
    # Housing Characteristics
    "hhldr{round}": "Tenure Status of the Main Residence",  # Whether owns or rents accommodation
    "ten1{round}": "Tenure",  # Tenure
    "llord{round}": "Landlord Category",  # Landlord – category of landlord
    "dvrentpaid{round}": "Rent Paid",  # How much is usual household rent
    "unumhs{round}": "Number of Second Homes",  # Number of second homes
    "unumhs{round}_i": "Number of Second Homes Imputed",  # Imputed Number of second homes
    "unumhs{round}_iflag": "Second Homes Imputation Flag",  # Imputation flag
    "ubuylet{round}": "Number of Buy to Let Properties",  # number of buy to let properties
    "ubuylet{round}_i": "Buy to Let Properties Imputed",  # Imputed number of buy to let properties
    "ubuylet{round}_iflag": "Buy to Let Imputation Flag",  # Imputation flag
    "unumbd{round}": "Number of Buildings",  # Number of buildings
    "unumbd{round}_i": "Buildings Imputed",  # imputed number of buildings
    "unumbd{round}_iflag": "Buildings Imputation Flag",  # imputation flag
    "unumla{round}": "Number of Land Pieces",  # number of pieces of land
    "unumla{round}_i": "Land Pieces Imputed",  # imputed number of pieces of land
    "unumla{round}_iflag": "Land Imputation Flag",  # imputation flag
    "unumov{round}": "Number of Overseas Properties",  # number of properties overseas
    "unumov{round}_i": "Overseas Properties Imputed",  # imputed number of properties overseas
    "unumov{round}_iflag": "Overseas Imputation Flag",  # imputation flag
    "unumre{round}": "Number of Other Properties",  # number of other properties
    "unumre{round}_i": "Other Properties Imputed",  # imputed number of other properties
    "unumre{round}_iflag": "Other Properties Imputation Flag",  # imputation flag
}


def get_var_mapping(round_number: int) -> dict[str, str]:
    """
    Generate variable mapping for a specific WAS round/wave.
    
    Parameters
    ----------
    round_number : int
        The round/wave number (1-8)
        
    Returns
    -------
    dict[str, str]
        Variable mapping with appropriate round/wave suffix
        
    Notes
    -----
    - For rounds 1-5: uses 'w1', 'w2', 'w3', 'w4', 'w5' (waves)
    - For rounds 6-8: uses 'r6', 'r7', 'r8' (rounds)
    """
    # Determine the suffix based on round number
    if round_number <= 5:
        suffix = f"w{round_number}"
    else:
        suffix = f"r{round_number}"
    
    # Generate the mapping by replacing the placeholder
    var_mapping = {}
    for was_var, hfcs_var in base_var_mapping.items():
        # Replace the {round} placeholder with the appropriate suffix
        mapped_var = was_var.format(round=suffix)
        var_mapping[mapped_var] = hfcs_var
    
    return var_mapping

# List of variables containing monetary values that need currency conversion
var_numerical = [
    # Income variables
    "Income",  # Total gross regular household annual income
    "Employee Income",  # Gross annual income from employment
    "Employee Income Net",  # Net annual income from employment
    "Employee Income ASHE",  # Annual gross pay from ASHE survey
    "Self-Employment Income",  # Gross annual self-employed income
    "Self-Employment Income Net",  # Net annual self-employed income
    "Self-Employment Income Total",  # Total Annual Gross self employed income
    "Rental Income from Real Estate",  # Annual rental income
    "Rental Income Gross",  # Gross rental income
    "Rental Income Net",  # Net rental income
    "Income from Financial Assets",  # Total income from investments
    "Investment Income Gross",  # Gross investment income
    "Investment Income Net",  # Net investment income
    "Income from Pensions",  # Annual Household Income - Gross occupational or private pension
    "Regular Social Transfers",  # Total benefits received
    "Working Age Benefits",  # Working Age Benefits
    "Disability Benefits",  # Disability Benefits
    "Pensioner Benefits",  # Pensioner Benefits
    
    # Asset variables
    "Value of the Main Residence",  # Current value of main residence
    "Value of the Main Residence Alt",  # Alternative main residence value
    "Value of other Properties",  # Total value of other houses
    "Value of Second Homes",  # Value of second homes
    "Value of Other Property",  # Total value of other property excluding main property
    "Total Property Value",  # Sum of all property values
    "Value of Household Vehicles",  # Total value of all vehicles
    "Value of Household Valuables",  # Value of all household goods and collectables
    "Value of Self-Employment Businesses",  # Approximate value of share of business
    "Wealth in Deposits",  # Total value of savings accounts
    "Formal Financial Assets",  # Total value of formal financial assets (covers mutual funds, bonds, shares, managed accounts)
    "Other Assets",  # Value of other investments
    "Voluntary Pension",  # Total value of individual pension wealth
    
    # Liability variables
    "Household Debt Burden",  # Burden of mortgage and other debt on household
    "Outstanding Balance of HMR Mortgages",  # Total mortgage on main residence
    "Other Property Debt",  # Total debt houses not main residence
    "Other Property Mortgage",  # Total property debt excluding main residence
    "Outstanding Balance of Credit Card Debt",  # Hhold total outstanding credit/store/charge card balance
    "Non-Mortgage Debt Burden",  # Burden from non-mortgage debt
    
    # Housing variables
    "Rent Paid",  # How much is usual household rent
]


class WASReader:
    """
    A class for reading and processing Wealth and Assets Survey (WAS) data.

    This class handles the reading and initial processing of WAS data, including:
    - Loading survey waves 1-5 and rounds 6-8
    - Joining household and individual data
    - Standardizing variable names to HFCS-compatible format
    - Handling Stata (.dta) file format

    Parameters
    ----------
    country_name_short : str
        Two-letter country code (must be "GB" for Great Britain)
    individuals_df : pd.DataFrame
        DataFrame containing individual-level survey data
    households_df : pd.DataFrame
        DataFrame containing household-level survey data

    Attributes
    ----------
    country_name_short : str
        Two-letter country code (always "GB")
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
    def from_stata(
        cls,
        country_name: str,
        country_name_short: str,
        year: int,
        was_data_path: Path,
        exchange_rates: ExchangeRatesReader,
        round_number: int = 7,
    ) -> "WASReader":
        """
        Create a WASReader instance from Stata (.dta) files.

        This method reads and processes WAS survey files from Stata format, including:
        - Person files: Personal characteristics and income
        - Household files: Household assets and liabilities

        Parameters
        ----------
        country_name : str
            Full country name (e.g., "United Kingdom")
        country_name_short : str
            Two-letter country code (e.g., "GB")
        year : int
            Survey year
        was_data_path : Path
            Base path to WAS data files
        exchange_rates : ExchangeRatesReader
            Exchange rate converter for monetary values
        round_number : int, optional
            WAS round/wave number to read (default: 7)

        Returns
        -------
        WASReader
            Initialized reader with processed survey data

        Notes
        -----
        - Files are expected to be named was_round_X_person_eul_*.dta and was_round_X_hhold_eul_*.dta
          OR was_wave_X_person_eul_*.dta and was_wave_X_hhold_eul_*.dta
        - The method searches for both "round" and "wave" naming patterns
        - All monetary values are converted to local currency
        - WAS data is already in GBP, so conversion is typically 1:1
        """
        import glob
        
        # Try both round and wave patterns
        round_person_path = was_data_path / f"was_round_{round_number}_person_eul_*.dta"
        round_household_path = was_data_path / f"was_round_{round_number}_hhold_eul_*.dta"
        wave_person_path = was_data_path / f"was_wave_{round_number}_person_eul_*.dta"
        wave_household_path = was_data_path / f"was_wave_{round_number}_hhold_eul_*.dta"
        
        # Find actual files matching the patterns
        round_person_files = glob.glob(str(round_person_path))
        round_household_files = glob.glob(str(round_household_path))
        wave_person_files = glob.glob(str(wave_person_path))
        wave_household_files = glob.glob(str(wave_household_path))
        
        # Determine which pattern to use (prefer round if both exist)
        if round_person_files and round_household_files:
            person_files = round_person_files
            household_files = round_household_files
            pattern_type = "round"
        elif wave_person_files and wave_household_files:
            person_files = wave_person_files
            household_files = wave_household_files
            pattern_type = "wave"
        else:
            # Try to find any matching files for better error messages
            all_person_files = round_person_files + wave_person_files
            all_household_files = round_household_files + wave_household_files
            
            if not all_person_files:
                raise FileNotFoundError(
                    f"No person files found matching patterns: "
                    f"{round_person_path} or {wave_person_path}"
                )
            if not all_household_files:
                raise FileNotFoundError(
                    f"No household files found matching patterns: "
                    f"{round_household_path} or {wave_household_path}"
                )
            # If we have some files but not both types, use what we have
            person_files = all_person_files
            household_files = all_household_files
            pattern_type = "mixed"
        
        # Read person data
        individuals_df = cls.read_stata(
            path=person_files[0],  # Take the first matching file
            country_name=country_name,
            country_name_short=country_name_short,
            year=year,
            exchange_rates=exchange_rates,
            round_number=round_number,
        )
        
        # Read household data
        households_df = cls.read_stata(
            path=household_files[0],  # Take the first matching file
            country_name=country_name,
            country_name_short=country_name_short,
            year=year,
            exchange_rates=exchange_rates,
            round_number=round_number,
        )

        return cls(
            country_name_short=country_name_short,
            individuals_df=individuals_df,
            households_df=households_df,
        )

    @staticmethod
    def read_stata(
        path: Path | str,
        country_name: str,
        country_name_short: str,
        year: int,
        exchange_rates: ExchangeRatesReader,
        round_number: int,
    ) -> pd.DataFrame:
        """
        Read and process a single WAS Stata (.dta) file.

        This method:
        1. Reads the Stata file
        2. Maps variable names to standardized format
        3. Converts monetary values to local currency
        4. Handles missing values and data types

        Parameters
        ----------
        path : Path | str
            Path to the Stata file
        country_name : str
            Full country name for exchange rate lookup
        country_name_short : str
            Two-letter country code for filtering
        year : int
            Year for exchange rate lookup
        exchange_rates : ExchangeRatesReader
            Exchange rate converter
        round_number : int
            WAS round/wave number for variable mapping

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with standardized columns and local currency values

        Notes
        -----
        - Missing values are converted to NaN
        - Monetary values are converted from GBP to local currency (typically 1:1)
        - Only variables in the dynamic var_mapping are kept
        """
        # Load data from Stata file
        df = pd.read_stata(path, preserve_dtypes=False, convert_categoricals=False)
        
        # Get the appropriate variable mapping for this round/wave
        var_mapping = get_var_mapping(round_number)
        
        # Keep only mapped variables that exist in the data
        available_vars = [col for col in var_mapping.keys() if col in df.columns]
        df = df[available_vars]
        df.rename(columns=var_mapping, inplace=True)
        
        # Set index to ID if available
        if "ID" in df.columns:
            df.set_index("ID", inplace=True)
        
        # Convert monetary values to local currency
        var_numerical_union = [v for v in var_numerical if v in df.columns]
        if var_numerical_union:
            # Convert to numeric, coercing errors to NaN
            for col in var_numerical_union:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Apply exchange rate conversion
            df.loc[:, var_numerical_union] = exchange_rates.from_eur_to_lcu(
                country=country_name,
                year=year,
            ) * df.loc[:, var_numerical_union]
        
        return df

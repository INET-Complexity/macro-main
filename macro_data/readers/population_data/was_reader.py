"""
This module provides functionality for reading and processing data from the ONS's Wealth and Assets Survey (WAS). 
The WAS is a detailed survey that collects household-level data on households' finances and consumption patterns 
for Great Britain (England, Wales, and Scotland).

Key Features:
- Read and process WAS survey data from multiple waves (Rounds 1-8)
- Handle both household and individual level data
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

    # Read WAS data for Great Britain in 2022
    was = WASReader.from_stata(
        country_name="United Kingdom",
        country_name_short="GB",
        year=2022,
        was_data_path=Path("path/to/was/data")
    )

    # Access household and individual data
    households = was.households_df
    individuals = was.individuals_df
    ```

Note:
    WAS data is already in GBP, so no currency conversion is needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Base mapping of WAS variable codes to standardized HFCS-compatible names
# This mapping uses placeholders for the round/wave suffix that will be dynamically replaced
# Descriptions are based on the official WAS data dictionary
base_var_mapping = {
    # Individual Characteristics
    "pidno": "Personal identifier",  # Personal identifier from WAS survey 
    "person{round}": "Person number R7",  # Individual within household
    "CASE{round}": "Household identifier",  # Household identifier (CASER7)
    "{round}xsperswgt": "R7 XS round-based person weight",  # Person weight from WAS survey 
    "sex{round}": "Sex",  # Sex from WAS survey 
    "DVAge17{round}": "Grouped age (17 categories)",  # Age from WAS survey 
    "edlevel{round}": "DV - Education level",  # Education level from WAS survey 
    "wrking{round}": "Whether working in reference week",  # Working status from WAS survey 
    "siccode{round}": "SICCODE",  # Industry SIC codes from WAS survey 
    "hholdtype{round}": "DV - Type of household",  # Household type from WAS survey 
    
    # Income Sources
    "dvgrspayannual{round}": "Gross annual income employee main job (including bonuses and commission received)",  # Gross annual income from WAS survey 
    "dvnetpayannual{round}": "Net annual income employee main job (including bonuses and commission received)",  # Net annual income from WAS survey 
    "dvsegrspay{round}": "Gross annual self-employed income main job",  # Gross annual self-employed income
    "dvsenetpay{round}": "Net annual self-employed income main job",  # Net annual self-employed income
    "DVGISE{round}": "Total Annual Gross self employed income (main and second job)",  # Total Annual Gross self employed income
    "dvrentincamannual{round}": "DV - Annual amount (£) received from rental income",  # Annual rental income
    "DVGrsRentAmtAnnual{round}_aggr": "Annual Household Income - Gross rental income",  # Gross rental income
    "DVNetRentAmtAnnual{round}_aggr": "Annual Household Income - Net rental income",  # Net rental income
    "fincv{round}": "Total income in £ over last 12 months received in dividends, interest or return on investments",  # Total income from investments
    "DVGIINV{round}_aggr": "Annual Household Income - Gross investment income",  # Gross investment income
    "DVNIINV{round}_aggr": "Annual Household Income - Net investment income",  # Net investment income
    "DVGIPPen{round}_aggr": "Annual Household Income - Gross occupational or private pension",  # Annual Household Income - Gross occupational or private pension
    "DVBenefitAnnual{round}_aggr": "Annual Household Income - Total benefits received",  # Total benefits received
    "wageben1{round}": "Working Age Benefits 1",  # Working Age Benefits
    "disben1{round}": "Disability Benefits 1",  # Disability Benefits
    "penben1{round}": "Pensioner Benefits 1",  # Pensioner Benefits
    "Dvtotgir{round}": "Total gross regular household annual income",  # Total gross regular household annual income
    "unidk{round}": "Whether don't know because it's paid in combination with another benefit",  # Unemployment benefit indicators
    "unifdk{round}": "Whether don't know because it's paid in combination with another benefit",  # Unemployment benefit indicators
    
    # Household Assets
    "hvalue{round}": "Current value of main residence",  # Current value of main residence
    "DVHValue{round}": "Value of main residence",  # Value of main residence
    "DVOPrVal{round}": "Total value of other houses",  # Total value of other houses
    "uvals{round}": "Value of second homes",  # Value of second homes
    "OthPropVal{round}": "Total value of other property excluding main property",  # Total value of other property excluding main property
    "DVProperty{round}": "Sum of all property values",  # Sum of all property values
    "Dvtotvehval{round}": "Total value of all vehicles",  # Total value of all vehicles
    "Allgd{round}": "Value of all household goods and collectables",  # Value of all household goods and collectables
    "bworthb{round}": "Approximate value of share of business after deducting outstanding debts",  # Approximate value of share of business
    "DVSaVal{round}_SUM": "Total value of savings accounts",  # Total value of savings accounts
    "DVFFAssets{round}_SUM": "Total value of all formal financial assets",  # Total value of formal financial assets (covers mutual funds, bonds, shares, managed accounts)
    "DVFInvOtV{round}": "Value of other investments (formal financial assets)",  # Value of other investments
    "TOTPEN{round}": "Total value of individual pension wealth",  # Total value of individual pension wealth
    
    # Household Liabilities
    "dburdh{round}": "Burden of mortgage and other debt on household",  # Burden of mortgage and other debt on household
    "Totmort{round}": "Total mortgage on main residence",  # Total mortgage on main residence
    "DVHseDebt{round}_sum": "Total debt houses not main residence",  # Total debt houses not main residence
    "OthMort{round}_sum": "Total property debt excluding main residence",  # Total property debt excluding main residence
    "TOTCSC{round}_aggr": "Hhold total outstanding credit/store/charge card balance",  # Hhold total outstanding credit/store/charge card balance
    "dburd{round}": "Burden from non-mortgage debt",  # Burden from non-mortgage debt
    
    # Housing Characteristics
    "hhldr{round}": "Whether owns or rents accomodation",  # Whether owns or rents accommodation
    "ten1{round}": "Tenure",  # Tenure
    "llord{round}": "Landlord",  # Landlord – category of landlord
    "dvrentpaid{round}": "How much is usual household rent",  # How much is usual household rent
    "unumhs{round}": "Number of second homes",  # Number of second homes
    "unumhs{round}_i": "Number of Second Homes Imputed",  # Imputed Number of second homes
    "unumhs{round}_iflag": "Second Homes Imputation Flag",  # Imputation flag
    "ubuylet{round}": "number of buy to let properties",  # number of buy to let properties
    "ubuylet{round}_i": "Buy to Let Properties Imputed",  # Imputed number of buy to let properties
    "ubuylet{round}_iflag": "Buy to Let Imputation Flag",  # Imputation flag
    "unumbd{round}": "Number of buildings",  # Number of buildings
    "unumbd{round}_i": "Buildings Imputed",  # imputed number of buildings
    "unumbd{round}_iflag": "Buildings Imputation Flag",  # imputation flag
    "unumla{round}": "number of pieces of land",  # number of pieces of land
    "unumla{round}_i": "Land Pieces Imputed",  # imputed number of pieces of land
    "unumla{round}_iflag": "Land Imputation Flag",  # imputation flag
    "unumov{round}": "number of properties overseas",  # number of properties overseas
    "unumov{round}_i": "Overseas Properties Imputed",  # imputed number of properties overseas
    "unumov{round}_iflag": "Overseas Imputation Flag",  # imputation flag
    "unumre{round}": "number of other properties",  # number of other properties
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
    "Total gross regular household annual income",  # Total gross regular household annual income
    "Gross annual income employee main job (including bonuses and commission received)",  # Gross annual income from employment
    "Net annual income employee main job (including bonuses and commission received)",  # Net annual income from employment
    "Employee Income ASHE",  # Annual gross pay from ASHE survey (not in data dictionary)
    "Gross annual self-employed income main job",  # Gross annual self-employed income
    "Net annual self-employed income main job",  # Net annual self-employed income
    "Total Annual Gross self employed income (main and second job)",  # Total Annual Gross self employed income
    "DV - Annual amount (£) received from rental income",  # Annual rental income
    "Annual Household Income - Gross rental income",  # Gross rental income
    "Annual Household Income - Net rental income",  # Net rental income
    "Total income in £ over last 12 months received in dividends, interest or return on investments",  # Total income from investments
    "Annual Household Income - Gross investment income",  # Gross investment income
    "Annual Household Income - Net investment income",  # Net investment income
    "Annual Household Income - Gross occupational or private pension",  # Annual Household Income - Gross occupational or private pension
    "Annual Household Income - Total benefits received",  # Total benefits received
    "Working Age Benefits 1",  # Working Age Benefits
    "Disability Benefits 1",  # Disability Benefits
    "Pensioner Benefits 1",  # Pensioner Benefits
    
    # Asset variables
    "Current value of main residence",  # Current value of main residence
    "Value of main residence",  # Alternative main residence value
    "Total value of other houses",  # Total value of other houses
    "Value of second homes",  # Value of second homes
    "Total value of other property excluding main property",  # Total value of other property excluding main property
    "Sum of all property values",  # Sum of all property values
    "Total value of all vehicles",  # Total value of all vehicles
    "Value of all household goods and collectables",  # Value of all household goods and collectables
    "Approximate value of share of business after deducting outstanding debts",  # Approximate value of share of business
    "Total value of savings accounts",  # Total value of savings accounts
    "Total value of all formal financial assets",  # Total value of formal financial assets (covers mutual funds, bonds, shares, managed accounts)
    "Value of other investments (formal financial assets)",  # Value of other investments
    "Total value of individual pension wealth",  # Total value of individual pension wealth
    
    # Liability variables
    "Burden of mortgage and other debt on household",  # Burden of mortgage and other debt on household
    "Total mortgage on main residence",  # Total mortgage on main residence
    "Total debt houses not main residence",  # Total debt houses not main residence
    "Total property debt excluding main residence",  # Total property debt excluding main residence
    "Hhold total outstanding credit/store/charge card balance",  # Hhold total outstanding credit/store/charge card balance
    "Burden from non-mortgage debt",  # Burden from non-mortgage debt
    
    # Housing variables
    "How much is usual household rent",  # How much is usual household rent
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
            round_number=round_number,
        )
        
        # Read household data
        households_df = cls.read_stata(
            path=household_files[0],  # Take the first matching file
            country_name=country_name,
            country_name_short=country_name_short,
            year=year,
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
        round_number: int,
    ) -> pd.DataFrame:
        """
        Read and process a single WAS Stata (.dta) file.

        This method:
        1. Reads the Stata file
        2. Maps variable names to standardized format
        3. Handles missing values and data types

        Parameters
        ----------
        path : Path | str
            Path to the Stata file
        country_name : str
            Full country name
        country_name_short : str
            Two-letter country code for filtering
        year : int
            Survey year
        round_number : int
            WAS round/wave number for variable mapping

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with standardized columns

        Notes
        -----
        - Missing values are converted to NaN
        - Only variables in the dynamic var_mapping are kept
        """
        # Load data from Stata file
        df = pd.read_stata(path, preserve_dtypes=False, convert_categoricals=False)
        
        # Get the appropriate variable mapping for this round/wave
        var_mapping = get_var_mapping(round_number)
        
        # Keep only mapped variables that exist in the data
        available_vars = [col for col in var_mapping.keys() if col in df.columns]
        df = df[available_vars].copy()
        df.rename(columns=var_mapping, inplace=True)
        
        # Set index to Personal identifier if available
        if "Personal identifier" in df.columns:
            df.set_index("Personal identifier", inplace=True)
        
        # Convert monetary values to local currency
        var_numerical_union = [v for v in var_numerical if v in df.columns]
        if var_numerical_union:
            # Convert to numeric, coercing errors to NaN
            for col in var_numerical_union:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                       
        return df

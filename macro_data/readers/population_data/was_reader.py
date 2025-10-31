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

    # Read most recent WAS data for Great Britain in 2022
    # Automatically selects round 8 (2020-2022) since it's the most recent <= 2022
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
        round_number: int | None = None,
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
        round_number : int | None, optional
            WAS round/wave number to read. If None, automatically loads the most recent dataset 
            with end year <= the specified year parameter (default: None)

        Returns
        -------
        WASReader
            Initialized reader with processed survey data

        Notes
        -----
        - Files are expected to be named was_round_X_person_eul_*.dta and was_round_X_hhold_eul_*.dta
          OR was_wave_X_person_eul_*.dta and was_wave_X_hhold_eul_*.dta
        - The method searches for both "round" and "wave" naming patterns
        - If round_number is None, automatically detects and loads the most recent dataset 
          available before or equal to the specified year
        - WAS data is already in GBP, so conversion is typically 1:1
        """
        import glob
        
        # Resolve the path to ensure it's absolute and correct
        was_data_path = Path(was_data_path).resolve()
        
        # If round_number not specified, find the most recent dataset before or equal to year
        if round_number is None:
            round_number = cls._find_most_recent_round(was_data_path, year)
        
        # Try both round and wave patterns
        round_person_path = str(was_data_path / f"was_round_{round_number}_person_eul_*.dta")
        round_household_path = str(was_data_path / f"was_round_{round_number}_hhold_eul_*.dta")
        wave_person_path = str(was_data_path / f"was_wave_{round_number}_person_eul_*.dta")
        wave_household_path = str(was_data_path / f"was_wave_{round_number}_hhold_eul_*.dta")
        
        # Find actual files matching the patterns
        round_person_files = glob.glob(round_person_path)
        round_household_files = glob.glob(round_household_path)
        wave_person_files = glob.glob(wave_person_path)
        wave_household_files = glob.glob(wave_household_path)
        
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
    def _get_round_end_year(round_num: int) -> int:
        """
        Get the end year for a given WAS round/wave number.
        
        Parameters
        ----------
        round_num : int
            Round/wave number (1-8)
            
        Returns
        -------
        int
            End year of the survey period
            
        Notes
        -----
        - Waves 1-5: 2006-2008, 2008-2010, 2010-2012, 2012-2014, 2014-2016
        - Rounds 6-8: 2016-2018, 2018-2020, 2020-2022
        """
        round_to_year = {
            1: 2008,  # Wave 1: 2006-2008
            2: 2010,  # Wave 2: 2008-2010
            3: 2012,  # Wave 3: 2010-2012
            4: 2014,  # Wave 4: 2012-2014
            5: 2016,  # Wave 5: 2014-2016
            6: 2018,  # Round 6: 2016-2018
            7: 2020,  # Round 7: 2018-2020
            8: 2022,  # Round 8: 2020-2022
        }
        return round_to_year.get(round_num, 0)
    
    @staticmethod
    def _find_most_recent_round(was_data_path: Path, year: int) -> int:
        """
        Find the most recent WAS round/wave number before or equal to the specified year.
        
        Parameters
        ----------
        was_data_path : Path
            Base path to WAS data files
        year : int
            Target year - returns most recent round/wave with end year <= this year
            
        Returns
        -------
        int
            The most recent round/wave number found that is <= the specified year
            
        Notes
        -----
        - Scans for both 'round' (6-8) and 'wave' (1-5) patterns
        - Returns the highest round number found where end year <= specified year
        - Waves 1-5 map directly to rounds 1-5
        """
        import glob
        import re
        
        # Resolve the path to ensure it's absolute and correct
        was_data_path = Path(was_data_path).resolve()
        
        available_rounds = set()
        
        # Search for all round files (rounds 6-8)
        round_pattern = str(was_data_path / "was_round_*_person_eul_*.dta")
        round_files = glob.glob(round_pattern)
        
        for file_path in round_files:
            match = re.search(r"was_round_(\d+)_person_eul_", file_path)
            if match:
                round_num = int(match.group(1))
                available_rounds.add(round_num)
        
        # Search for all wave files (waves 1-5, which correspond to rounds 1-5)
        wave_pattern = str(was_data_path / "was_wave_*_person_eul_*.dta")
        wave_files = glob.glob(wave_pattern)
        
        for file_path in wave_files:
            match = re.search(r"was_wave_(\d+)_person_eul_", file_path)
            if match:
                wave_num = int(match.group(1))
                available_rounds.add(wave_num)  # Waves map directly to rounds
        
        if not available_rounds:
            raise FileNotFoundError(
                f"No WAS data files found in {was_data_path}. "
                f"Expected files matching patterns: was_round_*_person_eul_*.dta or was_wave_*_person_eul_*.dta"
            )
        
        # Filter rounds to those with end year <= specified year, then get the maximum
        valid_rounds = [
            r for r in available_rounds 
            if WASReader._get_round_end_year(r) <= year
        ]
        
        if not valid_rounds:
            available_years = sorted([WASReader._get_round_end_year(r) for r in available_rounds])
            raise ValueError(
                f"No WAS dataset found for year {year} or earlier. "
                f"Available datasets end in years: {available_years}"
            )
        
        return max(valid_rounds)

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
        
        # Create case-insensitive column lookup
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        # Match variables case-insensitively and build actual column mapping
        available_vars = []
        actual_column_mapping = {}
        for expected_var, mapped_name in var_mapping.items():
            # Try exact match first
            if expected_var in df.columns:
                available_vars.append(expected_var)
                actual_column_mapping[expected_var] = mapped_name
            # Try case-insensitive match
            elif expected_var.lower() in df_columns_lower:
                actual_col = df_columns_lower[expected_var.lower()]
                available_vars.append(actual_col)
                actual_column_mapping[actual_col] = mapped_name
        
        # Keep only matched variables
        df = df[available_vars].copy()
        # Rename using the actual column names found
        df.rename(columns=actual_column_mapping, inplace=True)
        
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

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa
from sklearn.linear_model import LinearRegression

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_population.was_household_tools import (
    set_household_housing_data,
    set_household_types,
)
from macro_data.processing.synthetic_population.was_individual_tools import (
    process_individual_data,
)
from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
    default_target_investment,
)
from macro_data.processing.synthetic_population.utils import (
    ensure_minimum_workers_in_industries,
)
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.exogenous_data import ExogenousCountryData
from macro_data.readers.io_tables.industries import ALL_INDUSTRIES
from macro_data.util.clean_data import remove_outliers
from macro_data.util.imputation import apply_iterative_imputer
from macro_data.util.regressions import fit_linear

# WAS-specific column restrictions - using standard column names (matching HFCS and households.py expectations)
RESTRICT_COLS = [
    "Type",
    "Corresponding Individuals ID",
    "Corresponding Bank ID",
    "Corresponding Inhabited House ID",
    "Corresponding Renters",
    "Corresponding Property Owner",
    "Corresponding Additionally Owned Houses ID",
    "Income",
    "Investment",
    "Employee Income",
    "Regular Social Transfers",
    "Rental Income from Real Estate",
    "Income from Financial Assets",
    "Saving Rate",
    "Rent Paid",
    "Rent Imputed",
    "Wealth",
    "Net Wealth",
    "Wealth in Real Assets",
    "Value of the Main Residence",
    "Value of other Properties",
    "Wealth Other Real Assets",
    "Wealth in Deposits",
    "Wealth in Other Financial Assets",
    "Wealth in Financial Assets",
    "Outstanding Balance of HMR Mortgages",
    "Outstanding Balance of Mortgages on other Properties",
    "Outstanding Balance of other Non-Mortgage Loans",
    "Debt",
    "Debt Installments",
    "Tenure Status of the Main Residence",
    "Number of Properties other than Household Main Residence",
]

# WAS-specific monetary columns for processing - using original WAS data variable names
CONVERT_HH_COLS = [
    "How much is usual household rent",
    "Amount spent on Consumption of Goods and Services",
    "Annual Household Income - Gross rental income",
    "Total income in £ over last 12 months received in dividends, interest or return on investments",
    "Annual Household Income - Gross occupational or private pension",
    "Annual Household Income - Total benefits received",
    "Income",
    "Value of main residence",
    "Total value of other houses",
    "Total value of all vehicles",
    "Value of all household goods and collectables",
    "Approximate value of share of business after deducting outstanding debts",
    "Total value of savings accounts",
    "Total value of all formal financial assets",
    "Other Assets",
    "Total value of individual pension wealth",
    "Total mortgage on main residence",
    "Total property debt excluding main residence",
    "Hhold total outstanding credit/store/charge card balance",
    "Outstanding Balance of other Non-Mortgage Loans",
    # WAS-specific monetary variables using original names
    "Gross annual income employee main job (including bonuses and commission received)",
    "Net annual income employee main job (including bonuses and commission received)",
    "Employee Income ASHE",
    "Gross annual self-employed income main job",
    "Net annual self-employed income main job",
    "Total Annual Gross self employed income (main and second job)",
    "Annual Household Income - Gross rental income",
    "Annual Household Income - Net rental income",
    "Annual Household Income - Gross investment income",
    "Annual Household Income - Net investment income",
    "Working Age Benefits 1",
    "Disability Benefits 1",
    "Pensioner Benefits 1",
    "Sum of all property values",
    "Total value of other property excluding main property",
    "Value of second homes",
    "Total debt houses not main residence",
    "Total property debt excluding main residence",
]

CONVERT_IND_COLS = [
    "Gross annual income employee main job (including bonuses and commission received)", 
    "Net annual income employee main job (including bonuses and commission received)",
    "Employee Income ASHE",
    "Gross annual self-employed income main job",
    "Net annual self-employed income main job",
    "Total Annual Gross self employed income (main and second job)",
    "Income from Unemployment Benefits", 
    "Income"
]


class SyntheticWASPopulation(SyntheticPopulation):
    """
    A class representing a synthetic population generated from WAS (Wealth and Assets Survey) data.

    This class extends the base SyntheticPopulation class to handle WAS-specific data processing,
    including UK-specific household characteristics, wealth patterns, and housing data.
    
    Note: WAS data is UK-specific. While the interface maintains country_name parameters for
    compatibility with the base class, all instances will use UK data regardless of input.

    Attributes:
        country_name (str): Always "United Kingdom" (hardcoded for WAS data).
        country_name_short (str): Always "GB" (hardcoded for WAS data).
        scale (int): The scale of the population.
        year (int): The year of the population data.
        industries (list[str]): The list of industries.
        individual_data (pd.DataFrame): The DataFrame containing individual data.
        household_data (pd.DataFrame): The DataFrame containing household data.
        social_housing_rent (float): The rent for social housing.
        coefficient_fa_income (float): The coefficient for FA income.
        consumption_weights (np.ndarray): The array of consumption weights.
        consumption_weights_by_income (np.ndarray): The array of consumption weights by income.

    Methods:
        from_readers(cls, readers, country_name, country_name_short, scale, year, industry_data, industries, total_unemployment_benefits, rent_as_fraction_of_unemployment_rate, n_quantiles=5):
            Creates a SyntheticWASPopulation instance from data readers. Note: country_name parameters
            are maintained for interface compatibility but only UK data will be used.

        restrict(self):
            Restricts the household data to selected columns.

        compute_household_wealth(self):
            Computes the wealth-related fields in the household data.

        compute_household_income(self, total_social_transfers):
            Computes the income-related fields in the household data.

        set_household_housing_data(self, rent_as_fraction_of_unemployment_rate, unemployment_benefits_by_capita):
            Sets the housing-related fields in the household data.
    """

    def __init__(
        self,
        country_name: str,
        country_name_short: str,
        scale: int,
        year: int,
        industries: list[str],
        individual_data: pd.DataFrame,
        household_data: pd.DataFrame,
        social_housing_rent: float,
        coefficient_fa_income: float,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        investment: np.ndarray,
    ):
        # WAS data is UK-specific, so we hardcode UK values regardless of input
        # This maintains interface compatibility while ensuring UK-only usage
        uk_country_name = "GBR"
        uk_country_name_short = "GB"
        
        saving_rates_model = LinearRegression()
        social_transfers_model = LinearRegression()
        wealth_distribution_model = LinearRegression()
        super().__init__(
            uk_country_name,
            uk_country_name_short,
            scale,
            year,
            industries,
            individual_data,
            household_data,
            social_housing_rent,
            coefficient_fa_income,
            consumption_weights,
            consumption_weights_by_income,
            investment,
            saving_rates_model,
            social_transfers_model,
            wealth_distribution_model,
        )

    @classmethod
    def from_readers(
        cls,
        readers: DataReaders,
        country_name: Country,
        country_name_short: str,
        scale: int,
        year: int,
        quarter: int,
        industry_data: dict[str, pd.DataFrame],
        industries: list[str],
        total_unemployment_benefits: float,
        exogenous_data: ExogenousCountryData,
        rent_as_fraction_of_unemployment_rate: float = 0.25,
        n_quantiles: int = 5,
        population_ratio: float = 1.0,
        proxied_country: str | Country = None,
        yearly_factor: float = 4.0,
    ) -> "SyntheticWASPopulation":
        """
        Creates a synthetic population from WAS data readers.
        
        Note: WAS data is UK-specific. The country_name parameter is maintained for interface
        compatibility but only UK/GBR data will be used regardless of the input.

        Args:
            cls: The class object.
            readers (DataReaders): The data readers object.
            country_name (Country): The country object (should be UK/GBR for WAS data).
            country_name_short (str): The short name of the country (ignored, UK used internally).
            scale (int): The scaling factor.
            year (int): The year.
            quarter (int): The quarter.
            industry_data (dict[str, pd.DataFrame]): The industry data.
            industries (list[str]): The list of industries.
            total_unemployment_benefits (float): The total unemployment benefits.
            exogenous_data (ExogenousCountryData): The exogenous data.
            rent_as_fraction_of_unemployment_rate (float): The rent as fraction of unemployment rate.
            n_quantiles (int): The number of quantiles.
            population_ratio (float): The population ratio.
            proxied_country (str | Country): The proxied country.
            yearly_factor (float): The yearly factor.

        Returns:
            SyntheticWASPopulation: The synthetic population.
        """
        # WAS data is UK-specific - hardcode UK country for data retrieval
        uk_country = Country.UNITED_KINGDOM  # Use UNITED_KINGDOM enum value for UK
        
        # Get WAS data for UK (regardless of input country_name)
        # Try looking up by Country object first, then by string "GBR" if needed
        was_reader = readers.was.get(uk_country) or readers.was.get("GBR")
        if was_reader is None:
            raise ValueError(f"No WAS data available for UK (WAS is UK-specific)")

        # Get household and individual data from WAS
        households_df = was_reader.households_df.copy()
        individuals_df = was_reader.individuals_df.copy()

        # Get unemployment and participation rates from labour stats
        unemployment_rate = exogenous_data.labour_stats.loc[f"{year}-Q{quarter}", "Unemployment Rate (Value)"].iloc[0]
        participation_rate = exogenous_data.labour_stats.loc[f"{year}-Q{quarter}", "Participation Rate (Value)"].iloc[0]

        # Convert grouped age to continuous age if needed
        if "Grouped age (17 categories)" in individuals_df.columns and "Age" not in individuals_df.columns:
            # Map WAS 17-category age groups to continuous ages
            # Categories typically represent 5-year age bands: 0-4, 5-9, 10-14, ..., 75-79, 80+
            def convert_grouped_age_to_continuous(grouped_age):
                """Convert WAS grouped age categories to continuous ages."""
                if pd.isna(grouped_age):
                    return np.nan
                
                # WAS uses 17 age categories (typically 5-year bands up to 80+)
                # Map category number to midpoint of age range
                age_midpoints = {
                    1: 2.5,    # 0-4
                    2: 7.5,    # 5-9
                    3: 12.5,   # 10-14
                    4: 17.5,   # 15-19
                    5: 22.5,   # 20-24
                    6: 27.5,   # 25-29
                    7: 32.5,   # 30-34
                    8: 37.5,   # 35-39
                    9: 42.5,   # 40-44
                    10: 47.5,  # 45-49
                    11: 52.5,  # 50-54
                    12: 57.5,  # 55-59
                    13: 62.5,  # 60-64
                    14: 67.5,  # 65-69
                    15: 72.5,  # 70-74
                    16: 77.5,  # 75-79
                    17: 85.0,  # 80+ (using 85 as midpoint estimate)
                }
                
                # If it's already a numeric value, use it directly
                if isinstance(grouped_age, (int, float)):
                    category = int(grouped_age)
                    if category in age_midpoints:
                        # Add some randomness within the 5-year band for realism
                        base_age = age_midpoints[category]
                        if category == 17:  # 80+ category, use wider range
                            return base_age + np.random.uniform(-5, 10)
                        else:
                            return base_age + np.random.uniform(-2.5, 2.5)
                
                return np.nan
            
            individuals_df["Age"] = individuals_df["Grouped age (17 categories)"].apply(convert_grouped_age_to_continuous)
        
        # Ensure Age column exists (rename if present with different name, or create if missing)
        if "Age" not in individuals_df.columns:
            # Check for other possible age column names
            age_cols = [col for col in individuals_df.columns if "age" in col.lower() and "grouped" not in col.lower()]
            if age_cols:
                individuals_df["Age"] = individuals_df[age_cols[0]]
            else:
                # If no age column found, create one with NaN values (will be filled later)
                individuals_df["Age"] = np.nan

        # Map WAS column names to expected names for individual processing
        column_mapping = {
            "Sex": "Gender",
            "DV - Education level": "Education",
            "Whether working in reference week": "Labour Status",
            "SICCODE": "Employment Industry",
            "Gross annual income employee main job (including bonuses and commission received)": "Employee Income",
            "Total Annual Gross self employed income (main and second job)": "Self-Employment Income Total",
            "Household identifier": "HID",
        }
        
        # Apply column mappings (only if source column exists and target doesn't)
        for was_col, expected_col in column_mapping.items():
            if was_col in individuals_df.columns and expected_col not in individuals_df.columns:
                individuals_df[expected_col] = individuals_df[was_col]
        
        # Ensure HID is set from Household identifier if needed
        if "HID" not in individuals_df.columns:
            if "Household identifier" in individuals_df.columns:
                individuals_df["HID"] = individuals_df["Household identifier"]
            else:
                # Create sequential HIDs if no household identifier available
                individuals_df["HID"] = range(len(individuals_df))
        
        # Ensure all required columns exist (initialize with defaults if missing)
        required_columns = {
            "Gender": 1,  # Default to male (1=male, 2=female)
            "Age": np.nan,  # Already handled above
            "Education": np.nan,
            "Labour Status": 2,  # Default to unemployed (1=employed, 2=unemployed, 3=inactive, 4=student)
            "Employee Income": 0.0,
            "Self-Employment Income Total": 0.0,  # Self-employment income
            "Employment Industry": np.nan,
            "Activity Status": np.nan,  # Will be set during processing
            "Income from Unemployment Benefits": 0.0,  # Will be set during processing
        }
        
        for col, default_val in required_columns.items():
            if col not in individuals_df.columns:
                individuals_df[col] = default_val
        
        # Ensure numeric columns are numeric type
        numeric_cols = ["Age", "Education", "Labour Status", "Employee Income", "Self-Employment Income Total", "Activity Status"]
        for col in numeric_cols:
            if col in individuals_df.columns:
                individuals_df[col] = pd.to_numeric(individuals_df[col], errors="coerce")
        
        # Map working status to labour status codes if needed
        # WAS "Whether working in reference week" typically: 1 = working, 2 = not working
        # Expected Labour Status: 1 = employed, 2 = unemployed, 3 = inactive, 4 = student
        if "Labour Status" in individuals_df.columns:
            # Convert WAS working status (1=working, 2=not working) to Labour Status codes
            # Only update if values are in WAS format (1 or 2)
            unique_vals = individuals_df["Labour Status"].dropna().unique()
            if set(unique_vals).issubset({1, 2}) and not set(unique_vals).issubset({1, 2, 3, 4}):
                # Map WAS format: 1 -> 1 (employed), 2 -> 2 (unemployed)
                individuals_df["Labour Status"] = individuals_df["Labour Status"].replace({1: 1, 2: 2})

        # Process individual data with WAS-specific tools
        individuals_df = process_individual_data(
            individual_data=individuals_df,
            industries=industries,
            scale=scale,
            total_unemployment_benefits=total_unemployment_benefits,
            unemployment_rate=unemployment_rate,
            participation_rate=participation_rate,
            n_firms_by_industry=industry_data["industry_vectors"]["Number of Firms"].values,
        )

        # Ensure minimum workers in industries
        individuals_df = ensure_minimum_workers_in_industries(individuals_df, len(industries))

        # Initialize required household columns before processing
        if "Type" not in households_df.columns:
            households_df["Type"] = np.nan
        
        # Ensure household has HID column for linking
        if "Household identifier" in households_df.columns:
            if "HID" not in households_df.columns:
                households_df["HID"] = households_df["Household identifier"]
        
        # Create mapping from HID to list of individual indices
        if "Corresponding Individuals ID" not in households_df.columns:
            # Group individuals by household
            individuals_df_reset = individuals_df.reset_index()  # Reset index to get positional indices
            if "HID" in individuals_df_reset.columns:
                household_to_individuals = individuals_df_reset.groupby("HID").apply(
                    lambda x: x.index.tolist()
                ).to_dict()
                households_df["Corresponding Individuals ID"] = households_df["HID"].map(
                    household_to_individuals
                ).fillna("").apply(lambda x: x if isinstance(x, list) else [])
            else:
                # If no HID mapping, create sequential mapping
                households_df["Corresponding Individuals ID"] = [
                    [i] for i in range(len(individuals_df))
                ][:len(households_df)]

        # Set household types using WAS-specific tools
        households_df = set_household_types(households_df, individuals_df)

        # Map WAS household column names to expected names and initialize missing columns
        household_column_mapping = {
            "Tenure Status of the Main Residence": "Tenure",
            "Tenure": "Tenure",  # Keep if already named correctly
            "How much is usual household rent": "Rent Paid",
            "Value of main residence": "Value of the Main Residence",
            "Total value of other houses": "Value of other Properties",
            "Annual Household Income - Gross rental income": "Rental Income from Real Estate",
            "Annual Household Income - Net rental income": "Rental Income from Real Estate",  # Fallback
        }
        
        # Apply household column mappings
        for was_col, expected_col in household_column_mapping.items():
            if was_col in households_df.columns and expected_col not in households_df.columns:
                households_df[expected_col] = households_df[was_col]
        
        # Initialize required household columns with defaults if missing
        required_household_columns = {
            "Tenure": 1,  # Default to owning (1=own, 2=part own, 3=rent, 4=free use)
            "Rent Paid": 0.0,
            "Value of the Main Residence": 0.0,
            "Number of Properties other than Household Main Residence": 0,
            "Value of other Properties": 0.0,
            "Rental Income from Real Estate": 0.0,
        }
        
        for col, default_val in required_household_columns.items():
            if col not in households_df.columns:
                households_df[col] = default_val
        
        # Ensure standard column names exist (required by RESTRICT_COLS and households.py)
        # These columns are created by mapping or computation, but RESTRICT_COLS expects standard names
        
        # Ensure "Tenure Status of the Main Residence" exists
        if "Tenure Status of the Main Residence" not in households_df.columns:
            if "Tenure" in households_df.columns:
                households_df["Tenure Status of the Main Residence"] = households_df["Tenure"]
            else:
                households_df["Tenure Status of the Main Residence"] = 1
        
        # Ensure "Rent Paid" exists (standard name, mapped from "How much is usual household rent")
        if "Rent Paid" not in households_df.columns:
            if "How much is usual household rent" in households_df.columns:
                households_df["Rent Paid"] = households_df["How much is usual household rent"]
        
        # Ensure "Value of the Main Residence" exists (standard name, mapped from "Value of main residence")
        if "Value of the Main Residence" not in households_df.columns:
            if "Value of main residence" in households_df.columns:
                households_df["Value of the Main Residence"] = households_df["Value of main residence"]
        
        # Ensure "Value of other Properties" exists (standard name, mapped from "Total value of other houses")
        if "Value of other Properties" not in households_df.columns:
            if "Total value of other houses" in households_df.columns:
                households_df["Value of other Properties"] = households_df["Total value of other houses"]
        
        # Ensure "Rental Income from Real Estate" exists (standard name, mapped from WAS column)
        if "Rental Income from Real Estate" not in households_df.columns:
            if "Annual Household Income - Gross rental income" in households_df.columns:
                households_df["Rental Income from Real Estate"] = households_df["Annual Household Income - Gross rental income"]
            elif "Annual Household Income - Net rental income" in households_df.columns:
                households_df["Rental Income from Real Estate"] = households_df["Annual Household Income - Net rental income"]
        
        # Ensure "Employee Income" exists (computed from individual data, will be set later but ensure column exists)
        if "Employee Income" not in households_df.columns:
            households_df["Employee Income"] = 0.0
        
        # Ensure "Regular Social Transfers" exists (computed later, but ensure column exists)
        if "Regular Social Transfers" not in households_df.columns:
            if "Annual Household Income - Total benefits received" in households_df.columns:
                households_df["Regular Social Transfers"] = households_df["Annual Household Income - Total benefits received"]
            else:
                households_df["Regular Social Transfers"] = 0.0
        
        # Ensure "Income from Financial Assets" exists (computed later, but ensure column exists)
        if "Income from Financial Assets" not in households_df.columns:
            if "Total income in £ over last 12 months received in dividends, interest or return on investments" in households_df.columns:
                households_df["Income from Financial Assets"] = households_df["Total income in £ over last 12 months received in dividends, interest or return on investments"]
            else:
                households_df["Income from Financial Assets"] = 0.0
        
        # Ensure "Wealth in Deposits" exists (computed later in compute_household_wealth, but ensure column exists)
        if "Wealth in Deposits" not in households_df.columns:
            if "Total value of savings accounts" in households_df.columns:
                households_df["Wealth in Deposits"] = households_df["Total value of savings accounts"]
            else:
                households_df["Wealth in Deposits"] = 0.0
        
        # Ensure "Outstanding Balance of HMR Mortgages" exists (computed later, but ensure column exists)
        if "Outstanding Balance of HMR Mortgages" not in households_df.columns:
            if "Total mortgage on main residence" in households_df.columns:
                households_df["Outstanding Balance of HMR Mortgages"] = households_df["Total mortgage on main residence"]
            else:
                households_df["Outstanding Balance of HMR Mortgages"] = 0.0
        
        # Ensure "Outstanding Balance of Mortgages on other Properties" exists (computed later, but ensure column exists)
        if "Outstanding Balance of Mortgages on other Properties" not in households_df.columns:
            if "Total property debt excluding main residence" in households_df.columns:
                households_df["Outstanding Balance of Mortgages on other Properties"] = households_df["Total property debt excluding main residence"]
            else:
                households_df["Outstanding Balance of Mortgages on other Properties"] = 0.0
        
        # Ensure numeric columns are numeric
        household_numeric_cols = [
            "Tenure", "Tenure Status of the Main Residence", "Rent Paid", "Value of the Main Residence", 
            "Number of Properties other than Household Main Residence",
            "Value of other Properties", "Rental Income from Real Estate",
            "Employee Income", "Regular Social Transfers", "Income from Financial Assets",
            "Wealth in Deposits", "Outstanding Balance of HMR Mortgages",
            "Outstanding Balance of Mortgages on other Properties"
        ]
        for col in household_numeric_cols:
            if col in households_df.columns:
                households_df[col] = pd.to_numeric(households_df[col], errors="coerce")
                if col == "Number of Properties other than Household Main Residence":
                    households_df[col] = households_df[col].fillna(0).astype(int)

        # Set housing data using WAS-specific tools
        households_df = set_household_housing_data(
            household_data=households_df,
            scale=scale,
            rent_as_fraction_of_unemployment_rate=rent_as_fraction_of_unemployment_rate,
            unemployment_benefits_by_capita=total_unemployment_benefits / scale,
        )

        # Sample households and individuals
        households_df, individuals_df = sample_households(
            hfcs_households_data=households_df,
            hfcs_individuals_data=individuals_df,
            n_households=scale,
        )

        # Set initial values
        set_initial_values(households_df, scale)

        # Get consumption weights from ICIO data
        consumption_weights = industry_data["industry_vectors"]["Household Consumption in LCU"].values
        consumption_weights = consumption_weights / consumption_weights.sum()

        # Get investment weights
        investment_weights = industry_data["industry_vectors"]["Household Capital Inputs in LCU"].values
        investment_weights = investment_weights / investment_weights.sum()

        # Calculate social housing rent
        # Use Rent Paid if available, otherwise fall back to original column name
        rent_col = "Rent Paid" if "Rent Paid" in households_df.columns else "How much is usual household rent"
        if rent_col not in households_df.columns:
            # If neither exists, use a default value
            social_housing_rent = total_unemployment_benefits / scale * rent_as_fraction_of_unemployment_rate
        else:
            social_housing_rent = households_df[rent_col].mean() * rent_as_fraction_of_unemployment_rate

        # Calculate coefficient for financial assets income
        coefficient_fa_income = 0.05  # Default value, can be adjusted based on WAS data

        return cls(
            country_name="GBR",  # Hardcoded UK value
            country_name_short="GB",        # Hardcoded UK short code
            scale=scale,
            year=year,
            industries=industries,
            individual_data=individuals_df,
            household_data=households_df,
            social_housing_rent=social_housing_rent,
            coefficient_fa_income=coefficient_fa_income,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=np.array([consumption_weights] * n_quantiles),
            investment=investment_weights,
        )

    def restrict(self) -> None:
        """Restrict the household data to selected columns."""
        self.household_data = self.household_data[RESTRICT_COLS]

    def compute_household_wealth(self, independents: Optional[list[str]] = None) -> None:
        """
        Compute household wealth from WAS data.

        Args:
            independents (Optional[list[str]]): List of independent variables for wealth computation.
        """
        # Set real assets wealth
        self.set_household_other_real_assets_wealth()
        self.set_household_total_real_assets()

        # Set financial assets wealth
        self.set_household_deposits()
        self.set_household_other_financial_assets()
        self.set_household_financial_assets()

        # Set total wealth
        self.set_household_wealth()

        # Set debt
        self.set_household_mortgage_debt()
        self.set_household_other_debt()
        self.set_household_debt()

        # Set net wealth
        self.set_household_net_wealth()

    def compute_household_income(
        self,
        total_social_transfers: float,
        independents: Optional[list[str]] = None,
    ) -> None:
        """
        Compute household income from WAS data.

        Args:
            total_social_transfers (float): Total social transfers.
            independents (Optional[list[str]]): List of independent variables for income computation.
        """
        # Set employee income
        self.set_household_employee_income()

        # Set social transfers
        self.set_household_social_transfers(total_social_transfers, independents)

        # Set income from financial assets
        self.set_household_income_from_financial_assets()

        # Set total income
        self.set_household_income()

    def set_household_other_real_assets_wealth(self) -> None:
        """Set other real assets wealth from WAS data."""
        # Use mapped column name if available
        other_houses_col = "Value of other Properties" if "Value of other Properties" in self.household_data.columns else "Total value of other houses"
        if other_houses_col not in self.household_data.columns:
            self.household_data[other_houses_col] = 0.0
        
        # Sum up all real assets except main residence
        other_real_assets = (
            self.household_data[other_houses_col].fillna(0) +
            self.household_data.get("Total value of all vehicles", pd.Series(0, index=self.household_data.index)).fillna(0) +
            self.household_data.get("Value of all household goods and collectables", pd.Series(0, index=self.household_data.index)).fillna(0) +
            self.household_data.get("Approximate value of share of business after deducting outstanding debts", pd.Series(0, index=self.household_data.index)).fillna(0)
        )
        self.household_data["Wealth Other Real Assets"] = other_real_assets

    def set_household_total_real_assets(self) -> None:
        """Set total real assets wealth from WAS data."""
        # Use mapped column name if available
        main_residence_col = "Value of the Main Residence" if "Value of the Main Residence" in self.household_data.columns else "Value of main residence"
        if main_residence_col not in self.household_data.columns:
            self.household_data[main_residence_col] = 0.0
        
        self.household_data["Wealth in Real Assets"] = (
            self.household_data[main_residence_col].fillna(0) +
            self.household_data["Wealth Other Real Assets"]
        )

    def set_household_deposits(self) -> None:
        """Set deposits wealth from WAS data."""
        if "Total value of savings accounts" not in self.household_data.columns:
            self.household_data["Total value of savings accounts"] = 0.0
        self.household_data["Wealth in Deposits"] = self.household_data["Total value of savings accounts"].fillna(0)

    def set_household_other_financial_assets(self) -> None:
        """Set other financial assets wealth from WAS data."""
        # Ensure all columns exist
        for col in ["Total value of all formal financial assets", "Other Assets", "Total value of individual pension wealth"]:
            if col not in self.household_data.columns:
                self.household_data[col] = 0.0
        
        other_financial_assets = (
            self.household_data["Total value of all formal financial assets"].fillna(0) +
            self.household_data["Other Assets"].fillna(0) +
            self.household_data["Total value of individual pension wealth"].fillna(0)
        )
        self.household_data["Wealth in Other Financial Assets"] = other_financial_assets

    def set_household_financial_assets(self) -> None:
        """Set total financial assets wealth from WAS data."""
        self.household_data["Wealth in Financial Assets"] = (
            self.household_data["Wealth in Deposits"] +
            self.household_data["Wealth in Other Financial Assets"]
        )

    def set_household_wealth(self) -> None:
        """Set total household wealth from WAS data."""
        self.household_data["Wealth"] = (
            self.household_data["Wealth in Real Assets"] +
            self.household_data["Wealth in Financial Assets"]
        )

    def set_household_mortgage_debt(self) -> None:
        """Set mortgage debt from WAS data."""
        # Ensure columns exist
        for col in ["Total mortgage on main residence", "Total property debt excluding main residence"]:
            if col not in self.household_data.columns:
                self.household_data[col] = 0.0
        
        mortgage_debt = (
            self.household_data["Total mortgage on main residence"].fillna(0) +
            self.household_data["Total property debt excluding main residence"].fillna(0)
        )
        self.household_data["Outstanding Balance of HMR Mortgages"] = mortgage_debt

    def set_household_other_debt(self) -> None:
        """Set other debt from WAS data."""
        # Ensure columns exist
        for col in ["Hhold total outstanding credit/store/charge card balance", "Outstanding Balance of other Non-Mortgage Loans"]:
            if col not in self.household_data.columns:
                self.household_data[col] = 0.0
        
        other_debt = (
            self.household_data["Hhold total outstanding credit/store/charge card balance"].fillna(0) +
            self.household_data["Outstanding Balance of other Non-Mortgage Loans"].fillna(0)
        )
        self.household_data["Outstanding Balance of other Non-Mortgage Loans"] = other_debt

    def set_household_debt(self) -> None:
        """Set total household debt from WAS data."""
        self.household_data["Debt"] = (
            self.household_data["Outstanding Balance of HMR Mortgages"] +
            self.household_data["Outstanding Balance of other Non-Mortgage Loans"]
        )

    def set_household_net_wealth(self) -> None:
        """Set net household wealth from WAS data."""
        self.household_data["Net Wealth"] = (
            self.household_data["Wealth"] - self.household_data["Debt"]
        )

    def set_household_employee_income(self) -> None:
        """Set household employee income from WAS data."""
        # Sum employee income across all individuals in household
        # Use HID if available, otherwise fall back to Household identifier
        individual_id_col = "HID" if "HID" in self.individual_data.columns else "Household identifier"
        household_id_col = "HID" if "HID" in self.household_data.columns else "Household identifier"
        
        # Use Employee Income column if available, otherwise use original WAS column name
        income_col = "Employee Income" if "Employee Income" in self.individual_data.columns else "Gross annual income employee main job (including bonuses and commission received)"
        if income_col not in self.individual_data.columns:
            # If neither exists, set to 0
            self.individual_data[income_col] = 0.0
        
        employee_income = self.individual_data.groupby(individual_id_col)[income_col].sum()
        self.household_data["Employee Income"] = self.household_data[household_id_col].map(employee_income).fillna(0)

    def set_household_social_transfers(
        self, total_social_transfers: float, independents: Optional[list[str]] = None
    ) -> None:
        """
        Set household social transfers from WAS data.

        Args:
            total_social_transfers (float): Total social transfers.
            independents (Optional[list[str]]): List of independent variables.
        """
        # Use WAS-specific social transfers data
        if "Annual Household Income - Total benefits received" in self.household_data.columns:
            # Use actual WAS social transfers data
            self.household_data["Regular Social Transfers"] = self.household_data["Annual Household Income - Total benefits received"].fillna(0)
        else:
            # Fallback to proportional distribution
            total_households = len(self.household_data)
            self.household_data["Regular Social Transfers"] = total_social_transfers / total_households

    def set_household_income_from_financial_assets(self) -> None:
        """Set income from financial assets from WAS data."""
        if "Total income in £ over last 12 months received in dividends, interest or return on investments" in self.household_data.columns:
            self.household_data["Income from Financial Assets"] = self.household_data["Total income in £ over last 12 months received in dividends, interest or return on investments"].fillna(0)
        else:
            # Calculate based on financial assets and coefficient
            self.household_data["Income from Financial Assets"] = (
                self.household_data["Wealth in Financial Assets"] * self.coefficient_fa_income
            )

    def set_household_income(self) -> None:
        """Set total household income from WAS data."""
        if "Income" in self.household_data.columns:
            # Use actual WAS income data
            self.household_data["Income"] = self.household_data["Income"].fillna(0)
        else:
            # Calculate total income
            self.household_data["Income"] = (
                self.household_data["Employee Income"] +
                self.household_data["Regular Social Transfers"] +
                self.household_data["Income from Financial Assets"] +
                self.household_data["Annual Household Income - Gross rental income"].fillna(0)
            )

    def set_debt_installments(
        self, consumption_installments: np.ndarray, ce_installments: np.ndarray, mortgage_installments: np.ndarray
    ) -> None:
        """
        Set debt installments from WAS data.

        Args:
            consumption_installments (np.ndarray): Consumption loan installments.
            ce_installments (np.ndarray): Credit expansion installments.
            mortgage_installments (np.ndarray): Mortgage installments.
        """
        # Calculate total debt installments
        total_installments = consumption_installments + ce_installments + mortgage_installments
        # Ensure total_installments is the right length
        if isinstance(total_installments, (int, float)) or (hasattr(total_installments, '__len__') and len(total_installments) == 1):
            # If scalar or single value, broadcast to all households
            self.household_data["Debt Installments"] = np.full(len(self.household_data), total_installments if isinstance(total_installments, (int, float)) else total_installments[0])
        else:
            self.household_data["Debt Installments"] = total_installments

    def set_household_saving_rates(self, independents: Optional[list[str]] = None) -> None:
        """
        Set household saving rates from WAS data.

        Args:
            independents (Optional[list[str]]): List of independent variables.
        """
        # Use WAS-specific saving rate calculation
        # This is a simplified version - can be enhanced with more sophisticated modeling
        income = self.household_data["Income"]
        wealth = self.household_data["Wealth"]
        
        # Calculate saving rate based on income and wealth
        # Higher wealth households tend to have higher saving rates
        wealth_ratio = wealth / (wealth.mean() + 1e-6)  # Avoid division by zero
        base_saving_rate = 0.1  # Base saving rate of 10%
        wealth_effect = 0.05 * np.log(wealth_ratio + 1)  # Wealth effect on saving rate
        
        self.household_data["Saving Rate"] = np.clip(base_saving_rate + wealth_effect, 0, 0.5)

    def set_household_investment_rates(
        self,
        capital_formation_taxrate: float,
        default_investment_rates: np.ndarray | float = 0.2,
    ) -> None:
        """
        Set household investment rates from WAS data.

        Args:
            capital_formation_taxrate (float): Capital formation tax rate.
            default_investment_rates (np.ndarray | float): Default investment rates.
        """
        # Use WAS-specific investment rate calculation
        income = self.household_data["Income"]
        wealth = self.household_data["Wealth"]
        
        # Calculate investment rate based on income and wealth
        # Higher income and wealth households tend to invest more
        income_ratio = income / (income.mean() + 1e-6)
        wealth_ratio = wealth / (wealth.mean() + 1e-6)
        
        base_investment_rate = 0.15  # Base investment rate of 15%
        income_effect = 0.1 * np.log(income_ratio + 1)
        wealth_effect = 0.05 * np.log(wealth_ratio + 1)
        
        investment_rate = base_investment_rate + income_effect + wealth_effect
        investment_rate = np.clip(investment_rate, 0, 0.4)  # Cap at 40%
        
        self.household_data["Investment Rate"] = investment_rate

    @property
    def industry_consumption_before_vat(self):
        """Calculate household consumption by industry before VAT from WAS data."""
        cons_weights = self.consumption_weights
        income = self.household_data["Income"].values
        sr = self.household_data["Saving Rate"].values
        current_hh_consumption = default_desired_consumption(
            income_=income,
            consumption_weights_=cons_weights,
            saving_rates_=sr,
            tau_vat_=0,
        )
        return current_hh_consumption

    def normalise_household_consumption(
        self,
        iot_hh_consumption: np.ndarray | pd.Series,
        vat: float,
        positive_saving_rates_only: bool = True,
        independents: Optional[list[str]] = None,
    ) -> None:
        """
        Normalize household consumption from WAS data.

        Args:
            iot_hh_consumption (np.ndarray | pd.Series): IOT household consumption.
            vat (float): VAT rate.
            positive_saving_rates_only (bool): Whether to use only positive saving rates.
            independents (Optional[list[str]]): List of independent variables.
        """
        # Calculate desired consumption based on income and saving rates
        income = self.household_data["Income"].values
        saving_rates = self.household_data["Saving Rate"].values
        consumption_weights = self.consumption_weights
        
        # Calculate desired consumption
        desired_consumption = default_desired_consumption(income, consumption_weights, saving_rates, vat)
        
        # Normalize to match IOT consumption
        total_desired = desired_consumption.sum()
        total_iot = iot_hh_consumption.sum()
        scaling_factor = total_iot / (total_desired + 1e-6)
        
        # Apply scaling - sum across industries to get total consumption per household
        self.household_data["Amount spent on Consumption of Goods and Services"] = (
            desired_consumption * scaling_factor
        ).sum(axis=1) if len(desired_consumption.shape) > 1 else (desired_consumption * scaling_factor)

    def normalise_household_investment(
        self, tau_cf: float, iot_hh_investment: np.ndarray | pd.Series, positive_investment_rates: bool = True
    ):
        """
        Normalize household investment from WAS data.

        Args:
            tau_cf (float): Capital formation tax rate.
            iot_hh_investment (np.ndarray | pd.Series): IOT household investment.
            positive_investment_rates (bool): Whether to use only positive investment rates.
        """
        # Calculate desired investment based on income and investment rates
        income = self.household_data["Income"].values
        investment_rates = self.household_data["Investment Rate"].values
        investment_weights = self.investment_weights
        
        # Calculate desired investment
        desired_investment = default_target_investment(income, investment_weights, investment_rates, tau_cf)
        
        # Normalize to match IOT investment
        total_desired = desired_investment.sum()
        total_iot = iot_hh_investment.sum()
        scaling_factor = total_iot / (total_desired + 1e-6)
        
        # Apply scaling
        # Sum across industries to get total investment per household
        self.household_data["Investment"] = (
            desired_investment * scaling_factor
        ).sum(axis=1) if len(desired_investment.shape) > 1 else (desired_investment * scaling_factor)

    def match_consumption_weights_by_income(
        self,
        weights_by_income: np.ndarray | pd.DataFrame,
        iot_hh_consumption: pd.Series,
        vat: float,
        consumption_variance: float = 0.1,
    ) -> None:
        """
        Match consumption weights by income from WAS data.

        Args:
            weights_by_income (np.ndarray | pd.DataFrame): Weights by income.
            iot_hh_consumption (pd.Series): IOT household consumption.
            vat (float): VAT rate.
            consumption_variance (float): Consumption variance.
        """
        # Use WAS-specific consumption weight matching
        # This is a simplified version - can be enhanced with more sophisticated modeling
        income = self.household_data["Income"].values
        
        # Create income quantiles
        n_quantiles = weights_by_income.shape[1] if hasattr(weights_by_income, 'shape') else 5
        income_quantiles = pd.qcut(income, n_quantiles, labels=False, duplicates='drop')
        
        # Assign consumption weights based on income quantiles
        if isinstance(weights_by_income, pd.DataFrame):
            consumption_weights_by_income = weights_by_income.values
        else:
            consumption_weights_by_income = weights_by_income
        
        # Apply weights to households based on their income quantile
        household_weights = np.zeros((len(self.household_data), len(consumption_weights_by_income)))
        for i, quantile in enumerate(income_quantiles):
            if not np.isnan(quantile) and quantile < len(consumption_weights_by_income):
                household_weights[i] = consumption_weights_by_income[:, int(quantile)]
        
        self.consumption_weights_by_income = household_weights

    def set_wealth_distribution_function(self, independents: Optional[list[str]] = None) -> None:
        """
        Set wealth distribution function from WAS data.

        Args:
            independents (Optional[list[str]]): List of independent variables.
        """
        # Use WAS-specific wealth distribution modeling
        # This is a simplified version - can be enhanced with more sophisticated modeling
        income = self.household_data["Income"].values
        age = self.individual_data["Grouped age (17 categories)"].values if "Grouped age (17 categories)" in self.individual_data.columns else np.ones(len(income)) * 45
        
        # Simple wealth distribution model based on income and age
        # Higher income and older households tend to have more wealth
        income_effect = 0.8 * np.log(income + 1)
        age_effect = 0.1 * (age - 25) / 10  # Age effect normalized
        
        predicted_wealth = income_effect + age_effect
        predicted_wealth = np.maximum(predicted_wealth, 0)  # Ensure non-negative
        
        # Store the wealth distribution function
        self.wealth_distribution_model = LinearRegression()
        X = np.column_stack([income, age])
        self.wealth_distribution_model.fit(X, predicted_wealth)

    def add_emissions(
        self, emission_factors_array: np.ndarray, emitting_indices: list[int] | np.ndarray, tau_cf: float
    ) -> None:
        """
        Add emissions data from WAS data.

        Args:
            emission_factors_array (np.ndarray): Emission factors array.
            emitting_indices (list[int] | np.ndarray): Emitting indices.
            tau_cf (float): Capital formation tax rate.
        """
        # Calculate emissions based on consumption and investment
        consumption = self.household_data["Amount spent on Consumption of Goods and Services"].values
        investment = self.household_data["Investment"].values
        
        # Convert to proper numpy arrays if they contain lists
        if len(consumption) > 0 and isinstance(consumption[0], list):
            consumption = np.array([np.array(c) for c in consumption])
        if len(investment) > 0 and isinstance(investment[0], list):
            investment = np.array([np.array(i) for i in investment])
        
        # Calculate emissions from consumption
        consumption_emissions = np.zeros(len(consumption))
        for i, idx in enumerate(emitting_indices):
            if idx < consumption.shape[1] if len(consumption.shape) > 1 else idx < len(consumption):
                consumption_emissions += consumption[:, idx] * emission_factors_array[i]
        
        # Calculate emissions from investment
        investment_emissions = np.zeros(len(investment))
        for i, idx in enumerate(emitting_indices):
            if idx < investment.shape[1] if len(investment.shape) > 1 else idx < len(investment):
                investment_emissions += investment[:, idx] * emission_factors_array[i]
        
        # Total emissions
        total_emissions = consumption_emissions + investment_emissions
        self.household_data["Emissions"] = total_emissions

    @property
    def total_emissions(self) -> float:
        """Get total emissions from WAS data."""
        if "Emissions" in self.household_data.columns:
            return self.household_data["Emissions"].sum()
        else:
            return 0.0


def sample_households(
    hfcs_households_data: pd.DataFrame,
    hfcs_individuals_data: pd.DataFrame,
    n_households: int,
    output_shares: Optional[dict[str, pd.Series]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample households and individuals from WAS data.

    Args:
        hfcs_households_data (pd.DataFrame): WAS household data.
        hfcs_individuals_data (pd.DataFrame): WAS individual data.
        n_households (int): Number of households to sample.
        output_shares (Optional[dict[str, pd.Series]]): Output shares.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Sampled household and individual data.
    """
    # Sample households
    if len(hfcs_households_data) > n_households:
        sampled_households = hfcs_households_data.sample(n=n_households, random_state=42)
    else:
        sampled_households = hfcs_households_data.copy()
    
    # Get corresponding individuals
    # Use HID if available, otherwise fall back to Household identifier
    household_id_col = "HID" if "HID" in sampled_households.columns else "Household identifier"
    if household_id_col not in sampled_households.columns:
        # If neither exists, create HID from index
        sampled_households["HID"] = sampled_households.index
        household_id_col = "HID"
    
    household_ids = sampled_households[household_id_col].unique()
    
    # Ensure individuals have the same ID column
    individual_id_col = "HID" if "HID" in hfcs_individuals_data.columns else "Household identifier"
    if individual_id_col not in hfcs_individuals_data.columns:
        # If neither exists, try to match by creating HID from index or Corresponding Household ID
        if "Corresponding Household ID" in hfcs_individuals_data.columns:
            hfcs_individuals_data["HID"] = hfcs_individuals_data["Corresponding Household ID"]
        else:
            hfcs_individuals_data["HID"] = range(len(hfcs_individuals_data))
        individual_id_col = "HID"
    
    sampled_individuals = hfcs_individuals_data[hfcs_individuals_data[individual_id_col].isin(household_ids)]
    
    # Reset indices to ensure sequential indexing (0, 1, 2, ...) for compatibility with np.flatnonzero
    sampled_households = sampled_households.reset_index(drop=True)
    sampled_individuals = sampled_individuals.reset_index(drop=True)
    
    return sampled_households, sampled_individuals


def set_initial_values(household_data: pd.DataFrame, scale: int):
    """
    Set initial values for household data.

    Args:
        household_data (pd.DataFrame): Household data.
        scale (int): Scale factor.
    """
    # Set initial values for missing columns
    if "Corresponding Bank ID" not in household_data.columns:
        household_data["Corresponding Bank ID"] = np.random.randint(0, scale // 10, len(household_data))
    
    if "Corresponding Individuals ID" not in household_data.columns:
        household_data["Corresponding Individuals ID"] = range(len(household_data))
    
    # Set other missing columns to default values
    for col in ["Corresponding Inhabited House ID", "Corresponding Renters", 
                "Corresponding Property Owner", "Corresponding Additionally Owned Houses ID"]:
        if col not in household_data.columns:
            household_data[col] = 0


def default_desired_consumption(
    income_: np.ndarray,
    consumption_weights_: np.ndarray,
    saving_rates_: np.ndarray,
    tau_vat_: float,
) -> np.ndarray:
    """
    Calculate default desired consumption from WAS data.

    Args:
        income_ (np.ndarray): Income array.
        consumption_weights_ (np.ndarray): Consumption weights.
        saving_rates_ (np.ndarray): Saving rates.
        tau_vat_ (float): VAT rate.

    Returns:
        np.ndarray: Desired consumption.
    """
    return 1.0 / (1 + tau_vat_) * np.outer(consumption_weights_, (1 - saving_rates_) * income_).T
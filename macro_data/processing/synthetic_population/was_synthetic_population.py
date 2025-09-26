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

# WAS-specific column restrictions - using original WAS data variable names
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
    "Gross annual income employee main job (including bonuses and commission received)",
    "Annual Household Income - Total benefits received",
    "Annual Household Income - Gross rental income",
    "Total income in £ over last 12 months received in dividends, interest or return on investments",
    "Saving Rate",
    "How much is usual household rent",
    "Rent Imputed",
    "Wealth",
    "Net Wealth",
    "Wealth in Real Assets",
    "Value of main residence",
    "Total value of other houses",
    "Wealth Other Real Assets",
    "Total value of savings accounts",
    "Wealth in Other Financial Assets",
    "Wealth in Financial Assets",
    "Total mortgage on main residence",
    "Total property debt excluding main residence",
    "Outstanding Balance of other Non-Mortgage Loans",
    "Debt",
    "Debt Installments",
    "Tenure Status of the Main Residence",
    "Number of Properties other than Household Main Residence",
    # WAS-specific variables using original names
    "Total value of all vehicles",
    "Value of all household goods and collectables",
    "Approximate value of share of business after deducting outstanding debts",
    "Total value of all formal financial assets",
    "Total value of individual pension wealth",
    "Hhold total outstanding credit/store/charge card balance",
    "Burden of mortgage and other debt on household",
    "Burden from non-mortgage debt",
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
        uk_country = Country.GBR  # Use GBR enum value for UK
        
        # Get WAS data for UK (regardless of input country_name)
        was_reader = readers.was.get(uk_country)
        if was_reader is None:
            raise ValueError(f"No WAS data available for UK (WAS is UK-specific)")

        # Get household and individual data from WAS
        households_df = was_reader.households_df.copy()
        individuals_df = was_reader.individuals_df.copy()

        # Get unemployment and participation rates from labour stats
        unemployment_rate = exogenous_data.labour_stats.loc[f"{year}-Q{quarter}", "Unemployment Rate (Value)"].iloc[0]
        participation_rate = exogenous_data.labour_stats.loc[f"{year}-Q{quarter}", "Participation Rate (Value)"].iloc[0]

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

        # Set household types using WAS-specific tools
        households_df = set_household_types(households_df, individuals_df)

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
        social_housing_rent = households_df["How much is usual household rent"].mean() * rent_as_fraction_of_unemployment_rate

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
        # Sum up all real assets except main residence
        other_real_assets = (
            self.household_data["Total value of other houses"].fillna(0) +
            self.household_data["Total value of all vehicles"].fillna(0) +
            self.household_data["Value of all household goods and collectables"].fillna(0) +
            self.household_data["Approximate value of share of business after deducting outstanding debts"].fillna(0)
        )
        self.household_data["Wealth Other Real Assets"] = other_real_assets

    def set_household_total_real_assets(self) -> None:
        """Set total real assets wealth from WAS data."""
        self.household_data["Wealth in Real Assets"] = (
            self.household_data["Value of main residence"].fillna(0) +
            self.household_data["Wealth Other Real Assets"]
        )

    def set_household_deposits(self) -> None:
        """Set deposits wealth from WAS data."""
        self.household_data["Wealth in Deposits"] = self.household_data["Total value of savings accounts"].fillna(0)

    def set_household_other_financial_assets(self) -> None:
        """Set other financial assets wealth from WAS data."""
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
        mortgage_debt = (
            self.household_data["Total mortgage on main residence"].fillna(0) +
            self.household_data["Total property debt excluding main residence"].fillna(0)
        )
        self.household_data["Outstanding Balance of HMR Mortgages"] = mortgage_debt

    def set_household_other_debt(self) -> None:
        """Set other debt from WAS data."""
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
        employee_income = self.individual_data.groupby("Household identifier")["Gross annual income employee main job (including bonuses and commission received)"].sum()
        self.household_data["Employee Income"] = self.household_data["Household identifier"].map(employee_income).fillna(0)

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
        
        # Apply scaling
        self.household_data["Amount spent on Consumption of Goods and Services"] = (
            desired_consumption * scaling_factor
        )

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
        self.household_data["Investment"] = desired_investment * scaling_factor

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
    household_ids = sampled_households["Household identifier"].unique()
    sampled_individuals = hfcs_individuals_data[hfcs_individuals_data["Household identifier"].isin(household_ids)]
    
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
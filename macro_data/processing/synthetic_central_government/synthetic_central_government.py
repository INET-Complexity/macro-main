"""Module for preprocessing synthetic central government data.

This module provides an abstract base class for preprocessing and organizing central government
data that will be used to initialize behavioral models. Key preprocessing includes:

1. Data Collection and Organization:
   - Government revenue data preparation
   - Social benefits data processing
   - Tax revenue calculations

2. Financial Data Processing:
   - Tax revenue estimation
   - Benefits model parameter estimation
   - Revenue stream organization

3. Relationship Data:
   - Government-household transfers
   - Government-firm tax relationships
   - Social housing management

Note:
    This module is NOT used for simulating government behavior. It only handles
    the preprocessing and organization of data that will later be used to initialize
    behavioral models in the simulation package.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from macro_data.processing.country_data import TaxData
from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)


class SyntheticCentralGovernment(ABC):
    """Abstract base class for preprocessing and organizing central government data.

    This class provides a framework for collecting and organizing government data
    that will be used to initialize behavioral models. It is NOT used for simulating
    government behavior - it only handles data preprocessing.

    The preprocessed data is stored in a pandas DataFrame with the following columns:
        - Total Unemployment Benefits: Initial unemployment benefit levels
        - Other Social Benefits: Initial levels of other social transfers
        - Debt: Initial central government debt level
        - Bank Equity Injection: Initial bank support levels
        - Total Social Housing Rent: Initial social housing revenue
        - Taxes on Production: Initial production tax revenue
        - VAT: Initial value-added tax revenue
        - Capital Formation Taxes: Initial capital tax revenue
        - Export Taxes: Initial export tax revenue
        - Corporate Taxes: Initial corporate tax revenue
        - Employer SI Tax: Initial employer social insurance revenue
        - Employee SI Tax: Initial employee social insurance revenue
        - Income Taxes: Initial income tax revenue
        - Rental Income Taxes: Initial rental tax revenue
        - Revenue: Total initial revenue
        - Taxes on Products: Initial product tax revenue

    Note:
        This is a data container class. The actual government behavior (spending decisions,
        tax adjustments, etc.) is implemented in the simulation package, which uses this
        preprocessed data for initialization.

    Attributes:
        country_name (str): Country identifier for data collection
        year (int): Reference year for preprocessing
        central_gov_data (pd.DataFrame): Preprocessed government data
        other_benefits_model (Optional[LinearRegression]): Model for estimating other benefits
        unemployment_benefits_model (Optional[LinearRegression]): Model for estimating unemployment benefits
    """

    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
        central_gov_data: pd.DataFrame,
        other_benefits_model: Optional[LinearRegression],
        unemployment_benefits_model: Optional[LinearRegression],
    ):
        """Initialize the central government data container.

        Args:
            country_name (str): Country identifier for data collection
            year (int): Reference year for preprocessing
            central_gov_data (pd.DataFrame): Initial data to preprocess
            other_benefits_model (Optional[LinearRegression]): Model for estimating other benefits
            unemployment_benefits_model (Optional[LinearRegression]): Model for estimating unemployment benefits
        """
        self.country_name = country_name
        self.year = year

        # Central government data
        self.central_gov_data = central_gov_data

        # regressive models
        self.unemployment_benefits_model = unemployment_benefits_model
        self.other_benefits_model = other_benefits_model

    def update_fields(
        self,
        tax_data: TaxData,
        synthetic_population: SyntheticPopulation,
        synthetic_firms: SyntheticFirms,
        synthetic_banks: SyntheticBanks,
        industry_data: dict[str, pd.DataFrame],
    ) -> None:
        """Update preprocessed government data fields based on other economic agents.

        This method processes and organizes data from various economic agents to prepare:
        1. Social housing revenue data
        2. Tax revenue calculations
        3. Income and consumption data
        4. Financial asset data

        The preprocessing steps:
        1. Calculate social housing metrics
        2. Process firm and bank tax data
        3. Calculate income-based revenues
        4. Process consumption and investment taxes

        Args:
            tax_data (TaxData): Tax rates and parameters
            synthetic_population (SyntheticPopulation): Population data container
            synthetic_firms (SyntheticFirms): Firm data container
            synthetic_banks (SyntheticBanks): Bank data container
            industry_data (dict[str, pd.DataFrame]): Industry-level economic data
        """
        in_social_housing = synthetic_population.household_data["Tenure Status of the Main Residence"] == -1
        total_social_housing_rent = synthetic_population.social_housing_rent * in_social_housing.sum()
        firm_taxes_and_subsidies = float(synthetic_firms.firm_data["Taxes paid on Production"].sum())
        firm_corporate_taxes = float(synthetic_firms.firm_data["Corporate Taxes Paid"].sum())
        bank_corporate_taxes = float(synthetic_banks.bank_data["Corporate Taxes Paid"].sum())

        total_employee_income = synthetic_population.individual_data["Employee Income"].sum()

        firm_employer_si_tax = tax_data.employer_social_insurance_tax * total_employee_income

        # NOTE different to what was previously done, where consumption was computed using industry_data

        hh_saving_rate = synthetic_population.household_data["Saving Rate"]
        hh_income = synthetic_population.household_data["Income"]

        total_disposable_income = (hh_income * (1 - hh_saving_rate)).sum()

        household_vat = (
            tax_data.value_added_tax * industry_data["industry_vectors"]["Household Consumption in LCU"].sum()
        )

        export_tax = tax_data.export_tax * industry_data["industry_vectors"]["Exports in LCU"].sum()

        employee_si_tax = tax_data.employee_social_insurance_tax * total_employee_income

        employee_income_tax = tax_data.income_tax * (1 - tax_data.employee_social_insurance_tax) * total_employee_income

        total_rent_paid = synthetic_population.household_data["Rent Paid"].sum()
        rental_income_tax = tax_data.income_tax * total_rent_paid

        financial_assets_income = synthetic_population.household_data["Income from Financial Assets"].sum()
        financial_income_tax = tax_data.income_tax * financial_assets_income

        income_tax = employee_income_tax + rental_income_tax + financial_income_tax

        cf_tax = tax_data.capital_formation_tax

        household_gross_capital_inputs = industry_data["industry_vectors"]["Household Capital Inputs in LCU"].sum()

        self.set_revenue(
            total_social_housing_rent=total_social_housing_rent,
            firm_taxes_and_subsidies=firm_taxes_and_subsidies,
            firm_corporate_taxes=firm_corporate_taxes,
            bank_corporate_taxes=bank_corporate_taxes,
            firm_employer_si_tax=firm_employer_si_tax,
            household_vat=household_vat,
            export_tax=export_tax,
            employee_si_tax=employee_si_tax,
            income_tax=income_tax,
            rental_income_tax=rental_income_tax,
            cf_tax=cf_tax,
            household_gross_capital_inputs=household_gross_capital_inputs,
        )

    def set_revenue(
        self,
        total_social_housing_rent: float,
        firm_taxes_and_subsidies: float,
        firm_corporate_taxes: float,
        bank_corporate_taxes: float,
        firm_employer_si_tax: float,
        household_vat: float,
        export_tax: float,
        employee_si_tax: float,
        income_tax: float,
        rental_income_tax: float,
        household_gross_capital_inputs: float,
        cf_tax: float,
    ) -> None:
        """Set initial revenue values in the preprocessed government data.

        This method organizes and stores initial revenue values for:
        1. Social housing revenue
        2. Various tax revenues
        3. Total government revenue
        4. Product-specific taxes

        The preprocessing steps:
        1. Store individual revenue components
        2. Calculate and store total revenue
        3. Calculate and store product tax totals

        Args:
            total_social_housing_rent (float): Initial social housing revenue
            firm_taxes_and_subsidies (float): Initial production tax revenue
            firm_corporate_taxes (float): Initial firm tax revenue
            bank_corporate_taxes (float): Initial bank tax revenue
            firm_employer_si_tax (float): Initial employer SI revenue
            household_vat (float): Initial VAT revenue
            export_tax (float): Initial export tax revenue
            employee_si_tax (float): Initial employee SI revenue
            income_tax (float): Initial income tax revenue
            rental_income_tax (float): Initial rental tax revenue
            household_gross_capital_inputs (float): Initial capital inputs
            cf_tax (float): Capital formation tax rate
        """
        self.central_gov_data["Total Social Housing Rent"] = [total_social_housing_rent]
        self.central_gov_data["Taxes on Production"] = [firm_taxes_and_subsidies]
        self.central_gov_data["VAT"] = [household_vat]
        self.central_gov_data["Capital Formation Taxes"] = [cf_tax * household_gross_capital_inputs]
        self.central_gov_data["Export Taxes"] = [export_tax]
        self.central_gov_data["Corporate Taxes"] = [firm_corporate_taxes + bank_corporate_taxes]
        self.central_gov_data["Employer SI Tax"] = [firm_employer_si_tax]
        self.central_gov_data["Employee SI Tax"] = [employee_si_tax]
        self.central_gov_data["Income Taxes"] = [income_tax]
        self.central_gov_data["Rental Income Taxes"] = [rental_income_tax]
        self.central_gov_data["Revenue"] = [
            total_social_housing_rent
            + firm_taxes_and_subsidies
            + household_vat
            + cf_tax * household_gross_capital_inputs
            + export_tax
            + firm_corporate_taxes
            + bank_corporate_taxes
            + firm_employer_si_tax
            + employee_si_tax
            + income_tax
        ]
        self.central_gov_data["Taxes on Products"] = [
            firm_taxes_and_subsidies + household_vat + cf_tax * household_gross_capital_inputs + export_tax
        ]

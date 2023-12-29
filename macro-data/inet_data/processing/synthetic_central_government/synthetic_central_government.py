from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from inet_data.processing.synthetic_population.synthetic_population import SyntheticPopulation
from inet_data.readers.default_readers import DataReaders


class SyntheticCentralGovernment(ABC):
    """
    Represents a synthetic central government.

    Attributes:
        country_name (str): The name of the country.
        year (int): The year.
        central_gov_data (pd.DataFrame): The central government data.
        other_benefits_model (Optional[LinearRegression]): The model for other benefits (optional).
        unemployment_benefits_model (Optional[LinearRegression]): A linear regression model to determine unemployment benefits (optional).
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
        self.country_name = country_name
        self.year = year

        # Central government data
        self.central_gov_data = central_gov_data

        # regressive models
        self.unemployment_benefits_model = unemployment_benefits_model
        self.other_benefits_model = other_benefits_model

    def update_fields(
        self,
        readers: DataReaders,
        synthetic_population: SyntheticPopulation,
        synthetic_firms: SyntheticFirms,
        synthetic_banks: SyntheticBanks,
        industry_data: dict[str, pd.DataFrame],
    ) -> None:
        in_social_housing = synthetic_population.household_data["Tenure Status of the Main Residence"] == -1
        total_social_housing_rent = synthetic_population.social_housing_rent * in_social_housing.sum()
        firm_taxes_and_subsidies = float(synthetic_firms.firm_data["Taxes paid on Production"].sum())
        firm_corporate_taxes = float(synthetic_firms.firm_data["Corporate Taxes Paid"].sum())
        bank_corporate_taxes = float(synthetic_banks.bank_data["Corporate Taxes Paid"].sum())

        total_employee_income = synthetic_population.individual_data["Employee Income"].sum()

        firm_employer_si_tax = readers.oecd_econ.read_tau_sif(self.country_name, self.year) * total_employee_income

        # NOTE different to what was previously done, where consumption was computed using industry_data

        hh_saving_rate = synthetic_population.household_data["Saving Rate"]
        hh_income = synthetic_population.household_data["Income"]

        total_disposable_income = (hh_income * (1 - hh_saving_rate)).sum()

        household_vat = readers.world_bank.get_tau_vat(self.country_name, self.year) * total_disposable_income

        export_tax = (
            readers.world_bank.get_tau_exp(self.country_name, self.year)
            * industry_data["industry_vectors"]["Exports in LCU"].sum()
        )

        employee_si_tax = readers.oecd_econ.read_tau_siw(self.country_name, self.year) * total_employee_income

        employee_income_tax = (
            readers.oecd_econ.read_tau_income(self.country_name, self.year)
            * (1 - readers.oecd_econ.read_tau_siw(self.country_name, self.year))
            * total_employee_income
        )

        total_rent_paid = synthetic_population.household_data["Rent Paid"].sum()
        rental_income_tax = readers.oecd_econ.read_tau_income(self.country_name, self.year) * total_rent_paid

        financial_assets_income = synthetic_population.household_data["Income from Financial Assets"].sum()
        financial_income_tax = readers.oecd_econ.read_tau_income(self.country_name, self.year) * financial_assets_income

        income_tax = employee_income_tax + rental_income_tax + financial_income_tax

        # TODO this looks wrong, it's just a taxrate
        cf_tax = readers.eurostat.taxrate_on_capital_formation(self.country_name, self.year)

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
        cf_tax: float,
    ) -> None:
        self.central_gov_data["Total Social Housing Rent"] = [total_social_housing_rent]
        self.central_gov_data["Taxes on Production"] = [firm_taxes_and_subsidies]
        self.central_gov_data["VAT"] = [household_vat]
        self.central_gov_data["Capital Formation Taxes"] = [cf_tax * 0.0]
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
            + cf_tax * 0.0
            + export_tax
            + firm_corporate_taxes
            + bank_corporate_taxes
            + firm_employer_si_tax
            + employee_si_tax
            + income_tax
        ]
        self.central_gov_data["Taxes on Products"] = [
            firm_taxes_and_subsidies + household_vat + cf_tax * 0.0 + export_tax
        ]

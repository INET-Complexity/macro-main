import pandas as pd

from abc import abstractmethod, ABC

from typing import Any


class SyntheticCentralGovernment(ABC):
    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
    ):
        self.country_name = country_name
        self.year = year

        # Central government data
        self.central_gov_data = pd.DataFrame()

        # Parameters
        self.unemployment_benefits_model = None
        self.other_benefits_model = None

    def create(
        self,
        central_gov_debt: float,
        benefits_data: pd.DataFrame,
        exogenous_data: dict[str, Any],
        regression_window: int = 48,
    ) -> None:
        self.set_central_government_debt(central_gov_debt)
        self.set_total_unemployment_benefits(
            benefits_data=benefits_data,
            exogenous_data=exogenous_data,
            regression_window=regression_window,
        )
        self.set_other_social_benefits(
            benefits_data=benefits_data,
            exogenous_data=exogenous_data,
            regression_window=regression_window,
        )
        self.set_initial_bank_equity_injection()

    def update_fields(
        self,
        total_social_housing_rent: float,
        firm_taxes_and_subsidies: float,
        firm_corporate_taxes: float,
        firm_employer_si_tax: float,
        household_vat: float,
        export_tax: float,
        employee_si_tax: float,
        income_tax: float,
        rental_income_tax: float,
        cf_tax: float,
    ) -> None:
        self.set_revenue(
            total_social_housing_rent=total_social_housing_rent,
            firm_taxes_and_subsidies=firm_taxes_and_subsidies,
            firm_corporate_taxes=firm_corporate_taxes,
            firm_employer_si_tax=firm_employer_si_tax,
            household_vat=household_vat,
            export_tax=export_tax,
            employee_si_tax=employee_si_tax,
            income_tax=income_tax,
            rental_income_tax=rental_income_tax,
            cf_tax=cf_tax,
        )

    @abstractmethod
    def set_central_government_debt(self, central_gov_debt: float) -> None:
        pass

    @abstractmethod
    def set_total_unemployment_benefits(
        self,
        benefits_data: pd.DataFrame,
        exogenous_data: dict[str, Any],
        regression_window: int = 48,
    ) -> None:
        pass

    @abstractmethod
    def set_other_social_benefits(
        self,
        benefits_data: pd.DataFrame,
        exogenous_data: dict[str, Any],
        regression_window: int = 48,
    ) -> None:
        pass

    @abstractmethod
    def set_initial_bank_equity_injection(self) -> None:
        pass

    def set_revenue(
        self,
        total_social_housing_rent: float,
        firm_taxes_and_subsidies: float,
        firm_corporate_taxes: float,
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
        self.central_gov_data["Corporate Taxes"] = [firm_corporate_taxes]
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
            + firm_employer_si_tax
            + employee_si_tax
            + income_tax
        ]
        self.central_gov_data["Taxes on Products"] = [
            firm_taxes_and_subsidies + household_vat + cf_tax * 0.0 + export_tax
        ]

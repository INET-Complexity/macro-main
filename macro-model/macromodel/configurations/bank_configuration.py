from pydantic import BaseModel, Field
from typing import Literal, Any


class BankParameters(BaseModel):
    """
    Represents the parameters for a bank.

    Attributes:
        capital_adequacy_ratio (float): The Bank's capital adequacy ratio.
        firm_loans_debt_to_equity_ratio (float): The debt to equity ratio for firm loans.
        firm_loans_return_on_equity_ratio (float): The return on equity for firm loans.
        firm_loans_return_on_assets_ratio (float): The return on assets for firm loans.
        household_consumption_loans_loan_to_income_ratio (float): The loan to income ratio for household consumption loans.
        mortgage_loan_to_income_ratio (float): The loan to income ratio for mortgages.
        mortgage_loan_to_value_ratio (float): The loan to value ratio for mortgages.
        mortgage_debt_service_to_income_ratio (float): The debt service to income ratio for mortgages.
        household_consumption_loan_maturity (int): The maturity in periods of household consumption loans.
        long_term_firm_loan_maturity (int): The maturity in months of long-term firm loans.
        mortgage_maturity (int): The maturity in months of mortgages.
        short_term_firm_loan_maturity (int): The maturity in months of short-term firm loans.
    """  # noqa: D301

    capital_adequacy_ratio: float = Field(ge=0, le=1, default=0.08)
    firm_loans_debt_to_equity_ratio: float = Field(ge=0, le=1, default=0.03)
    firm_loans_return_on_equity_ratio: float = Field(ge=0, le=1, default=0.05)
    firm_loans_return_on_assets_ratio: float = Field(ge=0, le=1, default=0.05)
    household_consumption_loans_loan_to_income_ratio: float = Field(ge=0, le=1, default=0.05)
    mortgage_loan_to_income_ratio: float = Field(ge=0, le=1, default=0.05)
    mortgage_loan_to_value_ratio: float = Field(ge=0, le=1, default=0.05)
    mortgage_debt_service_to_income_ratio: float = Field(ge=0, le=1, default=0.05)
    household_consumption_loan_maturity: int = Field(ge=0, default=1)
    long_term_firm_loan_maturity: int = Field(ge=0, default=60)
    mortgage_maturity: int = Field(ge=0, default=120)
    short_term_firm_loan_maturity: int = Field(ge=0, default=20)


class DemographyFunction(BaseModel):
    """
    The function used to determine bank demography.
    """

    path_name: str = "demography"
    name: Literal["NoBankDemography", "DefaultBankDemography"] = "DefaultBankDemography"
    parameters: dict[str, Any] = {"solvency_ratio": 0.05}


class ProfitEstimatorFunction(BaseModel):
    """
    The function used to estimate bank profits.

    """

    path_name: str = "profit_estimator"
    name: Literal["DefaultBankProfitsSetter"] = "DefaultBankProfitsSetter"
    parameters: dict[str, Any] = {}


class InterestRateFunction(BaseModel):
    """
    The function used to set interest rates.
    """

    path_name: str = "interest_rates"
    name: Literal["DefaultInterestRatesSetter"] = "DefaultInterestRatesSetter"
    parameters: dict[str, Any] = {}


class BankFunctions(BaseModel):
    """
    Wrapper for the functions used by a bank
    """

    demography: DemographyFunction = DemographyFunction()
    interest_rates: InterestRateFunction = InterestRateFunction()
    profit_estimator: ProfitEstimatorFunction = ProfitEstimatorFunction()


class BanksConfiguration(BaseModel):
    """
    Represents the configuration for a bank.
    """

    parameters: BankParameters = BankParameters()
    functions: BankFunctions = BankFunctions()

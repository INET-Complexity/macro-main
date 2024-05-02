from pydantic import BaseModel, Field
from typing import Literal, Any


class BankParameters(BaseModel):
    """
    Represents the parameters for a bank.

    Attributes:
        capital_requirement_coefficient (float): The capital requirement coefficient. A capital requirement coefficient
                                                 of 3% corresponds to the maximum leverage ratio (tier 1 capital
                                                 in relation to total exposure) as recommended in the Basel III
                                                 framework.
        debt_service_to_income_ratio_mortgage (float): The debt service to income ratio for mortgages.
        loan_to_income_ratio_mortgage (float): The loan to income ratio for mortgages.
        loan_to_net_wealth_ratio (float): The loan to net wealth ratio for household consumption loans.
        loan_to_value_ratio (float): The loan to value ratio for firms.
        loan_to_value_ratio_mortgage (float): The loan to value ratio for mortgages.
        household_consumption_expansion_loan_maturity (int): The maturity in months of
                                                             household consumption expansion loans.
        household_payday_loan_maturity (int): The maturity in months of household payday loans.
        initial_markup_interest_rate_household_consumption_loans (float): The initial markup interest rate for household consumption loans.
        initial_markup_interest_rate_overdraft_households (float): The initial markup interest rate for overdrafts of households.
        initial_markup_mortgage_interest_rate (float): The initial markup interest rate for mortgages.
        long_term_firm_loan_maturity (int): The maturity in months of long-term firm loans.
        mortgage_maturity (int): The maturity in months of mortgages.
        short_term_firm_loan_maturity (int): The maturity in months of short-term firm loans.
        solvency_ratio (float): solvency ratio as the fraction of equity to assets, under which the bank is bankrupt.
    """  # noqa: D301

    capital_requirement_coefficient: float = Field(ge=0, le=1, default=0.03)
    debt_service_to_income_ratio_mortgage: float = Field(ge=0, le=1, default=0.6)
    loan_to_income_ratio_mortgage: float = Field(ge=0, le=1, default=0.2)
    loan_to_net_wealth_ratio: float = Field(ge=0, le=1, default=0.2)
    loan_to_value_ratio: float = Field(ge=0, le=1, default=0.6)
    loan_to_value_ratio_mortgage: float = Field(ge=0, le=1, default=0.5)
    household_consumption_expansion_loan_maturity: int = Field(ge=0, default=12)
    household_payday_loan_maturity: int = Field(ge=0, default=1)
    initial_markup_interest_rate_household_consumption_loans: float = Field(ge=0, default=0.01)
    initial_markup_interest_rate_overdraft_households: float = Field(ge=0, default=0.01)
    initial_markup_mortgage_interest_rate: float = Field(ge=0, default=0.1)
    long_term_firm_loan_maturity: int = Field(ge=0, default=60)
    mortgage_maturity: int = Field(ge=0, default=120)
    short_term_firm_loan_maturity: int = Field(ge=0, default=12)
    solvency_ratio: float = Field(ge=0, le=1, default=0.05)


class DemographyFunction(BaseModel):
    """
    The function used to determine bank demography.
    """

    path_name: str = "demography"
    name: Literal["NoBankDemography", "DefaultBankDemography"] = "DefaultBankDemography"
    parameters: dict[str, Any] = {"solvency_ratio": 0.05}


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


class BanksConfiguration(BaseModel):
    """
    Represents the configuration for a bank.
    """

    parameters: BankParameters = BankParameters()
    functions: BankFunctions = BankFunctions()

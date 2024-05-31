from pydantic import BaseModel
from typing import Literal, Any


class ClearingFunction(BaseModel):
    """
    The function used for clearing the credit market.
    """

    path_name: str = "clearing"
    name: Literal[
        "NoCreditMarketClearer",
        "DefaultCreditMarketClearer",
        "PolednaCreditMarketClearer",
        "WaterBucketCreditMarketClearer",
    ] = "WaterBucketCreditMarketClearer"
    parameters: dict[str, Any] = {
        "allow_short_term_firm_loans": True,
        "allow_household_loans": True,
        "firms_max_number_of_banks_visiting": 5,
        "households_max_number_of_banks_visiting": 5,
        "consider_loan_type_fractions": True,
        "credit_supply_temperature": 0.0,
        "interest_rates_selection_temperature": 0.0,
        "creditor_selection_is_deterministic": True,
        "creditor_minimum_fill": 0.0,
        "debtor_minimum_fill": 0.0,
    }


class CreditMarketFunctions(BaseModel):
    clearing: ClearingFunction = ClearingFunction()


class CreditMarketConfiguration(BaseModel):
    functions: CreditMarketFunctions = CreditMarketFunctions()

from typing import Any, Literal

from pydantic import BaseModel


class ClearingFunction(BaseModel):
    """Credit market clearing mechanism configuration.

    Defines the approach for matching credit supply and demand through:
    - Market clearing algorithms
    - Loan allocation rules
    - Bank-borrower matching
    - Credit rationing mechanisms

    The configuration supports multiple clearing strategies:
    - No clearing: Fixed credit relationships
    - Default clearing: Basic matching algorithm
    - Poledna clearing: Network-based allocation
    - Water bucket: Flow-based distribution

    The parameters control:
    - Loan type availability
    - Bank relationship limits
    - Credit allocation methods
    - Matching temperatures
    - Minimum fill rates

    Attributes:
        path_name (str): Module path for clearing functions
        name (Literal): Selected clearing mechanism
        parameters (dict): Configuration parameters including:
            - allow_short_term_firm_loans (bool): Enable firm short-term credit
            - allow_household_loans (bool): Enable household credit
            - firms_max_number_of_banks_visiting (int): Max bank relationships per firm
            - households_max_number_of_banks_visiting (int): Max bank relationships per household
            - consider_loan_type_fractions (bool): Use loan type quotas
            - credit_supply_temperature (float): Supply allocation randomness
            - interest_rates_selection_temperature (float): Rate selection randomness
            - creditor_selection_is_deterministic (bool): Use deterministic bank selection
            - creditor_minimum_fill (float): Minimum bank allocation rate
            - debtor_minimum_fill (float): Minimum borrower allocation rate
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
    """Collection of credit market function configurations.

    Aggregates the functional components that define credit market
    operations through:
    - Market clearing mechanisms
    - Credit allocation rules
    - Matching algorithms
    - Distribution methods

    Attributes:
        clearing (ClearingFunction): Market clearing mechanism configuration
    """

    clearing: ClearingFunction = ClearingFunction()


class CreditMarketConfiguration(BaseModel):
    """Complete credit market behavior configuration.

    Defines the overall configuration for credit market operations through:
    - Market clearing frameworks
    - Credit allocation mechanisms
    - Matching procedures
    - Distribution rules

    The configuration determines how the credit market:
    - Matches lenders and borrowers
    - Allocates available credit
    - Manages credit rationing
    - Implements matching rules

    Attributes:
        functions (CreditMarketFunctions): Collection of function configurations
            that define credit market behavior
    """

    functions: CreditMarketFunctions = CreditMarketFunctions()

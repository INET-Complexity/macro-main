from typing import Any, Literal

from pydantic import BaseModel, Field


class BankParameters(BaseModel):
    """Bank operational parameters configuration.

    Defines the key financial ratios and constraints that govern bank behavior through:
    - Capital requirements
    - Lending standards
    - Risk management
    - Maturity structures

    The parameters control:
    - Regulatory compliance (capital adequacy)
    - Credit risk management (loan ratios)
    - Portfolio performance (return ratios)
    - Loan terms (maturities)

    Attributes:
        capital_adequacy_ratio (float): Minimum capital to risk-weighted assets ratio
        firm_loans_debt_to_equity_ratio (float): Maximum firm loan leverage
        firm_loans_return_on_equity_ratio (float): Target return on equity for firm lending
        firm_loans_return_on_assets_ratio (float): Target return on assets for firm lending
        household_consumption_loans_loan_to_income_ratio (float): Maximum consumer loan to income
        mortgage_loan_to_income_ratio (float): Maximum mortgage to income ratio
        mortgage_loan_to_value_ratio (float): Maximum mortgage to property value
        mortgage_debt_service_to_income_ratio (float): Maximum mortgage payment to income
        household_consumption_loan_maturity (int): Consumer loan term in periods
        long_term_firm_loan_maturity (int): Long-term firm loan term in months
        mortgage_maturity (int): Mortgage term in months
        short_term_firm_loan_maturity (int): Short-term firm loan term in months
    """

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
    """Bank demography determination configuration.

    Defines the mechanism for managing bank population through:
    - Entry/exit decisions
    - Solvency assessment
    - Market structure evolution

    The configuration supports:
    - No demography changes (fixed population)
    - Default dynamics (entry/exit based on performance)

    Attributes:
        path_name (str): Module path for demography functions
        name (Literal): Selected demography mechanism
        parameters (dict): Configuration parameters including solvency thresholds
    """

    path_name: str = "demography"
    name: Literal["NoBankDemography", "DefaultBankDemography"] = "DefaultBankDemography"
    parameters: dict[str, Any] = {"solvency_ratio": 0.05}


class ProfitEstimatorFunction(BaseModel):
    """Bank profit estimation configuration.

    Defines the approach for calculating expected profits through:
    - Revenue projection
    - Cost estimation
    - Risk assessment
    - Performance targeting

    The configuration determines how banks:
    - Forecast earnings
    - Assess profitability
    - Set performance targets
    - Manage risk-return tradeoffs

    Attributes:
        path_name (str): Module path for profit estimation
        name (Literal): Selected estimation method
        parameters (dict): Configuration parameters for profit calculation
    """

    path_name: str = "profit_estimator"
    name: Literal["DefaultBankProfitsSetter"] = "DefaultBankProfitsSetter"
    parameters: dict[str, Any] = {}


class InterestRateFunction(BaseModel):
    """Bank interest rate determination configuration.

    Defines the mechanism for setting lending rates through:
    - Rate calculation methods
    - Risk premium determination
    - Market condition consideration
    - Policy rate transmission

    The configuration determines how banks:
    - Set loan rates
    - Adjust to market conditions
    - Incorporate risk premiums
    - Respond to policy changes

    Attributes:
        path_name (str): Module path for rate setting functions
        name (Literal): Selected rate setting mechanism
        parameters (dict): Configuration parameters for rate determination
    """

    path_name: str = "interest_rates"
    name: Literal["DefaultInterestRatesSetter"] = "DefaultInterestRatesSetter"
    parameters: dict[str, Any] = {}


class BankFunctions(BaseModel):
    """Collection of bank function configurations.

    Aggregates the various functional components that define
    bank operations through:
    - Population management
    - Rate setting behavior
    - Profit calculation
    - Performance assessment

    Attributes:
        demography (DemographyFunction): Bank population management
        interest_rates (InterestRateFunction): Interest rate determination
        profit_estimator (ProfitEstimatorFunction): Profit calculation
    """

    demography: DemographyFunction = DemographyFunction()
    interest_rates: InterestRateFunction = InterestRateFunction()
    profit_estimator: ProfitEstimatorFunction = ProfitEstimatorFunction()


class BanksConfiguration(BaseModel):
    """Complete bank behavior configuration.

    Defines the overall configuration for bank operations through:
    - Operational parameters
    - Functional components
    - Behavioral rules
    - Performance targets

    The configuration determines how banks:
    - Manage capital and risk
    - Set lending standards
    - Price loans
    - Assess performance
    - Enter/exit markets

    Attributes:
        parameters (BankParameters): Operational parameter settings
        functions (BankFunctions): Function implementations
    """

    parameters: BankParameters = BankParameters()
    functions: BankFunctions = BankFunctions()

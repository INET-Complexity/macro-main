"""Module for preprocessing synthetic credit market relationship data.

This module provides a framework for preprocessing and organizing credit relationship data
between banks, firms, and households. Key preprocessing includes:

1. Credit Relationship Data:
   - Bank-firm loan relationships
   - Bank-household loan relationships
   - Initial loan parameters

2. Loan Type Organization:
   - Long-term firm loans
   - Short-term firm loans
   - Consumer loans
   - Payday loans
   - Mortgage loans

3. Loan Parameter Processing:
   - Principal amounts
   - Interest rates
   - Installment calculations
   - Maturity periods

Note:
    This module is NOT used for simulating credit market behavior. It only handles
    the preprocessing and organization of credit relationship data that will later
    be used to initialize behavioral models in the simulation package.
"""

from dataclasses import dataclass

from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_credit_market.loan_data import (
    ConsumptionExpansionLoans,
    LongtermLoans,
    MortgageLoans,
    PaydayLoans,
    ShorttermLoans,
)
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)


@dataclass
class SyntheticCreditMarket:
    """Container for preprocessed credit market relationship data.

    This class organizes credit relationship data between financial institutions and borrowers,
    preprocessing initial loan states and parameters for model initialization. It does NOT
    implement credit market behavior - it only handles data preprocessing.

    The preprocessed data is organized into five loan categories:
    1. Long-term Loans: Initial firm long-term borrowing data
    2. Short-term Loans: Initial firm short-term credit data
    3. Consumer Loans: Initial household consumption borrowing data
    4. Payday Loans: Initial household short-term credit data
    5. Mortgage Loans: Initial household mortgage data

    Each loan category contains:
    - Principal amounts
    - Interest rates
    - Installment schedules
    - Bank-borrower relationships

    Note:
        This is a data container class. The actual credit market behavior (lending decisions,
        interest rate setting, etc.) is implemented in the simulation package, which uses
        this preprocessed data for initialization.

    Attributes:
        country_name (str): Country identifier for data collection
        year (int): Reference year for preprocessing
        longterm_loans (LongtermLoans): Preprocessed firm long-term loan data
        shortterm_loans (ShorttermLoans): Preprocessed firm short-term loan data
        consumption_expansion_loans (ConsumptionExpansionLoans): Preprocessed consumer loan data
        payday_loans (PaydayLoans): Preprocessed payday loan data
        mortgage_loans (MortgageLoans): Preprocessed mortgage loan data
    """

    country_name: str
    year: int
    longterm_loans: LongtermLoans
    shortterm_loans: ShorttermLoans
    consumption_expansion_loans: ConsumptionExpansionLoans
    payday_loans: PaydayLoans
    mortgage_loans: MortgageLoans

    @classmethod
    def create_from_agents(
        cls,
        firms: SyntheticFirms,
        population: SyntheticPopulation,
        banks: SyntheticBanks,
        zero_firm_debt: bool,
        firm_loan_maturity: int,
        hh_consumption_maturity: int,
        mortgage_maturity: int,
    ) -> "SyntheticCreditMarket":
        """Create a preprocessed credit market data container from agent data.

        This method processes credit relationship data from various economic agents to prepare:
        1. Firm loan data (long and short term)
        2. Household loan data (consumer and payday)
        3. Mortgage loan data

        The preprocessing steps:
        1. Extract loan data from firms and households
        2. Match with corresponding bank data
        3. Calculate initial loan parameters
        4. Organize into loan type categories

        Args:
            firms (SyntheticFirms): Firm data container
            population (SyntheticPopulation): Population data container
            banks (SyntheticBanks): Bank data container
            zero_firm_debt (bool): Whether to initialize with zero firm debt
            firm_loan_maturity (int): Initial maturity for firm loans
            hh_consumption_maturity (int): Initial maturity for consumer loans
            mortgage_maturity (int): Initial maturity for mortgages

        Returns:
            SyntheticCreditMarket: Container with preprocessed credit relationship data
        """
        # if zero_firm_debt:
        #     firm_loan_df = None
        # else:
        #     firm_loan_df = create_firm_loan_df(firms, banks, firm_loan_maturity)
        #
        # household_loan_df = create_household_loan_df(population, banks, hh_consumption_maturity)
        #
        # mortgage_loan_df = create_mortgage_loan_df(population, banks, mortgage_maturity)
        #
        # valid_firm_df = (firm_loan_df is not None) and (firm_loan_df.shape[0] > 0)
        #
        # if valid_firm_df:
        #     credit_list = [firm_loan_df, household_loan_df, mortgage_loan_df]
        # else:
        #     credit_list = [household_loan_df, mortgage_loan_df]
        #
        # credit_market_data = pd.concat(credit_list, ignore_index=True)
        #
        # credit_market_data.index.name = "Loans"
        # credit_market_data.columns.name = "Loan Properties"

        longterm_loans = LongtermLoans.from_agent_data(banks.bank_data, firms.firm_data, firm_loan_maturity)
        shortterm_loans = ShorttermLoans.from_agent_data(banks.bank_data, firms.firm_data, firm_loan_maturity)
        consumption_expansion_loans = ConsumptionExpansionLoans.from_agent_data(
            banks.bank_data, population.household_data, hh_consumption_maturity
        )
        payday_loans = PaydayLoans.from_agent_data(banks.bank_data, population.household_data, hh_consumption_maturity)
        mortgage_loans = MortgageLoans.from_agent_data(banks.bank_data, population.household_data, mortgage_maturity)

        return cls(
            country_name=firms.country_name,
            year=firms.year,
            longterm_loans=longterm_loans,
            shortterm_loans=shortterm_loans,
            consumption_expansion_loans=consumption_expansion_loans,
            payday_loans=payday_loans,
            mortgage_loans=mortgage_loans,
        )

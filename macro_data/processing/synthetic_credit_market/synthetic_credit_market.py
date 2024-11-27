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
    """
    Represents a synthetic credit market for a specific country and year.

    The credit market data is stored using dataclasses for long-term loans, short-term loans,
    consumption expansion loans, payday loans, and mortgage loans.


    Attributes:
        country_name (str): The name of the country.
        year (int): The year of the credit market data.
        longterm_loans (LongtermLoans): The long-term loans in the credit market.
        shortterm_loans (ShorttermLoans): The short-term loans in the credit market.
        consumption_expansion_loans (ConsumptionExpansionLoans): The consumption expansion loans in the credit market.
        payday_loans (PaydayLoans): The payday loans in the credit market.
        mortgage_loans (MortgageLoans): The mortgage loans in the credit market.

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

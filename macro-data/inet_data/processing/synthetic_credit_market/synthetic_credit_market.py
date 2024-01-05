from dataclasses import dataclass

import pandas as pd
from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks

from inet_data.processing.synthetic_population.synthetic_population import SyntheticPopulation

from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms

from inet_data.processing.synthetic_credit_market.default_synthetic_credit_market import (
    create_firm_loan_df,
    create_household_loan_df,
    create_mortgage_loan_df,
)


@dataclass
class SyntheticCreditMarket:
    """
    Represents a synthetic credit market for a specific country and year.

    The credit market data is stored in a pandas DataFrame with the following columns:
        - loan_type: The type of the loan (2 for firm loans, 4 for household loans, 5 for mortgage loans).
        - loan_value_initial Initial: The initial value of the loan.
        - loan_value: The current value of the loan.
        - loan_bank_id: The ID of the bank that issued the loan.
        - loan_recipient_id: The ID of the loan recipient.
        - loan_interest_rate: The interest rate of the loan.
        - loan_maturity: The maturity of the loan (in months).


    Attributes:
        country_name (str): The name of the country.
        year (int): The year of the credit market data.
        credit_market_data (pd.DataFrame): The credit market data for the country and year (contains information on loans
                                            including value, interest rate and maturity).
    """

    country_name: str
    year: int
    credit_market_data: pd.DataFrame

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
        if zero_firm_debt:
            firm_loan_df = None
        else:
            firm_loan_df = create_firm_loan_df(firms, banks, firm_loan_maturity)

        household_loan_df = create_household_loan_df(population, banks, hh_consumption_maturity)

        mortgage_loan_df = create_mortgage_loan_df(population, banks, mortgage_maturity)

        valid_firm_df = (firm_loan_df is not None) and (firm_loan_df.shape[0] > 0)

        if valid_firm_df:
            credit_list = [firm_loan_df, household_loan_df, mortgage_loan_df]
        else:
            credit_list = [household_loan_df, mortgage_loan_df]

        credit_market_data = pd.concat(credit_list, ignore_index=True)

        credit_market_data.index.name = "Loans"
        credit_market_data.columns.name = "Loan Properties"

        return cls(
            country_name=firms.country_name,
            year=firms.year,
            credit_market_data=credit_market_data,
        )

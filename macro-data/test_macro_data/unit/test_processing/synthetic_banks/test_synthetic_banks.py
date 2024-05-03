import pathlib

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.dataconfiguration import BanksDataConfiguration
from macro_data.processing.synthetic_banks.default_synthetic_banks import (
    DefaultSyntheticBanks,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticBanks:
    def test__create(self, readers, exogenous_data):
        banks_configuration = BanksDataConfiguration()
        banks = DefaultSyntheticBanks.from_readers(
            single_bank=True,
            country_name=Country("FRA"),
            year=2014,
            readers=readers,
            scale=10000,
            banks_data_configuration=banks_configuration,
            quarter=1,
            inflation_data=exogenous_data.inflation,
        )
        # banks.create(
        #     bank_equity=1000,
        # )
        # banks.bank_data["Corresponding Firms ID"] = [[0, 1]]
        # banks.bank_data["Corresponding Households ID"] = [[0, 1, 2]]
        # banks.set_initial_bank_fields(
        #     firm_deposits=np.array([200, 300]),
        #     firm_debt=np.array([100, 0]),
        #     household_deposits=np.array([100, 100, 100]),
        #     household_mortgage_debt=np.array([0.0, 1000.0, 700.0]),
        #     household_other_debt=np.array([10.0, 50.0, 100.0]),
        #     cb_policy_rate=0.01,
        #     tau_bank=0.1,
        #     bank_markup_interest_rate_short_term_firm_loans=0.01,
        #     bank_markup_interest_rate_long_term_firm_loans=0.01,
        #     bank_markup_interest_rate_household_payday_loans=0.01,
        #     bank_markup_interest_rate_household_consumption_loans=0.01,
        #     bank_markup_interest_rate_mortgages=0.01,
        #     bank_markup_interest_rate_overdraft_firm=0.01,
        #     bank_markup_interest_rate_overdraft_household=0.01,
        # )

        columns = {
            "Equity",
            "Loans to Firms",
            "Deposits from Firms",
            "Deposits from Households",
            "Mortgages to Households",
            "Consumption Loans to Households",
            "Loans to Households",
        }

        assert set(banks.bank_data.columns) == columns
        # Check if we have all the necessary fields
        # for bank_field in [
        #     "Equity",
        #     "Corresponding Firms ID",
        #     "Corresponding Households ID",
        #     "Deposits from Firms",
        #     "Deposits from Households",
        #     "Loans to Firms",
        #     "Mortgages to Households",
        #     "Consumption Loans to Households",
        #     "Deposits",
        #     "Short-Term Interest Rates on Firm Loans",
        #     "Long-Term Interest Rates on Firm Loans",
        #     "Interest Rates on Household Payday Loans",
        #     "Interest Rates on Household Consumption Loans",
        #     "Interest Rates on Mortgages",
        #     "Interest Rates on Firm Deposits",
        #     "Overdraft Rate on Firm Deposits",
        #     "Interest Rates on Household Deposits",
        #     "Overdraft Rate on Household Deposits",
        #     "Interest received from Loans",
        #     "Interest received from Deposits",
        #     "Profits",
        #     "Market Share",
        # ]:
        #     assert bank_field in banks.bank_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(banks.bank_data))

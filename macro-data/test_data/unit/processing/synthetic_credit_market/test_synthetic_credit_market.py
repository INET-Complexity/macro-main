import pathlib
import numpy as np
import pandas as pd

from data.processing.synthetic_credit_market.default_synthetic_credit_market import (
    DefaultSyntheticCreditMarket,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticCreditMarket:
    def test__create(
        self,
        readers,
    ):
        credit_market = DefaultSyntheticCreditMarket(
            country_name="FRA",
            year=2014,
        )
        credit_market.create(
            bank_data=pd.DataFrame(
                {
                    "Long-Term Interest Rates on Firm Loans": [0.02],
                    "Interest Rates on Household Consumption Loans": [0.03],
                    "Interest Rates on Mortgages": [0.1],
                }
            ),
            initial_firm_debt=np.array([100.0, 200.0]),
            initial_household_other_debt=np.array([10.0, 20.0]),
            initial_household_mortgage_debt=np.array([100.0, 200.0]),
            firms_corresponding_bank=np.array([0, 0]),
            households_corresponding_bank=np.array([0, 0]),
            initial_firm_loan_maturity=12,
            household_consumption_loan_maturity=12,
            mortgage_maturity=120,
            assume_zero_initial_firm_debt=True,
        )

        # Check if we have all the necessary fields
        for credit_market_field in [
            "loan_type",
            "loan_value_initial",
            "loan_value",
            "loan_maturity",
            "loan_interest_rate",
            "loan_bank_id",
            "loan_recipient_id",
        ]:
            assert credit_market_field in credit_market.credit_market_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(credit_market.credit_market_data))

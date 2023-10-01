import numpy as np
import pandas as pd

from inet_data.processing.synthetic_credit_market.synthetic_credit_market import (
    SyntheticCreditMarket,
)


class DefaultSyntheticCreditMarket(SyntheticCreditMarket):
    def __init__(
        self,
        country_name: str,
        year: int,
    ):
        super().__init__(
            country_name,
            year,
        )

    def create(
        self,
        bank_data: pd.DataFrame,
        initial_firm_debt: np.ndarray,
        initial_household_other_debt: np.ndarray,
        initial_household_mortgage_debt: np.ndarray,
        firms_corresponding_bank: np.ndarray,
        households_corresponding_bank: np.ndarray,
        initial_firm_loan_maturity: int,
        household_consumption_loan_maturity: int,
        mortgage_maturity: int,
        assume_zero_initial_firm_debt: bool,
    ) -> None:
        self.set_initial_loans(
            bank_data=bank_data,
            initial_firm_debt=initial_firm_debt,
            initial_household_other_debt=initial_household_other_debt,
            initial_household_mortgage_debt=initial_household_mortgage_debt,
            firms_corresponding_bank=firms_corresponding_bank,
            households_corresponding_bank=households_corresponding_bank,
            initial_firm_loan_maturity=initial_firm_loan_maturity,
            household_consumption_loan_maturity=household_consumption_loan_maturity,
            mortgage_maturity=mortgage_maturity,
            assume_zero_initial_firm_debt=assume_zero_initial_firm_debt,
        )

    def set_initial_loans(
        self,
        bank_data: pd.DataFrame,
        initial_firm_debt: np.ndarray,
        initial_household_other_debt: np.ndarray,
        initial_household_mortgage_debt: np.ndarray,
        firms_corresponding_bank: np.ndarray,
        households_corresponding_bank: np.ndarray,
        initial_firm_loan_maturity: int,
        household_consumption_loan_maturity: int,
        mortgage_maturity: int,
        assume_zero_initial_firm_debt: bool,
    ) -> None:
        loan_data: dict = {
            "loan_type": [],
            "loan_value_initial": [],
            "loan_value": [],
            "loan_maturity": [],
            "loan_interest_rate": [],
            "loan_bank_id": [],
            "loan_recipient_id": [],
        }

        # Firm loans
        if not assume_zero_initial_firm_debt:
            for firm_id in range(initial_firm_debt.shape[0]):
                if initial_firm_debt[firm_id] > 0:
                    loan_data["loan_type"].append(2)  # long-term loan
                    loan_data["loan_value_initial"].append(initial_firm_debt[firm_id])
                    loan_data["loan_value"].append(initial_firm_debt[firm_id])
                    loan_data["loan_maturity"].append(initial_firm_loan_maturity)
                    loan_data["loan_interest_rate"].append(
                        bank_data["Long-Term Interest Rates on Firm Loans"][firms_corresponding_bank[firm_id]]
                    )
                    loan_data["loan_bank_id"].append(firms_corresponding_bank[firm_id])
                    loan_data["loan_recipient_id"].append(firm_id)

        # Household consumption loans
        for household_id in range(initial_household_other_debt.shape[0]):
            if initial_household_other_debt[household_id] > 0:
                loan_data["loan_type"].append(4)  # consumption expansion loan
                loan_data["loan_value_initial"].append(initial_household_other_debt[household_id])
                loan_data["loan_value"].append(initial_household_other_debt[household_id])
                loan_data["loan_maturity"].append(household_consumption_loan_maturity)
                loan_data["loan_interest_rate"].append(
                    bank_data["Interest Rates on Household Consumption Loans"][
                        households_corresponding_bank[household_id]
                    ]
                )
                loan_data["loan_bank_id"].append(households_corresponding_bank[household_id])
                loan_data["loan_recipient_id"].append(household_id)

        # Mortgages
        for household_id in range(initial_household_mortgage_debt.shape[0]):
            if initial_household_mortgage_debt[household_id] > 0:
                loan_data["loan_type"].append(5)  # mortgage
                loan_data["loan_value_initial"].append(initial_household_mortgage_debt[household_id])
                loan_data["loan_value"].append(initial_household_mortgage_debt[household_id])
                loan_data["loan_maturity"].append(mortgage_maturity)
                loan_data["loan_interest_rate"].append(
                    bank_data["Interest Rates on Mortgages"][households_corresponding_bank[household_id]]
                )
                loan_data["loan_bank_id"].append(households_corresponding_bank[household_id])
                loan_data["loan_recipient_id"].append(household_id)

        # Compile it all
        self.credit_market_data = pd.DataFrame(data=loan_data)
        self.credit_market_data.index.name = "Loans"
        self.credit_market_data.columns.name = "Loan Properties"

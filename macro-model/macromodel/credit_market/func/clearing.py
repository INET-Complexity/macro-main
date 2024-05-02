import numpy as np
import pandas as pd
from abc import abstractmethod, ABC

from macromodel.banks.banks import Banks
from macromodel.credit_market.types_of_loans import LoanTypes
from macromodel.firms.firms import Firms
from macromodel.households.households import Households


class CreditMarketClearer(ABC):
    def __init__(
        self,
        firms_max_number_of_banks_visiting: int,
        households_max_number_of_banks_visiting: int,
    ):
        self.firms_max_number_of_banks_visiting = firms_max_number_of_banks_visiting
        self.households_max_number_of_banks_visiting = households_max_number_of_banks_visiting

    @staticmethod
    @abstractmethod
    def clear(
        banks: Banks,
        firms: Firms,
        households: Households,
    ) -> pd.DataFrame:
        pass


class NoCreditMarketClearer(CreditMarketClearer):
    @staticmethod
    def clear(
        banks: Banks,
        firms: Firms,
        households: Households,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "loan_type",
                "loan_value_initial",
                "loan_value",
                "loan_maturity",
                "loan_interest_rate",
                "loan_bank_id",
                "loan_recipient_id",
            ]
        )


class DefaultCreditMarketClearer(CreditMarketClearer):
    def clear(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
    ) -> pd.DataFrame:
        # Keeping track of new credit
        new_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_credit_by_firm = np.zeros(firms.ts.current("n_firms"))
        new_credit_by_household = np.zeros(households.ts.current("n_households"))

        # Firm loans
        new_short_term_firm_loans = self.clear_firm_loans(
            banks=banks,
            firms=firms,
            loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
        )
        new_long_term_firm_loans = self.clear_firm_loans(
            banks=banks,
            firms=firms,
            loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
        )

        # Household consumption loans
        new_household_payday_loans = self.clear_household_consumption_loans(
            banks=banks,
            households=households,
            loan_type=LoanTypes.HOUSEHOLD_PAYDAY_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_household=new_credit_by_household,
        )
        new_household_consumption_expansion_loans = self.clear_household_consumption_loans(
            banks=banks,
            households=households,
            loan_type=LoanTypes.HOUSEHOLD_CONSUMPTION_EXPANSION_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_household=new_credit_by_household,
        )

        # Mortgages
        new_mortgages = self.clear_mortgages(
            banks=banks,
            households=households,
            loan_type=LoanTypes.MORTGAGE,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_household=new_credit_by_household,
        )

        # Collect them all
        new_loans = pd.concat(
            (
                new_short_term_firm_loans,
                new_long_term_firm_loans,
                new_household_payday_loans,
                new_household_consumption_expansion_loans,
                new_mortgages,
            ),
            axis=0,
        ).reset_index(drop=True)
        new_loans["loan_bank_id"] = new_loans["loan_bank_id"].astype(int)
        new_loans["loan_recipient_id"] = new_loans["loan_recipient_id"].astype(int)

        return new_loans

    def clear_firm_loans(
        self,
        banks: Banks,
        firms: Firms,
        loan_type: LoanTypes,
        new_credit_by_bank: np.ndarray,
        new_credit_by_firm: np.ndarray,
    ) -> pd.DataFrame:
        # Data on new loans
        new_loan_types = []
        new_loan_value = []
        new_loan_maturity = []
        new_loan_interest_rate = []
        new_loan_bank_id = []
        new_loan_recipient_id = []

        # Select loan maturity
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            loan_maturity = banks.parameters.short_term_firm_loan_maturity
        else:
            loan_maturity = banks.parameters.long_term_firm_loan_maturity

        # Get bank interest rates
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            banks_ir = np.array(
                [
                    banks.states["interest_rate_on_firm_short_term_loans_function"](bank_id)
                    for bank_id in range(banks.ts.current("n_banks"))
                ]
            )
            banks.ts.average_interest_rates_on_short_term_firm_loans.append([banks_ir.mean()])
        else:
            banks_ir = np.array(
                [
                    banks.states["interest_rate_on_firm_long_term_loans_function"](bank_id)
                    for bank_id in range(banks.ts.current("n_banks"))
                ]
            )
            banks.ts.average_interest_rates_on_long_term_firm_loans.append([banks_ir.mean()])

        # Iterate over firms with financing needs
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            firm_target_credit = firms.ts.current("target_short_term_credit")
        else:
            firm_target_credit = firms.ts.current("target_long_term_credit")
        firms_with_needs = np.where(firm_target_credit > 0)[0]
        firms_with_needs_shuffled = np.random.choice(
            firms_with_needs,
            len(firms_with_needs),
            replace=False,
        )
        for firm_id in firms_with_needs_shuffled:
            # Take a subset of all banks
            banks_subset = np.random.choice(
                range(banks.ts.current("n_banks")),
                min(self.firms_max_number_of_banks_visiting, banks.ts.current("n_banks")),
                replace=False,
            )

            # Iterate over all banks based on the offered interest rate
            for bank_id in banks_subset[np.argsort(banks_ir[banks_subset])]:
                if firm_target_credit[firm_id] == 0:
                    break
                bank_cap_req = (
                    banks.ts.current("equity")[bank_id] / banks.parameters.capital_requirement_coefficient
                    - banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )
                firm_risk_assessment = (
                    banks.parameters.loan_to_value_ratio * firms.ts.current("capital_inputs_stock_value")[firm_id]
                    - firms.ts.current("debt")[firm_id]
                    - new_credit_by_firm[firm_id]
                    + min(0, firms.ts.current("deposits")[firm_id])
                )
                value_granted = max(
                    0.0,
                    min(
                        firm_target_credit[firm_id],
                        bank_cap_req,
                        firm_risk_assessment,
                    ),
                )

                # Record the new loans
                if value_granted > 0:
                    new_credit_by_bank[bank_id] += value_granted
                    new_credit_by_firm[firm_id] += value_granted
                    firm_target_credit[firm_id] -= value_granted
                    new_loan_types.append(loan_type)
                    new_loan_value.append(value_granted)
                    new_loan_maturity.append(loan_maturity)
                    new_loan_interest_rate.append(banks_ir[bank_id])
                    new_loan_bank_id.append(bank_id)
                    new_loan_recipient_id.append(firm_id)

        return pd.DataFrame(
            data={
                "loan_type": new_loan_types,
                "loan_value_initial": new_loan_value,
                "loan_value": new_loan_value,
                "loan_maturity": new_loan_maturity,
                "loan_interest_rate": new_loan_interest_rate,
                "loan_bank_id": new_loan_bank_id,
                "loan_recipient_id": new_loan_recipient_id,
            }
        )

    def clear_household_consumption_loans(
        self,
        banks: Banks,
        households: Households,
        loan_type: LoanTypes,
        new_credit_by_bank: np.ndarray,
        new_credit_by_household: np.ndarray,
    ) -> pd.DataFrame:
        # Data on new loans
        new_loan_types = []
        new_loan_value = []
        new_loan_maturity = []
        new_loan_interest_rate = []
        new_loan_bank_id = []
        new_loan_recipient_id = []

        # Select loan maturity
        if loan_type == LoanTypes.HOUSEHOLD_PAYDAY_LOAN:
            loan_maturity = banks.parameters.household_payday_loan_maturity
        else:
            loan_maturity = banks.parameters.household_consumption_expansion_loan_maturity

        # Get bank interest rates
        if loan_type == LoanTypes.HOUSEHOLD_PAYDAY_LOAN:
            banks_ir = np.array(
                [
                    banks.states["interest_rate_on_household_payday_loans_function"](bank_id)
                    for bank_id in range(banks.ts.current("n_banks"))
                ]
            )
            banks.ts.average_interest_rates_on_household_payday_loans.append([banks_ir.mean()])
        else:
            banks_ir = np.array(
                [
                    banks.states["interest_rate_on_household_consumption_expansion_loans_function"](bank_id)
                    for bank_id in range(banks.ts.current("n_banks"))
                ]
            )
            banks.ts.average_interest_rates_on_household_consumption_loans.append([banks_ir.mean()])

        # Iterate over households with financing needs
        if loan_type == LoanTypes.HOUSEHOLD_PAYDAY_LOAN:
            household_target_credit = households.ts.current("target_payday_loans")
        else:
            household_target_credit = households.ts.current("target_consumption_expansion_loans")
        households_with_needs = np.where(household_target_credit > 0)[0]
        households_with_needs_shuffled = np.random.choice(
            households_with_needs,
            len(households_with_needs),
            replace=False,
        )
        for household_id in households_with_needs_shuffled:
            # Take a subset of all banks
            banks_subset = np.random.choice(
                range(banks.ts.current("n_banks")),
                min(
                    self.households_max_number_of_banks_visiting,
                    banks.ts.current("n_banks"),
                ),
                replace=False,
            )

            # Iterate over all banks based on the offered interest rate
            for bank_id in banks_subset[np.argsort(banks_ir[banks_subset])]:
                if household_target_credit[household_id] == 0:
                    break
                bank_cap_req = (
                    banks.ts.current("equity")[bank_id] / banks.parameters.capital_requirement_coefficient
                    - banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )
                household_risk_assessment = (
                    banks.parameters.loan_to_net_wealth_ratio * households.ts.current("net_wealth")[household_id]
                    - new_credit_by_household[household_id]
                )
                value_granted = max(
                    0.0,
                    min(
                        household_target_credit[household_id],
                        bank_cap_req,
                        household_risk_assessment,
                    ),
                )

                # Record the new loans
                if value_granted > 0:
                    new_credit_by_bank[bank_id] += value_granted
                    new_credit_by_household[household_id] += value_granted
                    household_target_credit[household_id] -= value_granted
                    new_loan_types.append(loan_type)
                    new_loan_value.append(value_granted)
                    new_loan_maturity.append(loan_maturity)
                    new_loan_interest_rate.append(banks_ir[bank_id])
                    new_loan_bank_id.append(bank_id)
                    new_loan_recipient_id.append(household_id)

        return pd.DataFrame(
            data={
                "loan_type": new_loan_types,
                "loan_value_initial": new_loan_value,
                "loan_value": new_loan_value,
                "loan_maturity": new_loan_maturity,
                "loan_interest_rate": new_loan_interest_rate,
                "loan_bank_id": new_loan_bank_id,
                "loan_recipient_id": new_loan_recipient_id,
            }
        )

    def clear_mortgages(
        self,
        banks: Banks,
        households: Households,
        loan_type: LoanTypes,
        new_credit_by_bank: np.ndarray,
        new_credit_by_household: np.ndarray,
    ) -> pd.DataFrame:
        # Data on new loans
        new_loan_types = []
        new_loan_value = []
        new_loan_maturity = []
        new_loan_interest_rate = []
        new_loan_bank_id = []
        new_loan_recipient_id = []

        # Get bank interest rates
        banks_ir = np.array(
            [
                banks.states["interest_rate_on_mortgages_function"](bank_id)
                for bank_id in range(banks.ts.current("n_banks"))
            ]
        )
        banks.ts.average_interest_rates_on_mortgages.append([banks_ir.mean()])

        # Iterate over households with financing needs
        household_target_credit = households.ts.current("target_mortgage")
        households_with_needs = np.where(household_target_credit > 0)[0]
        households_with_needs_shuffled = np.random.choice(
            households_with_needs,
            len(households_with_needs),
            replace=False,
        )
        for household_id in households_with_needs_shuffled:
            # Take a subset of all banks
            banks_subset = np.random.choice(
                range(banks.ts.current("n_banks")),
                min(
                    self.households_max_number_of_banks_visiting,
                    banks.ts.current("n_banks"),
                ),
                replace=False,
            )

            # Iterate over all banks based on the offered interest rate
            for bank_id in banks_subset[np.argsort(banks_ir[banks_subset])]:
                if household_target_credit[household_id] == 0:
                    break
                loan_to_value_limit = (
                    banks.parameters.loan_to_value_ratio_mortgage
                    / (1 - banks.parameters.loan_to_value_ratio_mortgage)
                    * households.ts.current("wealth_financial_assets")[household_id]
                )
                loan_to_income_limit = (
                    banks.parameters.loan_to_income_ratio_mortgage * households.ts.current("income")[household_id]
                )
                debt_service_to_income_limit = (
                    banks.parameters.debt_service_to_income_ratio_mortgage
                    * households.ts.current("income")[household_id]
                    * (1 - (1 + banks_ir[bank_id]) ** (-banks.parameters.mortgage_maturity))
                    / banks_ir[bank_id]
                )
                bank_cap_req = (
                    banks.ts.current("equity")[bank_id] / banks.parameters.capital_requirement_coefficient
                    - banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )
                value_granted = max(
                    0.0,
                    min(
                        household_target_credit[household_id],
                        loan_to_value_limit,
                        loan_to_income_limit,
                        debt_service_to_income_limit,
                        bank_cap_req,
                    ),
                )

                # Only take the mortgage, if we get the full amount
                if value_granted < household_target_credit[household_id]:
                    continue

                # Record the new mortgage
                if value_granted > 0:
                    new_credit_by_bank[bank_id] += value_granted
                    new_credit_by_household[household_id] += value_granted
                    household_target_credit[household_id] -= value_granted
                    new_loan_types.append(loan_type)
                    new_loan_value.append(value_granted)
                    new_loan_maturity.append(banks.parameters.mortgage_maturity)
                    new_loan_interest_rate.append(banks_ir[bank_id])
                    new_loan_bank_id.append(bank_id)
                    new_loan_recipient_id.append(household_id)

        return pd.DataFrame(
            data={
                "loan_type": new_loan_types,
                "loan_value_initial": new_loan_value,
                "loan_value": new_loan_value,
                "loan_maturity": new_loan_maturity,
                "loan_interest_rate": new_loan_interest_rate,
                "loan_bank_id": new_loan_bank_id,
                "loan_recipient_id": new_loan_recipient_id,
            }
        )

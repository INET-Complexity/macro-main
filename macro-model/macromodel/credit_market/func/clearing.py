import numpy as np
import pandas as pd

from numba import njit

from abc import abstractmethod, ABC
from macromodel.banks.banks import Banks
from macromodel.firms.firms import Firms
from macromodel.households.households import Households
from macromodel.credit_market.types_of_loans import LoanTypes

from typing import Tuple


class CreditMarketClearer(ABC):
    def __init__(
        self,
        allow_short_term_firm_loans: bool,
        allow_household_loans: bool,
        firms_max_number_of_banks_visiting: int,
        households_max_number_of_banks_visiting: int,
        consider_loan_type_fractions: bool,
        credit_supply_temperature: float,
        interest_rates_selection_temperature: float,
        creditor_selection_is_deterministic: bool,
        creditor_minimum_fill: bool,
        debtor_minimum_fill: bool,
    ):
        self.allow_short_term_firm_loans = allow_short_term_firm_loans
        self.allow_household_loans = allow_household_loans
        self.firms_max_number_of_banks_visiting = firms_max_number_of_banks_visiting
        self.households_max_number_of_banks_visiting = households_max_number_of_banks_visiting
        self.consider_loan_type_fractions = consider_loan_type_fractions
        self.credit_supply_temperature = credit_supply_temperature
        self.interest_rates_selection_temperature = interest_rates_selection_temperature
        self.creditor_selection_is_deterministic = creditor_selection_is_deterministic
        self.creditor_minimum_fill = creditor_minimum_fill
        self.debtor_minimum_fill = debtor_minimum_fill

    @staticmethod
    @abstractmethod
    def clear(
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> pd.DataFrame:
        pass


class NoCreditMarketClearer(CreditMarketClearer):
    @staticmethod
    def clear(
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
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
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> pd.DataFrame:
        empty = pd.DataFrame(
            data={
                "loan_type": [],
                "loan_value_initial": [],
                "loan_value": [],
                "loan_maturity": [],
                "loan_interest_rate": [],
                "loan_bank_id": [],
                "loan_recipient_id": [],
            }
        )

        # Keeping track of new credit
        new_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_credit_by_firm = np.zeros(firms.ts.current("n_firms"))
        new_credit_by_household = np.zeros(households.ts.current("n_households"))

        # Banks may update their preferences for different types of loans, impacting their supply
        if self.consider_loan_type_fractions:
            max_car = np.maximum(
                0.0,
                banks.ts.current("equity") / banks.parameters["capital_adequacy_ratio"]["value"]
                - banks.ts.current("total_outstanding_loans")
                - new_credit_by_bank,
            )
            max_supply_based_on_preferences_firms = banks.ts.initial("new_loans_fraction_firms") * np.exp(
                -self.credit_supply_temperature * current_npl_firm_loans
            )
            max_supply_based_on_preferences_hh_cons = banks.ts.initial("new_loans_fraction_hh_cons") * np.exp(
                -self.credit_supply_temperature * current_npl_hh_cons_loans
            )
            max_supply_based_on_preferences_mortgages = banks.ts.initial("new_loans_fraction_mortgages") * np.exp(
                -self.credit_supply_temperature * current_npl_mortgages
            )
            current_sum = (
                max_supply_based_on_preferences_firms * max_car
                + max_supply_based_on_preferences_hh_cons * max_car
                + max_supply_based_on_preferences_mortgages * max_car
            )
            scale = np.divide(
                max_car,
                current_sum,
                out=np.zeros(max_car.shape),
                where=current_sum != 0.0,
            )
            max_supply_based_on_preferences_firms *= scale
            max_supply_based_on_preferences_hh_cons *= scale
            max_supply_based_on_preferences_mortgages *= scale
        else:
            max_supply_based_on_preferences_firms = np.full(banks.ts.current("n_banks"), np.inf)
            max_supply_based_on_preferences_hh_cons = np.full(banks.ts.current("n_banks"), np.inf)
            max_supply_based_on_preferences_mortgages = np.full(banks.ts.current("n_banks"), np.inf)

        # Firm loans
        if self.allow_short_term_firm_loans:
            new_short_term_firm_loans = self.clear_firm_loans(
                banks=banks,
                firms=firms,
                loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
                new_credit_by_bank=new_credit_by_bank,
                new_credit_by_firm=new_credit_by_firm,
                max_supply_based_on_preferences=max_supply_based_on_preferences_firms,
            )
        else:
            new_short_term_firm_loans = empty.copy()
        new_long_term_firm_loans = self.clear_firm_loans(
            banks=banks,
            firms=firms,
            loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            max_supply_based_on_preferences=max_supply_based_on_preferences_firms,
        )

        # Household loans
        if self.allow_household_loans:
            new_household_consumption_loans = self.clear_household_consumption_loans(
                banks=banks,
                households=households,
                new_credit_by_bank=new_credit_by_bank,
                new_credit_by_household=new_credit_by_household,
                max_supply_based_on_preferences=max_supply_based_on_preferences_hh_cons,
            )
            new_mortgages = self.clear_mortgages(
                banks=banks,
                households=households,
                loan_type=LoanTypes.MORTGAGE,
                new_credit_by_bank=new_credit_by_bank,
                new_credit_by_household=new_credit_by_household,
                max_supply_based_on_preferences=max_supply_based_on_preferences_mortgages,
            )
        else:
            new_household_consumption_loans = empty.copy()
            new_mortgages = empty.copy()

        # Collect them all
        new_loans = pd.concat(
            (
                new_short_term_firm_loans,
                new_long_term_firm_loans,
                new_household_consumption_loans,
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
        max_supply_based_on_preferences: np.ndarray,
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
            loan_maturity = banks.parameters["short_term_firm_loan_maturity"]["value"]
        else:
            loan_maturity = banks.parameters["long_term_firm_loan_maturity"]["value"]

        # Get bank interest rates
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            banks_ir = banks.ts.current("interest_rates_on_short_term_firm_loans")
        else:
            banks_ir = banks.ts.current("interest_rates_on_long_term_firm_loans")

        # Iterate over firms with financing needs
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            firm_target_credit = firms.ts.current("target_short_term_credit")
        else:
            firm_target_credit = firms.ts.current("target_long_term_credit")
        firms_with_needs = np.where(firm_target_credit > 0)[0]
        firms_with_needs_shuffled = np.random.choice(firms_with_needs, len(firms_with_needs), replace=False)
        for firm_id in firms_with_needs_shuffled:
            # Take a subset of all banks
            banks_subset = np.random.choice(
                range(banks.ts.current("n_banks")),
                min(
                    self.firms_max_number_of_banks_visiting,
                    banks.ts.current("n_banks"),
                ),
                replace=False,
            )

            # Iterate over all banks based on the offered interest rate
            for bank_id in banks_subset[np.argsort(banks_ir[banks_subset])]:
                if firm_target_credit[firm_id] == 0:
                    break

                # Supply
                total_credit_supply = (
                    banks.ts.current("equity")[bank_id] / banks.parameters["capital_adequacy_ratio"]["value"]
                    - banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )

                # Debt to equity
                debt_to_equity_restrictions = (
                    banks.parameters["firm_loans_debt_to_equity_ratio"]["value"]
                    * firms.ts.current("capital_inputs_stock_value")[firm_id]
                    - firms.ts.current("debt")[firm_id]
                    - new_credit_by_firm[firm_id]
                    + min(0.0, firms.ts.current("deposits")[firm_id])
                    + banks.ts.current("overdraft_rate_on_firm_deposits")[bank_id]
                    * min(0.0, firms.ts.current("deposits")[firm_id])
                    - firms.ts.current("interest_paid_on_loans")[firm_id]
                )

                # Return on equity
                return_on_equity_restrictions = (
                    firms.ts.current("capital_inputs_stock_value")[firm_id]
                    + firms.ts.current("deposits")[firm_id]
                    - firms.ts.current("debt")[firm_id]
                    - new_credit_by_firm[firm_id]
                    - firms.ts.current("expected_profits")[firm_id]
                    / banks.parameters["firm_loans_return_on_equity_ratio"]["value"]
                )

                # Return on assets
                return_on_assets_restrictions = (
                    np.inf
                    if firms.ts.current("expected_profits")[firm_id]
                    / (firms.ts.current("debt")[firm_id] + firms.ts.current("equity")[firm_id])
                    >= banks.parameters["firm_loans_return_on_assets_ratio"]["value"]
                    else 0.0
                )

                # Combine
                value_granted = max(
                    0.0,
                    min(
                        firm_target_credit[firm_id],
                        total_credit_supply,
                        max_supply_based_on_preferences[bank_id] - new_credit_by_bank[bank_id],
                        debt_to_equity_restrictions,
                        return_on_equity_restrictions,
                        return_on_assets_restrictions,
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
        new_credit_by_bank: np.ndarray,
        new_credit_by_household: np.ndarray,
        max_supply_based_on_preferences: np.ndarray,
    ) -> pd.DataFrame:
        # Data on new loans
        new_loan_types = []
        new_loan_value = []
        new_loan_maturity = []
        new_loan_interest_rate = []
        new_loan_bank_id = []
        new_loan_recipient_id = []

        # Loan maturity
        loan_maturity = banks.parameters["household_consumption_loan_maturity"]["value"]

        # Bank interest rates
        banks_ir = banks.ts.current("interest_rates_on_household_consumption_loans")

        # Iterate over households with financing needs
        household_target_credit = households.ts.current("target_consumption_loans")
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

                # Supply
                total_credit_supply = (
                    banks.ts.current("equity")[bank_id] / banks.parameters["capital_adequacy_ratio"]["value"]
                    - banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )

                # Loan to income
                loan_to_income_restrictions = (
                    banks.parameters["household_consumption_loans_loan_to_income_ratio"]["value"]
                    * 0.5
                    * (households.ts.prev("income")[household_id] + households.ts.current("income")[household_id])
                    - households.ts.current("debt")[household_id]
                    - new_credit_by_household[household_id]
                )

                # Combine
                value_granted = max(
                    0.0,
                    min(
                        household_target_credit[household_id],
                        total_credit_supply,
                        max_supply_based_on_preferences[bank_id] - new_credit_by_bank[bank_id],
                        loan_to_income_restrictions,
                    ),
                )

                # Record the new loans
                if value_granted > 0:
                    new_credit_by_bank[bank_id] += value_granted
                    new_credit_by_household[household_id] += value_granted
                    household_target_credit[household_id] -= value_granted
                    new_loan_types.append(LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN)
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
        max_supply_based_on_preferences: np.ndarray,
    ) -> pd.DataFrame:
        # Data on new loans
        new_loan_types = []
        new_loan_value = []
        new_loan_maturity = []
        new_loan_interest_rate = []
        new_loan_bank_id = []
        new_loan_recipient_id = []

        # Get bank interest rates
        banks_ir = banks.ts.current("interest_rates_on_mortgages")

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

                # Supply
                total_credit_supply = (
                    banks.ts.current("equity")[bank_id] / banks.parameters["capital_adequacy_ratio"]["value"]
                    - banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )

                # Loan to income
                loan_to_income_restrictions = (
                    banks.parameters["mortgage_loan_to_income_ratio"]["value"]
                    * 0.5
                    * (households.ts.prev("income")[household_id] + households.ts.current("income")[household_id])
                    - households.ts.current("debt")[household_id]
                    - new_credit_by_household[household_id]
                )

                # Loan to value
                loan_to_value_restrictions = (
                    banks.parameters["mortgage_loan_to_value_ratio"]["value"]
                    / (1 - banks.parameters["mortgage_loan_to_value_ratio"]["value"])
                    * households.ts.current("wealth_financial_assets")[household_id]
                )

                # Debt service to income
                debt_service_to_income_restrictions = (
                    banks.parameters["mortgage_debt_service_to_income_ratio"]["value"]
                    * households.ts.current("income")[household_id]
                    * (1 - (1 + banks_ir[bank_id]) ** (-banks.parameters["mortgage_maturity"]["value"]))
                    / banks_ir[bank_id]
                )

                # Combine
                value_granted = max(
                    0.0,
                    min(
                        household_target_credit[household_id],
                        total_credit_supply,
                        max_supply_based_on_preferences[bank_id] - new_credit_by_bank[bank_id],
                        loan_to_income_restrictions,
                        loan_to_value_restrictions,
                        debt_service_to_income_restrictions,
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
                    new_loan_maturity.append(banks.parameters["mortgage_maturity"]["value"])
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


class PolednaCreditMarketClearer(CreditMarketClearer):
    def clear(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        new_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_credit_by_firm = np.zeros(firms.ts.current("n_firms"))
        new_st_loans = self.clear_firm_loans(
            banks=banks,
            firms=firms,
            loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
        )
        new_lt_loans = np.zeros((3, banks.ts.current("n_banks"), firms.ts.current("n_firms")))
        new_cons_loans = np.zeros(
            (
                3,
                banks.ts.current("n_banks"),
                households.ts.current("n_households"),
            )
        )
        new_mort_loans = np.zeros(
            (
                3,
                banks.ts.current("n_banks"),
                households.ts.current("n_households"),
            )
        )

        return (
            new_st_loans,
            new_lt_loans,
            new_cons_loans,
            new_mort_loans,
        )

    def clear_firm_loans(
        self,
        banks: Banks,
        firms: Firms,
        loan_type: LoanTypes,
        new_credit_by_bank: np.ndarray,
        new_credit_by_firm: np.ndarray,  # noqa
    ) -> np.ndarray:
        # Select loan maturity
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            loan_maturity = banks.parameters["short_term_firm_loan_maturity"]["value"]
        else:
            loan_maturity = banks.parameters["long_term_firm_loan_maturity"]["value"]

        # Get bank interest rates
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            banks_ir = banks.ts.current("interest_rates_on_short_term_firm_loans")
        else:
            banks_ir = banks.ts.current("interest_rates_on_long_term_firm_loans")

        # Target credit
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            firm_target_credit = firms.ts.current("target_short_term_credit")
        else:
            firm_target_credit = firms.ts.current("target_long_term_credit")

        # For recording data
        new_loans = np.zeros((3, banks.ts.current("n_banks"), firm_target_credit.shape[0]))

        # Iterate over firms with financing needs
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
                min(
                    self.firms_max_number_of_banks_visiting,
                    banks.ts.current("n_banks"),
                ),
                replace=False,
            )

            # Iterate over all banks based on the offered interest rate
            for bank_id in banks_subset[np.argsort(banks_ir[banks_subset])]:
                if firm_target_credit[firm_id] == 0:
                    break
                bank_cap_req = (
                    banks.ts.current("equity")[bank_id] / banks.parameters["capital_adequacy_ratio"]["value"]
                    - (1 - 1.0 / loan_maturity) * banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )
                firm_risk_assessment = (
                    banks.parameters["firm_loans_debt_to_equity_ratio"]["value"]
                    * firms.ts.current("expected_capital_inputs_stock_value")[firm_id]
                    - (1 - 1.0 / loan_maturity) * firms.ts.current("debt")[firm_id]
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
                new_loans[0, bank_id, firm_id] += value_granted
                new_loans[1, bank_id, firm_id] += banks_ir[bank_id] * value_granted
                new_loans[2, bank_id, firm_id] += 1.0 / loan_maturity * value_granted

        return new_loans


class WaterBucketCreditMarketClearer(CreditMarketClearer):
    def clear(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Keeping track of new credit
        new_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_credit_by_firm = np.zeros(firms.ts.current("n_firms"))
        new_credit_by_household = np.zeros(households.ts.current("n_households"))

        # Banks may update their preferences for different types of loans, impacting their supply
        if self.consider_loan_type_fractions:
            max_car = np.maximum(
                0.0,
                banks.ts.current("equity") / banks.parameters.capital_adequacy_ratio
                - banks.ts.current("total_outstanding_loans")
                - new_credit_by_bank,
            )
            max_supply_based_on_preferences_firms = banks.ts.initial("new_loans_fraction_firms") * np.exp(
                -self.credit_supply_temperature * current_npl_firm_loans
            )
            max_supply_based_on_preferences_hh_cons = banks.ts.initial("new_loans_fraction_hh_cons") * np.exp(
                -self.credit_supply_temperature * current_npl_hh_cons_loans
            )
            max_supply_based_on_preferences_mortgages = banks.ts.initial("new_loans_fraction_mortgages") * np.exp(
                -self.credit_supply_temperature * current_npl_mortgages
            )
            current_sum = (
                max_supply_based_on_preferences_firms * max_car
                + max_supply_based_on_preferences_hh_cons * max_car
                + max_supply_based_on_preferences_mortgages * max_car
            )
            scale = np.divide(
                max_car,
                current_sum,
                out=np.zeros(max_car.shape),
                where=current_sum != 0.0,
            )
            max_supply_based_on_preferences_firms *= scale
            max_supply_based_on_preferences_hh_cons *= scale
            max_supply_based_on_preferences_mortgages *= scale
        else:
            max_supply_based_on_preferences_firms = np.full(banks.ts.current("n_banks"), np.inf)
            max_supply_based_on_preferences_hh_cons = np.full(banks.ts.current("n_banks"), np.inf)
            max_supply_based_on_preferences_mortgages = np.full(banks.ts.current("n_banks"), np.inf)

        # Firm loans
        new_st_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=max_supply_based_on_preferences_firms,
        )
        new_credit_by_firm += new_st_loans[0].sum(axis=0)
        new_credit_by_bank += new_st_loans[0].sum(axis=1)
        new_lt_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=max_supply_based_on_preferences_firms,
        )
        new_credit_by_bank += new_lt_loans[0].sum(axis=1)

        # Household loans
        new_cons_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=max_supply_based_on_preferences_hh_cons,
        )
        new_credit_by_household += new_cons_loans[0].sum(axis=0)
        new_credit_by_bank += new_cons_loans[0].sum(axis=1)
        new_mort_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.MORTGAGE,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=max_supply_based_on_preferences_mortgages,
        )

        return (
            new_st_loans,
            new_lt_loans,
            new_cons_loans,
            new_mort_loans,
        )

    def clear_loans(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
        loan_type: LoanTypes,
        new_credit_by_bank: np.ndarray,
        new_credit_by_firm: np.ndarray,
        new_credit_by_household: np.ndarray,
        max_supply_based_on_preferences: np.ndarray,
    ) -> np.ndarray:
        # Select loan properties and target credit
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            loan_maturity = banks.parameters.short_term_firm_loan_maturity
            banks_ir = banks.ts.current("interest_rates_on_short_term_firm_loans")
            target_credit = firms.ts.current("target_short_term_credit")
        elif loan_type == LoanTypes.FIRM_LONG_TERM_LOAN:
            loan_maturity = banks.parameters.long_term_firm_loan_maturity
            banks_ir = banks.ts.current("interest_rates_on_long_term_firm_loans")
            target_credit = firms.ts.current("target_long_term_credit")
        elif loan_type == LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN:
            loan_maturity = banks.parameters.household_consumption_loan_maturity
            banks_ir = banks.ts.current("interest_rates_on_household_consumption_loans")
            target_credit = households.ts.current("target_consumption_loans")
        elif loan_type == LoanTypes.MORTGAGE:
            loan_maturity = banks.parameters.mortgage_maturity
            banks_ir = banks.ts.current("interest_rates_on_mortgages")
            target_credit = households.ts.current("target_mortgage")
        else:
            raise ValueError("Unknown loan type", loan_type)

        # For recording data
        new_loans = np.zeros((3, banks.ts.current("n_banks"), target_credit.shape[0]))

        # Select agents wanting credit and priorities
        agents_with_demand = np.where(target_credit > 0)[0]
        debtor_priorities = self.get_debtor_priorities(n_agents=agents_with_demand.shape[0])

        # Determine capacities
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN or loan_type == LoanTypes.FIRM_LONG_TERM_LOAN:
            debt_to_equity_restrictions = (
                banks.parameters.firm_loans_debt_to_equity_ratio
                * firms.ts.current("capital_inputs_stock_value")[agents_with_demand]
                - firms.ts.current("debt")[agents_with_demand]
                - new_credit_by_firm[agents_with_demand]
                + np.minimum(0, firms.ts.current("deposits")[agents_with_demand])
            )
            return_on_equity_restrictions = (
                firms.ts.current("capital_inputs_stock_value")[agents_with_demand]
                + firms.ts.current("deposits")[agents_with_demand]
                - firms.ts.current("debt")[agents_with_demand]
                - new_credit_by_firm[agents_with_demand]
                - firms.ts.current("expected_profits")[agents_with_demand]
                / banks.parameters.firm_loans_return_on_equity_ratio
            )
            return_on_assets_restrictions = np.zeros(agents_with_demand.shape)
            return_on_assets_restrictions[
                firms.ts.current("expected_profits")[agents_with_demand]
                / np.maximum(
                    1.0,
                    firms.ts.current("capital_inputs_stock_value")[agents_with_demand],
                )
                >= banks.parameters.firm_loans_return_on_assets_ratio
            ] = np.inf
            credit_restrictions = np.minimum(
                np.minimum(debt_to_equity_restrictions, return_on_equity_restrictions),
                return_on_assets_restrictions,
            )
        elif loan_type == LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN:
            loan_to_income_restrictions = (
                banks.parameters.household_consumption_loans_loan_to_income_ratio
                * 0.5
                * (
                    households.ts.prev("income")[agents_with_demand]
                    + households.ts.current("income")[agents_with_demand]
                )
                - households.ts.current("debt")[agents_with_demand]
                - new_credit_by_household[agents_with_demand]
            )
            credit_restrictions = loan_to_income_restrictions
        elif loan_type == LoanTypes.MORTGAGE:
            loan_to_income_restrictions = (
                banks.parameters.mortgage_loan_to_income_ratio
                * 0.5
                * (
                    households.ts.prev("income")[agents_with_demand]
                    + households.ts.current("income")[agents_with_demand]
                )
                - households.ts.current("debt")[agents_with_demand]
                - new_credit_by_household[agents_with_demand]
            )
            loan_to_value_restrictions = (
                banks.parameters.mortgage_loan_to_value_ratio
                / (1 - banks.parameters.mortgage_loan_to_value_ratio)
                * households.ts.current("wealth_financial_assets")[agents_with_demand]
            )

            # BTL OR OO?? need to check.
            """
            # Debt service to income
            debt_service_to_income_restrictions = (
                    banks.parameters["mortgage_debt_service_to_income_ratio"]["value"]
                    * households.ts.current("income")[agents_with_demand]
                    * (1 - (1 + banks_ir[bank_id]) ** (-banks.parameters["mortgage_maturity"]["value"]))
                    / banks_ir[bank_id]
            )
            """
            credit_restrictions = np.minimum(loan_to_income_restrictions, loan_to_value_restrictions)
        else:
            raise ValueError("Unknown loan type", loan_type)
        capacities = np.maximum(
            0.0,
            np.minimum(target_credit[agents_with_demand], credit_restrictions),
        )
        capacities_sum = capacities.sum()
        if capacities_sum == 0.0:
            return new_loans
        capacities_weights = capacities / capacities_sum

        # Determine total supply and priorities
        supply = np.maximum(
            0.0,
            np.minimum(
                banks.ts.current("equity") / banks.parameters.capital_adequacy_ratio
                - banks.ts.current("total_outstanding_loans")
                - new_credit_by_bank,
                max_supply_based_on_preferences - new_credit_by_bank,
            ),
        )
        supply_sum = supply.sum()
        if supply_sum == 0.0:
            return new_loans
        supply_weights = supply / supply_sum
        if self.get_creditor_priorities_deterministic:
            creditor_priorities = self.get_creditor_priorities_deterministic(
                self.interest_rates_selection_temperature,
                interest_rates=banks_ir,
            )
        else:
            creditor_priorities = self.get_creditor_priorities_deterministic(
                self.interest_rates_selection_temperature,
                interest_rates=banks_ir,
            )

        # Hand out loans
        if supply_sum >= capacities_sum:
            granted_loans_by_banks = self.fill_buckets(
                capacities=supply,
                fill_amount=capacities_sum,
                priorities=creditor_priorities,
                minimum_fill=self.creditor_minimum_fill,
            )
            new_loans[0, :, agents_with_demand] = np.matmul(
                granted_loans_by_banks[np.newaxis].T,
                capacities_weights[np.newaxis],
            )[np.newaxis].T[:, 0, :]
        else:
            received_loans_by_debtors = self.fill_buckets(
                capacities=capacities,
                fill_amount=supply_sum,
                priorities=debtor_priorities,
                minimum_fill=self.debtor_minimum_fill,
            )
            new_loans[0, :, agents_with_demand] = np.matmul(
                received_loans_by_debtors[np.newaxis].T,
                supply_weights[np.newaxis],
            )[np.newaxis].T[:, 0, :]
        new_loans[1, :, agents_with_demand] = banks_ir[:, np.newaxis] * new_loans[0, :, agents_with_demand]
        new_loans[2, :, agents_with_demand] = 1.0 / loan_maturity * new_loans[0, :, agents_with_demand]

        return new_loans

    @staticmethod
    @njit
    def get_debtor_priorities(n_agents: int) -> np.ndarray:
        return np.random.choice(n_agents, n_agents, replace=False)

    @staticmethod
    @njit
    def get_creditor_priorities_deterministic(
        interest_rates_selection_temperature: float,
        interest_rates: np.ndarray,
    ) -> np.ndarray:
        distribution = np.exp(-interest_rates_selection_temperature * interest_rates)
        return np.argsort(distribution)[::-1]

    @staticmethod
    def get_creditor_priorities_stochastic(
        interest_rates_selection_temperature: float,
        interest_rates: np.ndarray,
    ) -> np.ndarray:
        distribution = np.exp(-interest_rates_selection_temperature * interest_rates)
        return np.random.choice(
            len(distribution),
            len(distribution),
            replace=False,
            p=distribution / np.sum(distribution),
        )

    @staticmethod
    def invert_permutation(p: np.ndarray) -> np.ndarray:
        s = np.empty_like(p)
        s[p] = np.arange(p.size)
        return s

    def fill_buckets(
        self,
        capacities: np.ndarray,
        fill_amount: float,
        priorities: np.ndarray,
        minimum_fill: float,
    ) -> np.ndarray:
        if np.sum(capacities) == np.sum(capacities) + 1:
            return np.full_like(capacities, fill_amount / len(capacities))
        if np.sum(capacities) == 0:
            return np.zeros(capacities.shape)
        capacities_sorted = capacities[priorities]
        filled_capacities = np.zeros(capacities_sorted.shape)
        if minimum_fill > 0.0:
            filled_capacities += np.minimum(
                capacities_sorted,
                capacities_sorted / np.sum(capacities_sorted) * minimum_fill * fill_amount,
            )
        filled_ind = np.where(
            (capacities_sorted - filled_capacities).cumsum() < fill_amount - np.sum(filled_capacities)
        )[0]
        filled_capacities[filled_ind] = capacities_sorted[filled_ind]
        if len(filled_ind) < len(filled_capacities):
            filled_capacities[len(filled_ind)] += fill_amount - np.sum(filled_capacities)
            filled_capacities[len(filled_ind)] = min(
                filled_capacities[len(filled_ind)],
                capacities_sorted[len(filled_ind)],
            )
        return filled_capacities[self.invert_permutation(priorities)]

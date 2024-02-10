from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from macro_data import SyntheticCreditMarket
from pathlib import Path
from typing import Any, TYPE_CHECKING

from macromodel.configurations import CreditMarketConfiguration
from macromodel.credit_market.credit_market_ts import create_credit_market_timeseries
from macromodel.credit_market.types_of_loans import LoanTypes
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import get_functions, functions_from_model
from macromodel.util.property_mapping import map_to_enum

if TYPE_CHECKING:
    from macromodel.firms.firms import Firms
    from macromodel.banks.banks import Banks
    from macromodel.households.households import Households


class CreditMarket:
    def __init__(
        self,
        country_name: str,
        functions: dict[str, Any],
        ts: TimeSeries,
        loan_data: pd.DataFrame,
    ):
        self.country_name = country_name
        self.functions = functions
        self.ts = ts
        self.loan_data = loan_data

    @classmethod
    def from_pickled_market(
        cls,
        synthetic_credit_market: SyntheticCreditMarket,
        credit_market_configuration: CreditMarketConfiguration,
        country_name: str,
    ) -> "CreditMarket":
        functions = functions_from_model(
            credit_market_configuration.functions,
            loc="inet_macromodel.credit_market",
        )

        loan_data = synthetic_credit_market.credit_market_data.astype(float)
        loan_data.rename_axis("Loans", inplace=True)

        loan_data["loan_type"] = np.array(map_to_enum(loan_data["loan_type"].values, LoanTypes))
        loan_data["loan_bank_id"] = loan_data["loan_bank_id"].astype(int)
        loan_data["loan_recipient_id"] = loan_data["loan_recipient_id"].astype(int)

        ts = create_credit_market_timeseries(loan_data)

        return cls(
            country_name,
            functions,
            ts,
            loan_data,
        )

    @classmethod
    def from_data(
        cls,
        country_name: str,
        data: pd.DataFrame,
        config: dict[str, Any],
    ) -> "CreditMarket":
        # Get corresponding functions and parameters
        functions = get_functions(
            config["functions"],
            loc="inet_macromodel.credit_market",
            func_dir=Path(__file__).parent / "func",
        )
        # Recording the loan_data of all loans
        loan_data = data.copy()
        loan_data["loan_type"] = np.array(map_to_enum(loan_data["loan_type"].values, LoanTypes))
        loan_data["loan_bank_id"] = loan_data["loan_bank_id"].astype(int)
        loan_data["loan_recipient_id"] = loan_data["loan_recipient_id"].astype(int)

        # Create the corresponding time series object
        ts = create_credit_market_timeseries(loan_data)

        return cls(
            country_name,
            functions,
            ts,
            loan_data,
        )

    def clear(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
    ) -> None:
        # Clear the credit market
        new_loans = self.functions["clearing"].clear(
            banks=banks,
            firms=firms,
            households=households,
        )

        # Record the new loans
        if len(new_loans) > 0:
            self.loan_data = pd.concat((self.loan_data, new_loans), axis=0).reset_index(drop=True)

        # Calculate new short-term loans granted by firm
        new_firm_short_term_loans = new_loans.loc[new_loans["loan_type"] == LoanTypes.FIRM_SHORT_TERM_LOAN]
        new_firm_short_term_loans_grouped = new_firm_short_term_loans.groupby("loan_recipient_id")[
            "loan_value_initial"
        ].sum()
        received_short_term_credit = np.zeros(firms.ts.current("n_firms"))
        received_short_term_credit[new_firm_short_term_loans_grouped.index] = new_firm_short_term_loans_grouped.values
        firms.ts.received_short_term_credit.append(received_short_term_credit)
        firms.ts.total_received_short_term_credit.append([firms.ts.current("received_short_term_credit").sum()])

        # Calculate new long-term loans granted by firm
        new_firm_long_term_loans = new_loans.loc[new_loans["loan_type"] == LoanTypes.FIRM_LONG_TERM_LOAN]
        new_firm_long_term_loans_grouped = new_firm_long_term_loans.groupby("loan_recipient_id")[
            "loan_value_initial"
        ].sum()
        received_long_term_credit = np.zeros(firms.ts.current("n_firms"))
        received_long_term_credit[new_firm_long_term_loans_grouped.index] = new_firm_long_term_loans_grouped.values
        firms.ts.received_long_term_credit.append(received_long_term_credit)
        firms.ts.total_received_long_term_credit.append([firms.ts.current("received_long_term_credit").sum()])

        # Total firm credit
        firms.ts.received_credit.append(
            firms.ts.current("received_short_term_credit") + firms.ts.current("received_long_term_credit")
        )

        # Calculate new payday loans granted by household
        new_hh_payday_loans = new_loans.loc[new_loans["loan_type"] == LoanTypes.HOUSEHOLD_PAYDAY_LOAN]
        new_hh_payday_loans_grouped = new_hh_payday_loans.groupby("loan_recipient_id")["loan_value_initial"].sum()
        received_hh_payday_loans_credit = np.zeros(households.ts.current("n_households"))
        received_hh_payday_loans_credit[new_hh_payday_loans_grouped.index] = new_hh_payday_loans_grouped.values
        households.ts.received_payday_loans.append(received_hh_payday_loans_credit)
        households.ts.total_received_payday_loans.append([households.ts.current("received_payday_loans").sum()])

        # Calculate new consumption expansion granted by household
        new_hh_ce_loans = new_loans.loc[new_loans["loan_type"] == LoanTypes.HOUSEHOLD_CONSUMPTION_EXPANSION_LOAN]
        new_hh_ce_loans_grouped = new_hh_ce_loans.groupby("loan_recipient_id")["loan_value_initial"].sum()
        received_hh_ce_loans_credit = np.zeros(households.ts.current("n_households"))
        received_hh_ce_loans_credit[new_hh_ce_loans_grouped.index] = new_hh_ce_loans_grouped.values
        households.ts.received_consumption_expansion_loans.append(received_hh_ce_loans_credit)
        households.ts.total_received_consumption_expansion_loans.append(
            [households.ts.current("received_consumption_expansion_loans").sum()]
        )

        # Calculate new mortgages granted by household
        new_mortgages = new_loans.loc[new_loans["loan_type"] == LoanTypes.MORTGAGE]
        new_mortgages_grouped = new_mortgages.groupby("loan_recipient_id")["loan_value_initial"].sum()
        received_mortgages_credit = np.zeros(households.ts.current("n_households"))
        received_mortgages_credit[new_mortgages_grouped.index] = new_mortgages_grouped.values
        households.ts.received_mortgages.append(received_mortgages_credit)
        households.ts.total_received_mortgages.append([households.ts.current("received_mortgages").sum()])

        # Update aggregates
        self.ts.total_newly_loans_granted_firms_short_term.append(
            [firms.ts.current("received_short_term_credit").sum()]
        )
        self.ts.total_newly_loans_granted_firms_long_term.append([firms.ts.current("received_long_term_credit").sum()])
        self.ts.total_newly_loans_granted_households_payday.append(
            [households.ts.current("received_payday_loans").sum()]
        )
        self.ts.total_newly_loans_granted_households_consumption_expansion.append(
            [households.ts.current("received_consumption_expansion_loans").sum()]
        )
        self.ts.total_newly_loans_granted_mortgages.append([households.ts.current("received_mortgages").sum()])

    def pay_firm_installments(self, n_firms: int) -> np.ndarray:
        firm_loans = self.loan_data.loc[
            self.loan_data["loan_type"].isin([LoanTypes.FIRM_SHORT_TERM_LOAN, LoanTypes.FIRM_LONG_TERM_LOAN])
        ].copy()
        firm_loans.loc[:, "loan_installment"] = (
            firm_loans.loc[:, "loan_value_initial"] / firm_loans.loc[:, "loan_maturity"]
        )
        installments_grouped = firm_loans.groupby("loan_recipient_id")["loan_installment"].sum()
        self.loan_data.loc[firm_loans.index, "loan_value"] -= firm_loans["loan_installment"]
        firm_installments = np.zeros(n_firms)
        firm_installments[installments_grouped.index] = installments_grouped
        return firm_installments

    def pay_household_installments(self, n_households: int) -> np.ndarray:
        household_loans = self.loan_data.loc[
            self.loan_data["loan_type"].isin([LoanTypes.FIRM_SHORT_TERM_LOAN, LoanTypes.FIRM_LONG_TERM_LOAN])
        ].copy()
        household_loans.loc[:, "loan_installment"] = (
            household_loans.loc[:, "loan_value_initial"] / household_loans.loc[:, "loan_maturity"]
        )
        installments_grouped = household_loans.groupby("loan_recipient_id")["loan_installment"].sum()
        self.loan_data.loc[household_loans.index, "loan_value"] -= household_loans["loan_installment"]
        household_installments = np.zeros(n_households)
        household_installments[installments_grouped.index] = installments_grouped
        return household_installments

    def remove_repaid_loans(self) -> None:
        self.loan_data = self.loan_data.loc[~np.isclose(self.loan_data["loan_value"], 0.0, atol=1e-2)]

    def compute_aggregates(self) -> None:
        self.ts.total_outstanding_loans_granted_firms_short_term.append(
            [
                self.loan_data.loc[
                    self.loan_data["loan_type"] == LoanTypes.FIRM_SHORT_TERM_LOAN,
                    "loan_value",
                ].sum()
            ]
        )
        self.ts.total_outstanding_loans_granted_firms_long_term.append(
            [
                self.loan_data.loc[
                    self.loan_data["loan_type"] == LoanTypes.FIRM_LONG_TERM_LOAN,
                    "loan_value",
                ].sum()
            ]
        )
        self.ts.total_outstanding_loans_granted_households_payday.append(
            [
                self.loan_data.loc[
                    self.loan_data["loan_type"] == LoanTypes.HOUSEHOLD_PAYDAY_LOAN,
                    "loan_value",
                ].sum()
            ]
        )
        self.ts.total_outstanding_loans_granted_households_consumption_expansion.append(
            [
                self.loan_data.loc[
                    self.loan_data["loan_type"] == LoanTypes.HOUSEHOLD_CONSUMPTION_EXPANSION_LOAN,
                    "loan_value",
                ].sum()
            ]
        )
        self.ts.total_outstanding_loans_granted_mortgages.append(
            [self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.MORTGAGE, "loan_value"].sum()]
        )

    def compute_outstanding_short_term_loans_by_firm(self, n_firms: int) -> np.ndarray:
        new_firm_short_term_loans = self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.FIRM_SHORT_TERM_LOAN]
        new_firm_short_term_loans_grouped = new_firm_short_term_loans.groupby("loan_recipient_id")["loan_value"].sum()
        received_short_term_credit = np.zeros(n_firms)
        received_short_term_credit[new_firm_short_term_loans_grouped.index] = new_firm_short_term_loans_grouped.values
        return received_short_term_credit

    def compute_outstanding_long_term_loans_by_firm(self, n_firms: int) -> np.ndarray:
        new_firm_long_term_loans = self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.FIRM_LONG_TERM_LOAN]
        new_firm_long_term_loans_grouped = new_firm_long_term_loans.groupby("loan_recipient_id")["loan_value"].sum()
        received_long_term_credit = np.zeros(n_firms)
        received_long_term_credit[new_firm_long_term_loans_grouped.index] = new_firm_long_term_loans_grouped.values
        return received_long_term_credit

    def compute_outstanding_payday_loans_by_household(self, n_households: int) -> np.ndarray:
        new_hh_payday_loans = self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.HOUSEHOLD_PAYDAY_LOAN]
        new_hh_payday_grouped = new_hh_payday_loans.groupby("loan_recipient_id")["loan_value"].sum()
        received_hh_payday_loans = np.zeros(n_households)
        received_hh_payday_loans[new_hh_payday_grouped.index] = new_hh_payday_grouped.values
        return received_hh_payday_loans

    def compute_outstanding_consumption_expansion_loans_by_household(self, n_households: int) -> np.ndarray:
        new_hh_ce_loans = self.loan_data.loc[
            self.loan_data["loan_type"] == LoanTypes.HOUSEHOLD_CONSUMPTION_EXPANSION_LOAN
        ]
        new_hh_ce_grouped = new_hh_ce_loans.groupby("loan_recipient_id")["loan_value"].sum()
        received_hh_ce_loans = np.zeros(n_households)
        received_hh_ce_loans[new_hh_ce_grouped.index] = new_hh_ce_grouped.values
        return received_hh_ce_loans

    def compute_outstanding_mortgages_by_household(self, n_households: int) -> np.ndarray:
        new_hh_mortgages = self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.MORTGAGE]
        new_hh_mortgages_grouped = new_hh_mortgages.groupby("loan_recipient_id")["loan_value"].sum()
        received_hh_mortgage_loans = np.zeros(n_households)
        received_hh_mortgage_loans[new_hh_mortgages_grouped.index] = new_hh_mortgages_grouped.values
        return received_hh_mortgage_loans

    def compute_outstanding_loans_by_bank(self, n_banks: int) -> np.ndarray:
        all_loans_grouped = self.loan_data.groupby("loan_bank_id")["loan_value"].sum()
        loans_by_bank = np.zeros(n_banks)
        loans_by_bank[all_loans_grouped.index] = all_loans_grouped.values
        return loans_by_bank

    def compute_outstanding_short_term_firm_loans_by_bank(self, n_banks: int) -> np.ndarray:
        firm_loans = self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.FIRM_SHORT_TERM_LOAN]
        all_loans_grouped = firm_loans.groupby("loan_bank_id")["loan_value"].sum()
        loans_by_bank = np.zeros(n_banks)
        loans_by_bank[all_loans_grouped.index] = all_loans_grouped.values
        return loans_by_bank

    def compute_outstanding_long_term_firm_loans_by_bank(self, n_banks: int) -> np.ndarray:
        firm_loans = self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.FIRM_LONG_TERM_LOAN]
        all_loans_grouped = firm_loans.groupby("loan_bank_id")["loan_value"].sum()
        loans_by_bank = np.zeros(n_banks)
        loans_by_bank[all_loans_grouped.index] = all_loans_grouped.values
        return loans_by_bank

    def compute_outstanding_household_payday_loans_by_bank(self, n_banks: int) -> np.ndarray:
        hh_loans = self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.HOUSEHOLD_PAYDAY_LOAN]
        all_loans_grouped = hh_loans.groupby("loan_bank_id")["loan_value"].sum()
        loans_by_bank = np.zeros(n_banks)
        loans_by_bank[all_loans_grouped.index] = all_loans_grouped.values
        return loans_by_bank

    def compute_outstanding_household_ce_loans_by_bank(self, n_banks: int) -> np.ndarray:
        hh_loans = self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.HOUSEHOLD_CONSUMPTION_EXPANSION_LOAN]
        all_loans_grouped = hh_loans.groupby("loan_bank_id")["loan_value"].sum()
        loans_by_bank = np.zeros(n_banks)
        loans_by_bank[all_loans_grouped.index] = all_loans_grouped.values
        return loans_by_bank

    def compute_outstanding_mortgages_by_bank(self, n_banks: int) -> np.ndarray:
        hh_loans = self.loan_data.loc[self.loan_data["loan_type"] == LoanTypes.MORTGAGE]
        all_loans_grouped = hh_loans.groupby("loan_bank_id")["loan_value"].sum()
        loans_by_bank = np.zeros(n_banks)
        loans_by_bank[all_loans_grouped.index] = all_loans_grouped.values
        return loans_by_bank

    def compute_interest_paid_by_firm(self, n_firms: int) -> np.ndarray:
        firm_loans = self.loan_data.loc[
            self.loan_data["loan_type"].isin([LoanTypes.FIRM_SHORT_TERM_LOAN, LoanTypes.FIRM_LONG_TERM_LOAN])
        ]
        firm_loans_grouped = firm_loans.groupby("loan_recipient_id").apply(
            lambda x: (x.loan_value_initial * x.loan_interest_rate).sum()
        )
        if len(firm_loans_grouped) == 0:
            return np.zeros(n_firms)
        interest_paid = np.zeros(n_firms)
        interest_paid[firm_loans_grouped.index.values.astype(int)] = firm_loans_grouped.values
        return interest_paid

    def compute_interest_paid_by_household(self, n_households: int) -> np.ndarray:
        household_loans = self.loan_data.loc[
            self.loan_data["loan_type"].isin(
                [
                    LoanTypes.HOUSEHOLD_PAYDAY_LOAN,
                    LoanTypes.HOUSEHOLD_CONSUMPTION_EXPANSION_LOAN,
                    LoanTypes.MORTGAGE,
                ]
            )
        ]
        household_loans_grouped = household_loans.groupby("loan_recipient_id").apply(
            lambda x: (x.loan_value_initial * x.loan_interest_rate).sum()
        )
        if len(household_loans_grouped) == 0:
            return np.zeros(n_households)
        interest_paid = np.zeros(n_households)
        interest_paid[household_loans_grouped.index] = household_loans_grouped.values
        return interest_paid

    def compute_interest_received_by_bank(self, n_banks: int) -> np.ndarray:
        loans_grouped = self.loan_data.groupby("loan_bank_id").apply(
            lambda x: (x.loan_value_initial * x.loan_interest_rate).sum()
        )
        interest_received = np.zeros(n_banks)
        interest_received[loans_grouped.index] = loans_grouped.values
        return interest_received

    def remove_loans_to_firm(self, firm_id: int) -> None:
        firm_loans = self.loan_data["loan_type"].isin([LoanTypes.FIRM_SHORT_TERM_LOAN, LoanTypes.FIRM_LONG_TERM_LOAN])
        self.loan_data = self.loan_data.loc[~np.logical_and(firm_loans, self.loan_data["loan_recipient_id"] == firm_id)]

    def remove_loans_to_households(self, household_id: int | list[int]) -> None:
        household_loans = self.loan_data["loan_type"].isin(
            [
                LoanTypes.HOUSEHOLD_PAYDAY_LOAN,
                LoanTypes.HOUSEHOLD_CONSUMPTION_EXPANSION_LOAN,
                LoanTypes.MORTGAGE,
            ]
        )
        if isinstance(household_id, int):
            self.loan_data = self.loan_data.loc[
                ~np.logical_and(household_loans, self.loan_data["loan_recipient_id"] == household_id)
            ]
        else:
            self.loan_data = self.loan_data.loc[
                ~np.logical_and(household_loans, self.loan_data["loan_recipient_id"].isin(household_id))
            ]

    def remove_loans_by_bank(self, bank_id: int) -> None:
        self.loan_data = self.loan_data.loc[self.loan_data["loan_bank_id"] != bank_id]

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("credit_market", group)

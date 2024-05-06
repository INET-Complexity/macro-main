from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LongtermLoans:
    principal: np.ndarray
    interest: np.ndarray
    installments: np.ndarray

    @classmethod
    def from_agent_data(
        cls, bank_data: pd.DataFrame, firm_data: pd.DataFrame, firm_loan_maturity: int = 60
    ) -> "LongtermLoans":
        firm_debt = firm_data["Debt"].values
        firms_corresponding_bank = firm_data["Corresponding Bank ID"].values

        principal = np.zeros((bank_data.shape[0], firm_debt.shape[0]))
        interest = np.zeros((bank_data.shape[0], firm_debt.shape[0]))
        discount = np.zeros((bank_data.shape[0], firm_debt.shape[0]))

        for firm_id in range(firm_debt.shape[0]):
            principal[firms_corresponding_bank[firm_id], firm_id] = firm_debt[firm_id]
            interest[firms_corresponding_bank[firm_id], firm_id] = (
                bank_data["Long-Term Interest Rates on Firm Loans"].values[firms_corresponding_bank[firm_id]]
                * firm_debt[firm_id]
            )
            discount[firms_corresponding_bank[firm_id], firm_id] = 1.0 / firm_loan_maturity * firm_debt[firm_id]

        return cls(principal, interest, discount)


@dataclass
class ShorttermLoans:
    principal: np.ndarray
    interest: np.ndarray
    installments: np.ndarray

    @classmethod
    def from_agent_data(
        cls, bank_data: pd.DataFrame, firm_data: pd.DataFrame, firm_loan_maturity: int = 60
    ) -> "ShorttermLoans":
        firm_debt = firm_data["Debt"].values

        principal = np.zeros((bank_data.shape[0], firm_debt.shape[0]))
        interest = np.zeros((bank_data.shape[0], firm_debt.shape[0]))
        discount = np.zeros((bank_data.shape[0], firm_debt.shape[0]))

        return cls(principal, interest, discount)


@dataclass
class ConsumptionExpansionLoans:
    principal: np.ndarray
    interest: np.ndarray
    installments: np.ndarray

    @classmethod
    def from_agent_data(
        cls, bank_data: pd.DataFrame, household_data: pd.DataFrame, consumption_loan_maturity: int = 1
    ) -> "ConsumptionExpansionLoans":
        household_other_debt = household_data["Outstanding Balance of other Non-Mortgage Loans"].values
        households_corresponding_bank = household_data["Corresponding Bank ID"].values

        principal = np.zeros((bank_data.shape[0], household_other_debt.shape[0]))
        interest = np.zeros((bank_data.shape[0], household_other_debt.shape[0]))
        discount = np.zeros((bank_data.shape[0], household_other_debt.shape[0]))

        for household_id in range(household_other_debt.shape[0]):
            principal[households_corresponding_bank[household_id], household_id] = household_other_debt[household_id]
            interest[households_corresponding_bank[household_id], household_id] = (
                bank_data["Interest Rates on Household Consumption Loans"].values[
                    households_corresponding_bank[household_id]
                ]
                * household_other_debt[household_id]
            )
            discount[households_corresponding_bank[household_id], household_id] = (
                1.0 / consumption_loan_maturity * household_other_debt[household_id]
            )

        return cls(principal, interest, discount)


@dataclass
class PaydayLoans:
    principal: np.ndarray
    interest: np.ndarray
    installments: np.ndarray

    @classmethod
    def from_agent_data(
        cls, bank_data: pd.DataFrame, household_data: pd.DataFrame, payday_loan_maturity: int = 1
    ) -> "PaydayLoans":
        principal = np.zeros((bank_data.shape[0], household_data.shape[0]))
        interest = np.zeros((bank_data.shape[0], household_data.shape[0]))
        discount = np.zeros((bank_data.shape[0], household_data.shape[0]))

        return cls(principal, interest, discount)


@dataclass
class MortgageLoans:
    principal: np.ndarray
    interest: np.ndarray
    installments: np.ndarray

    @classmethod
    def from_agent_data(cls, bank_data: pd.DataFrame, household_data: pd.DataFrame, mortgage_maturity: int = 120):
        household_mortgage_debt = household_data["Outstanding Balance of HMR Mortgages"].values
        households_corresponding_bank = household_data["Corresponding Bank ID"].values

        principal = np.zeros((bank_data.shape[0], household_mortgage_debt.shape[0]))
        interest = np.zeros((bank_data.shape[0], household_mortgage_debt.shape[0]))
        discount = np.zeros((bank_data.shape[0], household_mortgage_debt.shape[0]))

        for household_id in range(household_mortgage_debt.shape[0]):
            principal[households_corresponding_bank[household_id], household_id] = household_mortgage_debt[household_id]
            interest[households_corresponding_bank[household_id], household_id] = (
                bank_data["Interest Rates on Mortgages"].values[households_corresponding_bank[household_id]]
                * household_mortgage_debt[household_id]
            )
            discount[households_corresponding_bank[household_id], household_id] = (
                1.0 / mortgage_maturity * household_mortgage_debt[household_id]
            )

        return cls(principal, interest, discount)

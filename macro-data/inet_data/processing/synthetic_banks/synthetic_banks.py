from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SyntheticBanks(ABC):
    @abstractmethod
    def __init__(self, country_name: str, year: int, number_of_banks: int, bank_data: pd.DataFrame):
        # Parameters
        self.country_name = country_name
        self.year = year
        self.number_of_banks = number_of_banks

        # Bank data
        self.bank_data = bank_data

    def create_agents(self, bank_equity: float) -> None:
        # Set initial bank equity
        self.set_bank_equity(bank_equity=bank_equity)

    def set_initial_bank_fields(
        self,
        firm_deposits: np.ndarray,
        firm_debt: np.ndarray,
        household_deposits: np.ndarray,
        household_mortgage_debt: np.ndarray,
        household_other_debt: np.ndarray,
        cb_policy_rate: float,
        tau_bank: float,
        bank_markup_interest_rate_short_term_firm_loans: float,
        bank_markup_interest_rate_long_term_firm_loans: float,
        bank_markup_interest_rate_household_payday_loans: float,
        bank_markup_interest_rate_household_consumption_loans: float,
        bank_markup_interest_rate_mortgages: float,
        bank_markup_interest_rate_overdraft_firm: float,
        bank_markup_interest_rate_overdraft_household: float,
    ) -> None:
        # Set deposits from firms
        self.set_deposits_from_firms(firm_deposits=firm_deposits)

        # Set deposits from households
        self.set_deposits_from_households(household_deposits=household_deposits)

        # Set loans to firms
        self.set_loans_to_firms(firm_debt=firm_debt)

        # Set loans to households
        self.set_loans_to_households(
            household_mortgage_debt=household_mortgage_debt,
            household_other_debt=household_other_debt,
        )

        # Set initial bank deposits
        self.set_bank_deposits(
            firm_deposits=firm_deposits,
            firm_debt=firm_debt,
            household_deposits=household_deposits,
            household_debt=household_mortgage_debt + household_other_debt,
        )

        # Set initial interest rates
        self.set_initial_interest_rates(
            central_bank_policy_rate=cb_policy_rate,
            bank_markup_interest_rate_short_term_firm_loans=bank_markup_interest_rate_short_term_firm_loans,
            bank_markup_interest_rate_long_term_firm_loans=bank_markup_interest_rate_long_term_firm_loans,
            bank_markup_interest_rate_household_payday_loans=bank_markup_interest_rate_household_payday_loans,
            bank_markup_interest_rate_household_consumption_loans=bank_markup_interest_rate_household_consumption_loans,
            bank_markup_interest_rate_mortgages=bank_markup_interest_rate_mortgages,
            bank_markup_interest_rate_overdraft_firm=bank_markup_interest_rate_overdraft_firm,
            bank_markup_interest_rate_overdraft_household=bank_markup_interest_rate_overdraft_household,
        )

        # Set initial bank profits
        self.set_interest_received_from_loans()
        self.set_interest_received_from_deposits(central_bank_policy_rate=cb_policy_rate)
        self.set_profits()
        self.set_corporate_taxes_paid(tau_bank=tau_bank)

        # Set initial bank liabilities
        self.set_liability()

        # Compute the initial market share
        self.set_market_share()

    @abstractmethod
    def set_deposits_from_firms(self, firm_deposits: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_deposits_from_households(self, household_deposits: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_loans_to_firms(self, firm_debt: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_loans_to_households(
        self,
        household_mortgage_debt: np.ndarray,
        household_other_debt: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def set_bank_equity(self, bank_equity: float) -> None:
        pass

    @abstractmethod
    def set_bank_deposits(
        self,
        firm_deposits: np.ndarray,
        household_deposits: np.ndarray,
        firm_debt: np.ndarray,
        household_debt: np.ndarray,
    ) -> None:
        pass

    def set_initial_interest_rates(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_short_term_firm_loans: float,
        bank_markup_interest_rate_long_term_firm_loans: float,
        bank_markup_interest_rate_household_payday_loans: float,
        bank_markup_interest_rate_household_consumption_loans: float,
        bank_markup_interest_rate_mortgages: float,
        bank_markup_interest_rate_overdraft_firm: float,
        bank_markup_interest_rate_overdraft_household: float,
    ) -> None:
        # Short-term interest rates for firm loans
        self.bank_data["Short-Term Interest Rates on Firm Loans"] = (
            central_bank_policy_rate + bank_markup_interest_rate_short_term_firm_loans
        )

        # Long-term interest rates for firm loans
        self.bank_data["Long-Term Interest Rates on Firm Loans"] = (
            central_bank_policy_rate + bank_markup_interest_rate_long_term_firm_loans
        )

        # Interest rates for household payday loans
        self.bank_data["Interest Rates on Household Payday Loans"] = (
            central_bank_policy_rate + bank_markup_interest_rate_household_payday_loans
        )

        # Interest rates for household consumption loans
        self.bank_data["Interest Rates on Household Consumption Loans"] = (
            central_bank_policy_rate + bank_markup_interest_rate_household_consumption_loans
        )

        # Interest rates for mortgages
        self.bank_data["Interest Rates on Mortgages"] = central_bank_policy_rate + bank_markup_interest_rate_mortgages

        # Interest rates on firm deposits
        self.bank_data["Interest Rates on Firm Deposits"] = central_bank_policy_rate

        # Overdraft rate on firm deposits
        self.bank_data["Overdraft Rate on Firm Deposits"] = (
            central_bank_policy_rate + bank_markup_interest_rate_overdraft_firm
        )

        # Interest rates on household deposits
        self.bank_data["Interest Rates on Household Deposits"] = central_bank_policy_rate

        # Overdraft rate on household deposits
        self.bank_data["Overdraft Rate on Household Deposits"] = (
            central_bank_policy_rate + bank_markup_interest_rate_overdraft_household
        )

    def set_interest_received_from_loans(self) -> None:
        self.bank_data["Interest received from Loans"] = (
            self.bank_data["Long-Term Interest Rates on Firm Loans"] * self.bank_data["Loans to Firms"]
            + self.bank_data["Interest Rates on Household Consumption Loans"]
            * self.bank_data["Consumption Loans to Households"]
            + self.bank_data["Interest Rates on Mortgages"] * self.bank_data["Mortgages to Households"]
        )

    def set_interest_received_from_deposits(self, central_bank_policy_rate: float) -> None:
        self.bank_data["Interest received from Deposits"] = (
            central_bank_policy_rate * self.bank_data["Deposits"]
            + self.bank_data["Overdraft Rate on Firm Deposits"] * np.maximum(0, -self.bank_data["Deposits from Firms"])
            + self.bank_data["Overdraft Rate on Household Deposits"]
            * np.maximum(0, -self.bank_data["Deposits from Households"])
        ) - (
            +self.bank_data["Interest Rates on Firm Deposits"] * np.maximum(0, self.bank_data["Deposits from Firms"])
            + self.bank_data["Interest Rates on Household Deposits"]
            * np.maximum(0, self.bank_data["Deposits from Households"])
        )

    def set_profits(self) -> None:
        self.bank_data["Profits"] = (
            self.bank_data["Interest received from Loans"] + self.bank_data["Interest received from Deposits"]
        )

    def set_corporate_taxes_paid(self, tau_bank: float) -> None:
        self.bank_data["Corporate Taxes Paid"] = tau_bank * np.maximum(0.0, self.bank_data["Profits"])

    def set_market_share(self) -> None:
        total_amount_of_loans_and_deposits = (
            np.absolute(self.bank_data["Loans to Firms"]).sum()
            + np.absolute(self.bank_data["Consumption Loans to Households"]).sum()
            + np.absolute(self.bank_data["Mortgages to Households"]).sum()
            + np.absolute(self.bank_data["Deposits from Firms"]).sum()
            + np.absolute(self.bank_data["Deposits from Households"]).sum()
        )
        if total_amount_of_loans_and_deposits > 0:
            self.bank_data["Market Share"] = (
                np.absolute(self.bank_data["Loans to Firms"])
                + np.absolute(self.bank_data["Consumption Loans to Households"])
                + np.absolute(self.bank_data["Mortgages to Households"])
                + np.absolute(self.bank_data["Deposits from Firms"])
                + np.absolute(self.bank_data["Deposits from Households"])
            ) / total_amount_of_loans_and_deposits
        else:
            self.bank_data["Market Share"] = np.full(len(self.bank_data), 1.0 / len(self.bank_data))

    def set_liability(self) -> None:
        self.bank_data["Liability"] = (
            self.bank_data["Equity"]
            + np.maximum(0, self.bank_data["Deposits from Firms"])
            + np.maximum(0, self.bank_data["Deposits from Households"])
            + np.maximum(0, -self.bank_data["Deposits"])
        )

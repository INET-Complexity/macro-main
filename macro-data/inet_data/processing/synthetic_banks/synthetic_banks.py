from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from inet_data.processing.synthetic_population.synthetic_population import SyntheticPopulation
from inet_data.readers.default_readers import DataReaders


class SyntheticBanks(ABC):
    """
    Abstract base class representing a collection of synthetic banks.

    The bank data is stored in a pandas DataFrame with the following columns (one row per bank):
    - Equity: The equity of the bank.
    - Corresponding Firms ID: The IDs of the firms corresponding to the bank.
    - Corresponding Households ID: The IDs of the households corresponding to the bank.
    - Deposits from Households: The deposits from households.
    - Mortgages to Households: The mortgages to households.
    - Consumption Loans to Households: The consumption loans to households.
    - Deposits from Firms: The deposits from firms.
    - Loans to Firms: The loans to firms.
    - Deposits: The deposits.
    - Short-Term Interest Rates on Firm Loans: The short-term interest rates on firm loans.
    - Long-Term Interest Rates on Firm Loans: The long-term interest rates on firm loans.
    - Interest Rates on Household Payday Loans: The interest rates on household payday loans.
    - Interest Rates on Household Consumption Loans: The interest rates on household consumption loans.
    - Interest Rates on Mortgages: The interest rates on mortgages.
    - Interest Rates on Firm Deposits: The interest rates on firm deposits.
    - Overdraft Rate on Firm Deposits: The overdraft rate on firm deposits.
    - Interest Rates on Household Deposits: The interest rates on household deposits.
    - Overdraft Rate on Household Deposits: The overdraft rate on household deposits.
    - Interest received from Loans: The interest received from loans.
    - Interest received from Deposits: The interest received from deposits.
    - Profits: The profits.
    - Corporate Taxes Paid: The corporate taxes paid.
    - Liability: The liability.
    - Market Share: The market share.


    Attributes:
        country_name (str): The name of the country.
        year (int): The year of the data.
        number_of_banks (int): The number of banks.
        bank_data (pd.DataFrame): The bank data.

    Methods:
        __init__(self, country_name: str, year: int, number_of_banks: int, bank_data: pd.DataFrame): Initializes the SyntheticBanks object.
        create_agents(self, bank_equity: float) -> None: Creates agents with the specified bank equity.
        initialise_deposits_and_loans(self, synthetic_population: SyntheticPopulation, synthetic_firms: SyntheticFirms) -> None: Initializes the deposits and loans for households and firms.
        set_deposits_from_firms(self, firm_deposits: np.ndarray) -> None: Sets the deposits from firms.
        set_deposits_from_households(self, household_deposits: np.ndarray) -> None: Sets the deposits from households.
        set_loans_to_firms(self, firm_debt: np.ndarray) -> None: Sets the loans to firms.
        set_loans_to_households(self, household_mortgage_debt: np.ndarray, household_other_debt: np.ndarray) -> None: Sets the loans to households.
        set_bank_equity(self, bank_equity: float) -> None: Sets the bank equity.
        set_bank_deposits(self, firm_deposits: np.ndarray, household_deposits: np.ndarray, firm_debt: np.ndarray, household_debt: np.ndarray) -> None: Sets the bank deposits.
        initialise_rates_profits_liabilities(self, readers: DataReaders,
                                             bank_markup_interest_rate_household_consumption_loans: float,
                                             bank_markup_interest_rate_mortgages: float,
                                             bank_markup_interest_rate_overdraft_household: float): Initializes the rates, profits, and liabilities for the banks.
        set_initial_interest_rates(self, central_bank_policy_rate: float,
                                    bank_markup_interest_rate_short_term_firm_loans: float,
                                    bank_markup_interest_rate_long_term_firm_loans: float,
                                    bank_markup_interest_rate_household_payday_loans: float,
                                    bank_markup_interest_rate_household_consumption_loans: float,
                                    bank_markup_interest_rate_mortgages: float,
                                    bank_markup_interest_rate_overdraft_firm: float,
                                    bank_markup_interest_rate_overdraft_household: float) -> None: Sets the initial interest rates for the banks.
        set_interest_received_from_loans(self) -> None: Sets the interest received from loans.
        set_interest_received_from_deposits(self, central_bank_policy_rate: float) -> None: Sets the interest received from deposits.
        set_profits(self) -> None: Sets the profits for the banks.
        set_corporate_taxes_paid(self, tau_bank: float) -> None: Sets the corporate taxes paid by the banks.
        set_market_share(self) -> None: Sets the market share for the banks.
        set_liability(self) -> None: Sets the liability for the banks.
    """

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

    def initialise_deposits_and_loans(
        self, synthetic_population: SyntheticPopulation, synthetic_firms: SyntheticFirms
    ) -> None:
        # Set initial household deposits
        household_deposits = synthetic_population.household_data["Wealth in Deposits"].values
        self.set_deposits_from_households(household_deposits=household_deposits)

        # Set initial household loans
        household_mortgage_debt = (
            synthetic_population.household_data["Outstanding Balance of HMR Mortgages"].values
            + synthetic_population.household_data["Outstanding Balance of Mortgages on other Properties"].values
        )
        household_other_debt = synthetic_population.household_data[
            "Outstanding Balance of other Non-Mortgage Loans"
        ].values
        self.set_loans_to_households(
            household_mortgage_debt=household_mortgage_debt,
            household_other_debt=household_other_debt,
        )

        # Set initial firm deposits
        firm_deposits = synthetic_firms.firm_data["Deposits"].values
        self.set_deposits_from_firms(firm_deposits=firm_deposits)

        # Set initial firm loans
        firm_debt = synthetic_firms.firm_data["Debt"].values
        self.set_loans_to_firms(firm_debt=firm_debt)

        # Set initial bank deposits
        self.set_bank_deposits(
            firm_deposits=firm_deposits,
            firm_debt=firm_debt,
            household_deposits=household_deposits,
            household_debt=household_mortgage_debt + household_other_debt,
        )

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

    def initialise_rates_profits_liabilities(
        self,
        readers: DataReaders,
        consumption_loans_markup: float,
        mortgage_markup: float,
        household_overdraft_markup: float,
    ):
        policy_rate = readers.policy_rates.cb_policy_rate(self.country_name, self.year)

        # bank tax rate set to same as corporate tax rate
        tau_bank = readers.oecd_econ.read_tau_firm(self.country_name, self.year)

        risk_premium = readers.eurostat.firm_risk_premium(self.country_name, self.year)
        bank_markup_interest_rate_short_term_firm_loans = risk_premium
        bank_markup_interest_rate_long_term_firm_loans = risk_premium
        bank_markup_interest_rate_household_payday_loans = risk_premium
        bank_markup_interest_rate_overdraft_firm = risk_premium

        self.set_initial_interest_rates(
            central_bank_policy_rate=policy_rate,
            bank_markup_interest_rate_short_term_firm_loans=bank_markup_interest_rate_short_term_firm_loans,
            bank_markup_interest_rate_long_term_firm_loans=bank_markup_interest_rate_long_term_firm_loans,
            bank_markup_interest_rate_household_payday_loans=bank_markup_interest_rate_household_payday_loans,
            bank_markup_interest_rate_household_consumption_loans=consumption_loans_markup,
            bank_markup_interest_rate_mortgages=mortgage_markup,
            bank_markup_interest_rate_overdraft_firm=bank_markup_interest_rate_overdraft_firm,
            bank_markup_interest_rate_overdraft_household=household_overdraft_markup,
        )

        self.set_interest_received_from_loans()
        self.set_interest_received_from_deposits(central_bank_policy_rate=policy_rate)
        self.set_profits()
        self.set_corporate_taxes_paid(tau_bank=tau_bank)

        # Set initial bank liabilities
        self.set_liability()

        # Compute the initial market share
        self.set_market_share()

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

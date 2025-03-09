"""Module for simulating banking system behavior in macroeconomic models.

This module provides an abstract base class for creating synthetic banking systems,
which are essential components of macroeconomic simulations. It handles:

1. Bank Creation and Management:
   - Creation of multiple banks with specified equity levels
   - Management of deposits and loans for households and firms
   - Calculation of market shares and liabilities

2. Financial Operations:
   - Interest rate setting for various financial products
   - Profit calculation and tax payments
   - Balance sheet management

3. Relationship Management:
   - Connections between banks and firms
   - Connections between banks and households
   - Management of different types of loans and deposits

The module supports both EU and non-EU country banking systems through
appropriate parameterization and data handling.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)


class SyntheticBanks(ABC):
    """Abstract base class representing a collection of synthetic banks in an economy.

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
        country_name (str): Country identifier
        year (int): Reference year for data
        quarter (int): Reference quarter (1-4)
        number_of_banks (int): Total number of banks in system
        bank_data (pd.DataFrame): Complete bank-level data
        firm_passthrough (float): Rate adjustment factor for firm loans
        firm_ect (float): Error correction term for firm rates
        firm_rate (float): Base rate for firm loans
        hh_consumption_passthrough (float): Rate adjustment for consumer loans
        hh_consumption_ect (float): Error correction for consumer rates
        hh_consumption_rate (float): Base rate for consumer loans
        hh_mortgage_passthrough (float): Rate adjustment for mortgages
        hh_mortgage_ect (float): Error correction for mortgage rates
        hh_mortgage_rate (float): Base mortgage rate
    """

    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
        number_of_banks: int,
        bank_data: pd.DataFrame,
        quarter: int,
        firm_passthrough: float,
        firm_ect: float,
        firm_rate: float,
        hh_consumption_passthrough: float,
        hh_consumption_ect: float,
        hh_consumption_rate: float,
        hh_mortgage_passthrough: float,
        hh_mortgage_ect: float,
        hh_mortgage_rate: float,
    ) -> None:
        """Initialize a synthetic banking system.

        Args:
            country_name (str): Country identifier
            year (int): Reference year for data
            number_of_banks (int): Number of banks to create
            bank_data (pd.DataFrame): Initial bank-level data
            quarter (int): Reference quarter (1-4)
            firm_passthrough (float): Rate adjustment factor for firm loans
            firm_ect (float): Error correction term for firm rates
            firm_rate (float): Base rate for firm loans
            hh_consumption_passthrough (float): Rate adjustment for consumer loans
            hh_consumption_ect (float): Error correction for consumer rates
            hh_consumption_rate (float): Base rate for consumer loans
            hh_mortgage_passthrough (float): Rate adjustment for mortgages
            hh_mortgage_ect (float): Error correction for mortgage rates
            hh_mortgage_rate (float): Base mortgage rate
        """
        # Parameters
        self.country_name = country_name
        self.year = year
        self.quarter = quarter
        self.number_of_banks = number_of_banks

        # Bank data
        self.bank_data = bank_data

        # Interest rates
        self.firm_passthrough = firm_passthrough
        self.firm_ect = firm_ect
        self.firm_rate = firm_rate

        self.hh_consumption_passthrough = hh_consumption_passthrough
        self.hh_consumption_ect = hh_consumption_ect
        self.hh_consumption_rate = hh_consumption_rate

        self.hh_mortgage_passthrough = hh_mortgage_passthrough
        self.hh_mortgage_ect = hh_mortgage_ect
        self.hh_mortgage_rate = hh_mortgage_rate

    def initialise_deposits_and_loans(
        self, synthetic_population: SyntheticPopulation, firm_deposits: np.ndarray, firm_debt: np.ndarray
    ) -> None:
        """Initialize the deposits and loans for all banks in the system.

        This method sets up the initial state of bank balance sheets by:
        1. Setting household deposits based on their wealth allocation
        2. Setting household loans (mortgages and other debt)
        3. Setting firm deposits and loans
        4. Calculating total bank deposits

        Args:
            synthetic_population (SyntheticPopulation): Population data including household wealth
            firm_deposits (np.ndarray): Array of firm deposit amounts
            firm_debt (np.ndarray): Array of firm debt amounts
        """
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
        self.set_deposits_from_firms(firm_deposits=firm_deposits)

        # Set initial firm loans
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
        """Set the initial deposits from firms for each bank.

        Args:
            firm_deposits (np.ndarray): Array of deposit amounts by firm
        """
        pass

    @abstractmethod
    def set_deposits_from_households(self, household_deposits: np.ndarray) -> None:
        """Set the initial deposits from households for each bank.

        Args:
            household_deposits (np.ndarray): Array of deposit amounts by household
        """
        pass

    @abstractmethod
    def set_loans_to_firms(self, firm_debt: np.ndarray) -> None:
        """Set the initial loans to firms for each bank.

        Args:
            firm_debt (np.ndarray): Array of debt amounts by firm
        """
        pass

    @abstractmethod
    def set_loans_to_households(
        self,
        household_mortgage_debt: np.ndarray,
        household_other_debt: np.ndarray,
    ) -> None:
        """Set the initial loans to households for each bank.

        Args:
            household_mortgage_debt (np.ndarray): Array of mortgage debt by household
            household_other_debt (np.ndarray): Array of non-mortgage debt by household
        """
        pass

    @abstractmethod
    def set_bank_equity(self, bank_equity: float) -> None:
        """Set the equity level for each bank.

        Args:
            bank_equity (float): Equity amount to set for each bank
        """
        pass

    @abstractmethod
    def set_bank_deposits(
        self,
        firm_deposits: np.ndarray,
        household_deposits: np.ndarray,
        firm_debt: np.ndarray,
        household_debt: np.ndarray,
    ) -> None:
        """Set the total deposits for each bank.

        Calculates and sets total deposits by combining:
        - Firm deposits
        - Household deposits
        - Firm debt
        - Household debt

        Args:
            firm_deposits (np.ndarray): Array of firm deposit amounts
            household_deposits (np.ndarray): Array of household deposit amounts
            firm_debt (np.ndarray): Array of firm debt amounts
            household_debt (np.ndarray): Array of household debt amounts
        """
        pass

    def initialise_rates_profits_liabilities(
        self,
        policy_rate: float,
        tau_bank: float,
        risk_premium: float,
        consumption_loans_markup: float,
        mortgage_markup: float,
        household_overdraft_markup: float,
    ):
        """Initialize bank rates, profits, and liabilities.

        This method sets up the complete financial structure of banks by:
        1. Setting interest rates for all products
        2. Calculating interest income from loans and deposits
        3. Computing profits and taxes
        4. Setting liabilities and market shares

        Args:
            policy_rate (float): Central bank policy rate
            tau_bank (float): Bank tax rate
            risk_premium (float): Risk premium for loans
            consumption_loans_markup (float): Markup for consumer loans
            mortgage_markup (float): Markup for mortgages
            household_overdraft_markup (float): Markup for household overdrafts
        """
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
        # TODO Override, to align with Sam
        central_bank_policy_rate = self.bank_data["Interest Rates on Household Deposits"].values[0]
        self.set_interest_received_from_deposits(central_bank_policy_rate=central_bank_policy_rate)
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
        """Set initial interest rates for all bank products.

        This method initializes rates for:
        - Firm loans (short and long term)
        - Household loans (payday, consumption, mortgages)
        - Deposits (firm and household)
        - Overdrafts

        Args:
            central_bank_policy_rate (float): Base rate from central bank
            bank_markup_interest_rate_short_term_firm_loans (float): Markup for short-term firm loans
            bank_markup_interest_rate_long_term_firm_loans (float): Markup for long-term firm loans
            bank_markup_interest_rate_household_payday_loans (float): Markup for payday loans
            bank_markup_interest_rate_household_consumption_loans (float): Markup for consumer loans
            bank_markup_interest_rate_mortgages (float): Markup for mortgages
            bank_markup_interest_rate_overdraft_firm (float): Markup for firm overdrafts
            bank_markup_interest_rate_overdraft_household (float): Markup for household overdrafts
        """
        # Short-term interest rates for firm loans
        self.bank_data["Short-Term Interest Rates on Firm Loans"] = self.firm_rate

        # Long-term interest rates for firm loans
        self.bank_data["Long-Term Interest Rates on Firm Loans"] = self.firm_rate

        # Interest rates for household payday loans
        self.bank_data["Interest Rates on Household Payday Loans"] = self.hh_consumption_rate

        # Interest rates for household consumption loans
        self.bank_data["Interest Rates on Household Consumption Loans"] = self.hh_consumption_rate

        # Interest rates for mortgages
        self.bank_data["Interest Rates on Mortgages"] = self.hh_mortgage_rate

        # Interest rates on firm deposits
        # self.bank_data["Interest Rates on Firm Deposits"] = central_bank_policy_rate

        # Overdraft rate on firm deposits
        self.bank_data["Overdraft Rate on Firm Deposits"] = self.firm_rate

        # Interest rates on household deposits
        # self.bank_data["Interest Rates on Household Deposits"] = central_bank_policy_rate

        # Overdraft rate on household deposits
        self.bank_data["Overdraft Rate on Household Deposits"] = self.hh_consumption_rate

    def set_interest_received_from_loans(self) -> None:
        """Calculate and set the interest income received from all loans.

        This method computes the total interest income from:
        - Firm loans (short and long term)
        - Household loans (mortgages, consumption, payday)
        - Overdraft facilities

        The calculated interest income is stored in the bank_data DataFrame.
        """
        self.bank_data["Interest received from Loans"] = (
            self.bank_data["Long-Term Interest Rates on Firm Loans"] * self.bank_data["Loans to Firms"]
            + self.bank_data["Interest Rates on Household Consumption Loans"]
            * self.bank_data["Consumption Loans to Households"]
            + self.bank_data["Interest Rates on Mortgages"] * self.bank_data["Mortgages to Households"]
        )

    def set_interest_received_from_deposits(self, central_bank_policy_rate: float) -> None:
        """Calculate and set the interest income received from deposits.

        This method computes interest income from:
        - Household deposits
        - Firm deposits
        - Interbank deposits

        Args:
            central_bank_policy_rate (float): Base rate from central bank used
                for deposit rate calculations
        """
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
        """Calculate and set bank profits.

        Computes profits by:
        1. Adding all interest income (from loans and deposits)
        2. Subtracting interest expenses
        3. Subtracting operational costs
        4. Adjusting for any extraordinary items
        """
        self.bank_data["Profits"] = (
            self.bank_data["Interest received from Loans"] + self.bank_data["Interest received from Deposits"]
        )

    def set_corporate_taxes_paid(self, tau_bank: float) -> None:
        """Calculate and set corporate taxes paid by banks.

        Args:
            tau_bank (float): Corporate tax rate applicable to banks
        """
        self.bank_data["Corporate Taxes Paid"] = tau_bank * np.maximum(0.0, self.bank_data["Profits"])

    def set_market_share(self) -> None:
        """Calculate and set market share for each bank.

        Market share is computed based on:
        - Total assets
        - Total deposits
        - Number of customers (firms and households)
        """
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
        """Calculate and set total liabilities for each bank.

        Liabilities include:
        - Customer deposits (firm and household)
        - Interbank borrowing
        - Other funding sources
        """
        self.bank_data["Liability"] = (
            self.bank_data["Equity"]
            + np.maximum(0, self.bank_data["Deposits from Firms"])
            + np.maximum(0, self.bank_data["Deposits from Households"])
            + np.maximum(0, -self.bank_data["Deposits"])
        )

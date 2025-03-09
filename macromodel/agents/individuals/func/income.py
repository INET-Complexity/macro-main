"""Individual income determination and calculation.

This module implements strategies for calculating individual incomes from
various sources through:
- Employment earnings
- Social benefits
- Investment returns (firms/banks)
- Dividend income

The implementation handles:
- Income source identification
- Tax adjustments
- Inflation effects
- Activity status impacts
"""

from abc import ABC, abstractmethod

import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus


class IncomeSetter(ABC):
    """Abstract base class for individual income calculation.

    This class defines strategies for computing individual incomes from
    multiple sources based on:
    - Employment status
    - Investment positions
    - Social benefits
    - Economic conditions

    The calculations consider:
    - Activity status effects
    - Tax implications
    - Price level changes
    - Investment returns
    """

    @abstractmethod
    def compute_expected_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
        expected_firm_profits: np.ndarray,
        corr_invested_firms: np.ndarray,
        expected_bank_profits: np.ndarray,
        corr_invested_banks: np.ndarray,
        cpi: float,
        expected_inflation: float,
        dividend_payout_ratio: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        """Calculate expected future income for individuals.

        Args:
            current_individual_activity_status (np.ndarray): Activity status by individual
            current_wage (np.ndarray): Current wages by individual
            individual_social_benefits (np.ndarray): Benefits by individual
            expected_firm_profits (np.ndarray): Expected profits by firm
            corr_invested_firms (np.ndarray): Individual-firm investment links
            expected_bank_profits (np.ndarray): Expected profits by bank
            corr_invested_banks (np.ndarray): Individual-bank investment links
            cpi (float): Current price index
            expected_inflation (float): Expected inflation rate
            dividend_payout_ratio (float): Share of profits paid as dividends
            income_taxes (float): Personal income tax rate
            tau_firm (float): Corporate tax rate

        Returns:
            np.ndarray: Expected income by individual
        """
        pass

    def compute_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
        firm_profits: np.ndarray,
        corr_invested_firms: np.ndarray,
        bank_profits: np.ndarray,
        corr_invested_banks: np.ndarray,
        cpi: float,
        dividend_payout_ratio: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        """Calculate current period income for individuals.

        Args:
            current_individual_activity_status (np.ndarray): Activity status by individual
            current_wage (np.ndarray): Current wages by individual
            individual_social_benefits (np.ndarray): Benefits by individual
            firm_profits (np.ndarray): Current profits by firm
            corr_invested_firms (np.ndarray): Individual-firm investment links
            bank_profits (np.ndarray): Current profits by bank
            corr_invested_banks (np.ndarray): Individual-bank investment links
            cpi (float): Current price index
            dividend_payout_ratio (float): Share of profits paid as dividends
            income_taxes (float): Personal income tax rate
            tau_firm (float): Corporate tax rate

        Returns:
            np.ndarray: Current income by individual
        """
        pass


class DefaultIncomeSetter(IncomeSetter):
    """Default implementation of individual income calculation.

    This class implements income computation through:
    - Activity-based income determination
    - Investment return calculation
    - Tax and inflation adjustment
    - Benefit incorporation

    The approach:
    - Differentiates by activity status
    - Applies appropriate tax rates
    - Adjusts for price levels
    - Includes all income sources
    """

    def compute_expected_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
        expected_firm_profits: np.ndarray,
        corr_invested_firms: np.ndarray,
        expected_bank_profits: np.ndarray,
        corr_invested_banks: np.ndarray,
        cpi: float,
        expected_inflation: float,
        dividend_payout_ratio: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        """Calculate expected future income for individuals.

        Computes income based on:
        - Employment status (wages)
        - Investment positions (dividends)
        - Social benefits
        Adjusted for:
        - Expected inflation
        - Tax rates
        - Price levels

        Args:
            current_individual_activity_status (np.ndarray): Activity status by individual
            current_wage (np.ndarray): Current wages by individual
            individual_social_benefits (np.ndarray): Benefits by individual
            expected_firm_profits (np.ndarray): Expected profits by firm
            corr_invested_firms (np.ndarray): Individual-firm investment links
            expected_bank_profits (np.ndarray): Expected profits by bank
            corr_invested_banks (np.ndarray): Individual-bank investment links
            cpi (float): Current price index
            expected_inflation (float): Expected inflation rate
            dividend_payout_ratio (float): Share of profits paid as dividends
            income_taxes (float): Personal income tax rate
            tau_firm (float): Corporate tax rate

        Returns:
            np.ndarray: Expected income by individual
        """
        income = np.zeros_like(current_individual_activity_status)

        # Employed individuals
        emp_ind = current_individual_activity_status == ActivityStatus.EMPLOYED
        income[emp_ind] = (1 + expected_inflation) * cpi * current_wage[emp_ind]

        # Unemployed individuals
        unemp_ind = current_individual_activity_status == ActivityStatus.UNEMPLOYED
        income[unemp_ind] = 0.0

        # Not-economically active individuals
        nea_ind = current_individual_activity_status == ActivityStatus.NOT_ECONOMICALLY_ACTIVE
        income[nea_ind] = 0.0

        # Firm investors
        firm_inv_ind = current_individual_activity_status == ActivityStatus.FIRM_INVESTOR
        income[firm_inv_ind] = (
            dividend_payout_ratio
            * (1 - income_taxes)
            * (1 - tau_firm)
            * np.maximum(0.0, expected_firm_profits[corr_invested_firms[firm_inv_ind]])
        )

        # Bank investors
        bank_inv_ind = current_individual_activity_status == ActivityStatus.BANK_INVESTOR
        income[bank_inv_ind] = (
            dividend_payout_ratio
            * (1 - income_taxes)
            * (1 - tau_firm)
            * np.maximum(0.0, expected_bank_profits[corr_invested_banks[bank_inv_ind]])
        )
        return (1 + expected_inflation) * cpi * individual_social_benefits + income

    def compute_income(
        self,
        current_individual_activity_status: np.ndarray,
        current_wage: np.ndarray,
        individual_social_benefits: np.ndarray,
        firm_profits: np.ndarray,
        corr_invested_firms: np.ndarray,
        bank_profits: np.ndarray,
        corr_invested_banks: np.ndarray,
        cpi: float,
        dividend_payout_ratio: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        """Calculate current period income for individuals.

        Computes income based on:
        - Employment status (wages)
        - Investment positions (dividends)
        - Social benefits
        Adjusted for:
        - Current price levels
        - Tax rates

        Args:
            current_individual_activity_status (np.ndarray): Activity status by individual
            current_wage (np.ndarray): Current wages by individual
            individual_social_benefits (np.ndarray): Benefits by individual
            firm_profits (np.ndarray): Current profits by firm
            corr_invested_firms (np.ndarray): Individual-firm investment links
            bank_profits (np.ndarray): Current profits by bank
            corr_invested_banks (np.ndarray): Individual-bank investment links
            cpi (float): Current price index
            dividend_payout_ratio (float): Share of profits paid as dividends
            income_taxes (float): Personal income tax rate
            tau_firm (float): Corporate tax rate

        Returns:
            np.ndarray: Current income by individual
        """
        income = np.zeros_like(current_individual_activity_status)

        # Employed individuals
        emp_ind = current_individual_activity_status == ActivityStatus.EMPLOYED
        income[emp_ind] = cpi * current_wage[emp_ind]

        # Unemployed individuals
        unemp_ind = current_individual_activity_status == ActivityStatus.UNEMPLOYED
        income[unemp_ind] = 0.0

        # Not-economically active individuals
        nea_ind = current_individual_activity_status == ActivityStatus.NOT_ECONOMICALLY_ACTIVE
        income[nea_ind] = 0.0

        # Firm investors
        firm_inv_ind = current_individual_activity_status == ActivityStatus.FIRM_INVESTOR
        income[firm_inv_ind] = (
            dividend_payout_ratio
            * (1 - income_taxes)
            * (1 - tau_firm)
            * np.maximum(0.0, firm_profits[corr_invested_firms[firm_inv_ind].astype(int)])
        )

        # Bank investors
        bank_inv_ind = current_individual_activity_status == ActivityStatus.BANK_INVESTOR
        income[bank_inv_ind] = (
            dividend_payout_ratio
            * (1 - income_taxes)
            * (1 - tau_firm)
            * np.maximum(0.0, bank_profits[corr_invested_banks[bank_inv_ind].astype(int)])
        )

        return cpi * individual_social_benefits + income

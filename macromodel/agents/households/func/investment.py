"""Household investment behavior implementation.

This module implements household investment decisions through:
- Target investment calculation
- Income-based allocation
- Industry-specific investment
- Tax-adjusted spending
- Price level adjustments

The implementation handles:
- Investment rate application
- Industry allocation weights
- Tax considerations
- Inflation adjustments
- External targets
"""

from abc import ABC, abstractmethod

import numpy as np


class HouseholdInvestment(ABC):
    """Abstract base class for household investment behavior.

    Defines interface for computing target investment levels based on:
    - Income and investment rates
    - Industry allocations
    - Price level changes
    - Tax considerations
    """

    def __init__(self):
        pass

    @abstractmethod
    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        income: np.ndarray,
        exogenous_total_investment: np.ndarray,
        current_time: int,
        investment_weights: np.ndarray,
        investment_rate: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate target investment levels.

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            income (np.ndarray): Household income
            exogenous_total_investment (np.ndarray): External investment target
            current_time (int): Current period
            investment_weights (np.ndarray): Industry investment shares
            investment_rate (np.ndarray): Investment/income ratios
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Target investment by household and industry
        """
        pass


class NoHouseholdInvestment(HouseholdInvestment):
    """Zero investment implementation.

    Returns zero investment for all households and industries.
    Used for scenarios where household investment is not modeled.
    """

    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        income: np.ndarray,
        exogenous_total_investment: np.ndarray,
        current_time: int,
        investment_weights: np.ndarray,
        investment_rate: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Return zero investment targets.

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            income (np.ndarray): Household income
            exogenous_total_investment (np.ndarray): External investment target
            current_time (int): Current period
            investment_weights (np.ndarray): Industry investment shares
            investment_rate (np.ndarray): Investment/income ratios
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Zero investment array
        """
        return np.zeros((income.shape[0], investment_weights.shape[0]))


class DefaultHouseholdInvestment(HouseholdInvestment):
    """Default implementation of household investment behavior.

    Implements investment decisions based on:
    - Income and investment rates
    - Industry allocation weights
    - Tax adjustments
    """

    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        income: np.ndarray,
        exogenous_total_investment: np.ndarray,
        current_time: int,
        investment_weights: np.ndarray,
        investment_rate: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate target investment using default behavior.

        Determines investment based on:
        - Income-based investment rates
        - Industry allocation weights
        - Tax adjustments

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            income (np.ndarray): Household income
            exogenous_total_investment (np.ndarray): External investment target
            current_time (int): Current period
            investment_weights (np.ndarray): Industry investment shares
            investment_rate (np.ndarray): Investment/income ratios
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Target investment by household and industry
        """
        return 1.0 / (1 + tau_cf) * np.outer(investment_weights, investment_rate * income).T


class ExogenousHouseholdInvestment(HouseholdInvestment):
    """Exogenous household investment implementation.

    Implements investment decisions based on:
    - External investment targets
    - Price level adjustments
    - Income-based allocation
    - Tax considerations
    """

    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        income: np.ndarray,
        exogenous_total_investment: np.ndarray,
        current_time: int,
        investment_weights: np.ndarray,
        investment_rate: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate target investment using exogenous targets.

        Determines investment based on:
        - External investment targets
        - Price level changes
        - Income-based allocation
        - Tax adjustments

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            income (np.ndarray): Household income
            exogenous_total_investment (np.ndarray): External investment target
            current_time (int): Current period
            investment_weights (np.ndarray): Industry investment shares
            investment_rate (np.ndarray): Investment/income ratios
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Target investment by household and industry
        """
        target_investment = np.maximum(
            0.0,
            (1.0 / (1 + tau_cf) * np.outer(investment_weights, investment_rate * income).T),
        )
        return (
            (1 + expected_inflation)
            * current_cpi
            / initial_cpi
            * 1.0
            / (1 + tau_cf)
            * exogenous_total_investment[current_time]
            * target_investment
            / target_investment.sum()
        )

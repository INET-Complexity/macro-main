"""Bank profit estimation strategies.

This module implements approaches for estimating future bank profits
through:
- Current profit extrapolation
- Growth rate adjustments
- Inflation expectations
- Economic condition incorporation

The estimates consider:
- Historical profitability
- Economic growth outlook
- Price level changes
- Bank-specific factors
"""

from abc import ABC, abstractmethod

import numpy as np


class BankProfitsSetter(ABC):
    """Abstract base class for bank profit estimation.

    This class defines strategies for estimating future bank profits
    based on:
    - Current profit levels
    - Economic growth expectations
    - Inflation projections
    - Bank-specific conditions

    The estimates consider:
    - Historical performance
    - Macroeconomic outlook
    - Price level dynamics
    - Individual bank factors
    """

    @abstractmethod
    def compute_estimated_profits(
        self,
        current_profits: np.ndarray,
        estimated_growth: float,
        estimated_inflation: float,
    ) -> np.ndarray:
        """Calculate estimated future profits.

        Args:
            current_profits (np.ndarray): Current profits by bank
            estimated_growth (float): Expected economic growth rate
            estimated_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Estimated future profits by bank
        """
        pass


class DefaultBankProfitsSetter(BankProfitsSetter):
    """Default implementation of bank profit estimation.

    This class implements profit estimation through:
    - Growth rate adjustments
    - Inflation expectations
    - Current profit scaling
    - Bank-level projections

    The approach:
    - Applies growth expectations
    - Incorporates inflation effects
    - Maintains bank-specific patterns
    - Ensures non-negative estimates
    """

    def compute_estimated_profits(
        self,
        current_profits: np.ndarray,
        estimated_growth: float,
        estimated_inflation: float,
    ) -> np.ndarray:
        """Calculate estimated future profits.

        Estimates profits based on:
        - Current profit levels
        - Expected economic growth
        - Expected inflation
        - Bank-specific factors

        Args:
            current_profits (np.ndarray): Current profits by bank
            estimated_growth (float): Expected economic growth rate
            estimated_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Estimated future profits by bank
        """
        return (1 + estimated_growth) * (1 + estimated_inflation) * current_profits

"""Household financial asset management implementation.

This module implements household financial asset management through:
- Income expectation calculation
- Actual income realization
- Asset return modeling
- Stochastic return adjustments

The implementation handles:
- Expected income computation
- Realized income determination
- Return coefficient application
- Random noise incorporation
"""

from abc import ABC, abstractmethod

import numpy as np


class FinancialAssets(ABC):
    """Abstract base class for household financial asset management.

    Defines interface for managing financial asset returns through:
    - Income expectation calculation
    - Actual income realization
    - Return coefficient application
    - Stochastic adjustments

    Attributes:
        income_from_fa_noise_std (float): Standard deviation for return noise
    """

    def __init__(self, income_from_fa_noise_std: float):
        """Initialize financial asset management.

        Args:
            income_from_fa_noise_std (float): Standard deviation for return noise
        """
        self.income_from_fa_noise_std = income_from_fa_noise_std

    @abstractmethod
    def compute_expected_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate expected income from financial assets.

        Args:
            income_coefficient (float): Return coefficient
            initial_other_financial_assets (np.ndarray): Initial asset values
            current_other_financial_assets (np.ndarray): Current asset values

        Returns:
            np.ndarray: Expected income by household
        """
        pass

    @abstractmethod
    def compute_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate realized income from financial assets.

        Args:
            income_coefficient (float): Return coefficient
            initial_other_financial_assets (np.ndarray): Initial asset values
            current_other_financial_assets (np.ndarray): Current asset values

        Returns:
            np.ndarray: Realized income by household
        """
        pass


class DefaultFinancialAssets(FinancialAssets):
    """Default implementation of financial asset management.

    Implements asset return calculation through:
    - Linear return model
    - Stochastic noise addition
    - Asset value tracking
    """

    def compute_expected_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate expected income using default behavior.

        Determines expected returns through:
        - Linear return model
        - Current asset values
        - Return coefficient

        Args:
            income_coefficient (float): Return coefficient
            initial_other_financial_assets (np.ndarray): Initial asset values
            current_other_financial_assets (np.ndarray): Current asset values

        Returns:
            np.ndarray: Expected income by household
        """
        return income_coefficient * current_other_financial_assets

    def compute_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate realized income using default behavior.

        Determines actual returns through:
        - Linear return model
        - Stochastic noise
        - Current asset values

        Args:
            income_coefficient (float): Return coefficient
            initial_other_financial_assets (np.ndarray): Initial asset values
            current_other_financial_assets (np.ndarray): Current asset values

        Returns:
            np.ndarray: Realized income by household
        """
        return (
            (
                1
                + np.random.normal(
                    0.0,
                    self.income_from_fa_noise_std,
                    initial_other_financial_assets.shape,
                )
            )
            * income_coefficient
            * current_other_financial_assets
        )

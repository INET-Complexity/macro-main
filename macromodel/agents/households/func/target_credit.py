"""Household credit demand determination implementation.

This module implements household credit demand through:
- Consumption loan targeting
- Mortgage demand calculation
- Down payment requirements
- Financial asset consideration

The implementation handles:
- Credit need assessment
- Loan amount calculation
- Asset-based adjustments
- Affordability checks
"""

from abc import ABC, abstractmethod

import numpy as np


class HouseholdTargetCredit(ABC):
    """Abstract base class for household credit demand behavior.

    Defines interface for determining credit needs based on:
    - Consumption financing
    - Property purchases
    - Asset holdings
    - Income flows

    Attributes:
        down_payment_fraction (float): Required down payment ratio
    """

    def __init__(
        self,
        down_payment_fraction: float,
    ) -> None:
        """Initialize household target credit behavior.

        Args:
            down_payment_fraction (float): Required down payment ratio
        """
        self.down_payment_fraction = down_payment_fraction

    @abstractmethod
    def compute_target_consumption_loans(
        self,
        target_consumption: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate target consumption loan demand.

        Args:
            target_consumption (np.ndarray): Desired consumption
            income (np.ndarray): Household income
            rent (np.ndarray): Rental payments
            wealth_in_financial_assets (np.ndarray): Financial assets

        Returns:
            np.ndarray: Target consumption loans by household
        """
        pass

    @abstractmethod
    def compute_target_mortgage(
        self,
        target_house_price: np.ndarray,
        target_consumption: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate target mortgage demand.

        Args:
            target_house_price (np.ndarray): Property purchase prices
            target_consumption (np.ndarray): Desired consumption
            income (np.ndarray): Household income
            rent (np.ndarray): Rental payments
            wealth_in_financial_assets (np.ndarray): Financial assets

        Returns:
            np.ndarray: Target mortgage amounts by household
        """
        pass


class DefaultHouseholdTargetCredit(HouseholdTargetCredit):
    """Default implementation of household credit demand behavior.

    Implements credit demand through:
    - Consumption gap assessment
    - Down payment calculation
    - Asset availability check
    - Income consideration
    """

    def compute_target_consumption_loans(
        self,
        target_consumption: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate consumption loans using default behavior.

        Determines loan needs based on:
        - Consumption targets
        - Available income
        - Asset holdings
        - Rental obligations

        Args:
            target_consumption (np.ndarray): Desired consumption
            income (np.ndarray): Household income
            rent (np.ndarray): Rental payments
            wealth_in_financial_assets (np.ndarray): Financial assets

        Returns:
            np.ndarray: Target consumption loans by household
        """
        return np.maximum(
            0.0,
            target_consumption.sum(axis=1) - (income - rent) - wealth_in_financial_assets,
        )

    def compute_target_mortgage(
        self,
        target_house_price: np.ndarray,
        target_consumption: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate mortgages using default behavior.

        Determines mortgage amounts based on:
        - Property prices
        - Down payment requirement
        - Available assets
        - Consumption needs

        Args:
            target_house_price (np.ndarray): Property purchase prices
            target_consumption (np.ndarray): Desired consumption
            income (np.ndarray): Household income
            rent (np.ndarray): Rental payments
            wealth_in_financial_assets (np.ndarray): Financial assets

        Returns:
            np.ndarray: Target mortgage amounts by household
        """
        return np.maximum(
            0.0,
            target_house_price
            - self.down_payment_fraction
            * np.maximum(
                0.0,
                wealth_in_financial_assets - (target_consumption.sum(axis=1) - (income - rent)),
            ),
        )

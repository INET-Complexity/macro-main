"""Household wealth management implementation.

This module implements household wealth management through:
- New wealth allocation
- Wealth usage decisions
- Asset value tracking
- Wealth composition updates

The implementation handles:
- Wealth distribution
- Asset depreciation
- Financial holdings
- Deposit management
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

from macromodel.timeseries import TimeSeries


class WealthSetter(ABC):
    """Abstract base class for household wealth management.

    Defines interface for managing wealth through:
    - Wealth allocation decisions
    - Asset value tracking
    - Financial holdings
    - Deposit management

    Attributes:
        other_real_assets_depreciation_rate (float): Asset depreciation rate
        independents (list[str]): Independent variables for wealth decisions
    """

    def __init__(
        self,
        other_real_assets_depreciation_rate: float,
    ):
        self.other_real_assets_depreciation_rate = other_real_assets_depreciation_rate
        self.independents = ["Income", "Debt"]

    @abstractmethod
    def distribute_new_wealth(
        self,
        new_wealth: np.ndarray,
        model: Optional[Any],
        ts: TimeSeries,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate new wealth between deposits and other assets.

        Args:
            new_wealth (np.ndarray): New wealth to allocate
            model (Optional[Any]): Allocation model
            ts (TimeSeries): Time series data

        Returns:
            Tuple[np.ndarray, np.ndarray]: New deposits and other assets
        """
        pass

    @abstractmethod
    def use_up_wealth(
        self,
        used_up_wealth: np.ndarray,
        current_wealth_in_deposits: np.ndarray,
        current_wealth_in_other_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Determine wealth usage from deposits and other assets.

        Args:
            used_up_wealth (np.ndarray): Wealth to be used
            current_wealth_in_deposits (np.ndarray): Current deposits
            current_wealth_in_other_financial_assets (np.ndarray): Other assets

        Returns:
            Tuple[np.ndarray, np.ndarray]: Used deposits and other assets
        """
        pass

    @abstractmethod
    def compute_wealth_in_other_real_assets(
        self,
        current_wealth_in_other_real_assets: np.ndarray,
        current_investment_in_other_real_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate other real asset values.

        Args:
            current_wealth_in_other_real_assets (np.ndarray): Current assets
            current_investment_in_other_real_assets (np.ndarray): New investment

        Returns:
            np.ndarray: Updated real asset values
        """
        pass

    @abstractmethod
    def compute_wealth_in_other_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        new_wealth_in_other_financial_assets: np.ndarray,
        used_up_wealth_in_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate other financial asset values.

        Args:
            current_wealth_in_other_financial_assets (np.ndarray): Current assets
            new_wealth_in_other_financial_assets (np.ndarray): New assets
            used_up_wealth_in_other_financial_assets (np.ndarray): Used assets

        Returns:
            np.ndarray: Updated financial asset values
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_wealth_in_deposits(
        current_wealth_in_deposits: np.ndarray,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        current_interest_paid: np.ndarray,
        price_paid_for_property: np.ndarray,
        debt_installments: np.ndarray,
        new_loans: np.ndarray,
        new_real_wealth: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate deposit values.

        Args:
            current_wealth_in_deposits (np.ndarray): Current deposits
            new_wealth_in_deposits (np.ndarray): New deposits
            used_up_wealth_in_deposits (np.ndarray): Used deposits
            current_interest_paid (np.ndarray): Interest payments
            price_paid_for_property (np.ndarray): Property purchases
            debt_installments (np.ndarray): Debt payments
            new_loans (np.ndarray): New borrowing
            new_real_wealth (np.ndarray): New real assets
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Updated deposit values
        """
        pass


class SimpleWealthSetter(WealthSetter):
    """Simple wealth management implementation.

    Implements basic wealth management through:
    - All new wealth to deposits
    - All usage from deposits
    - Basic asset tracking
    """

    def distribute_new_wealth(
        self,
        new_wealth: np.ndarray,
        model: Optional[Any],
        ts: TimeSeries,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate all new wealth to deposits.

        Args:
            new_wealth (np.ndarray): New wealth to allocate
            model (Optional[Any]): Allocation model
            ts (TimeSeries): Time series data

        Returns:
            Tuple[np.ndarray, np.ndarray]: New deposits and zero other assets
        """
        return new_wealth, np.zeros_like(new_wealth)

    def use_up_wealth(
        self,
        used_up_wealth: np.ndarray,
        current_wealth_in_deposits: np.ndarray,
        current_wealth_in_other_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use all wealth from deposits.

        Args:
            used_up_wealth (np.ndarray): Wealth to be used
            current_wealth_in_deposits (np.ndarray): Current deposits
            current_wealth_in_other_financial_assets (np.ndarray): Other assets

        Returns:
            Tuple[np.ndarray, np.ndarray]: Used deposits and zero other assets
        """
        return used_up_wealth, np.zeros_like(used_up_wealth)

    def compute_wealth_in_other_real_assets(
        self,
        current_wealth_in_other_real_assets: np.ndarray,
        current_investment_in_other_real_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate real assets with depreciation.

        Args:
            current_wealth_in_other_real_assets (np.ndarray): Current assets
            current_investment_in_other_real_assets (np.ndarray): New investment

        Returns:
            np.ndarray: Updated real asset values
        """
        return (
            1 - self.other_real_assets_depreciation_rate
        ) * current_wealth_in_other_real_assets + current_investment_in_other_real_assets

    def compute_wealth_in_other_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        new_wealth_in_other_financial_assets: np.ndarray,
        used_up_wealth_in_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate financial assets with flows.

        Args:
            current_wealth_in_other_financial_assets (np.ndarray): Current assets
            new_wealth_in_other_financial_assets (np.ndarray): New assets
            used_up_wealth_in_other_financial_assets (np.ndarray): Used assets

        Returns:
            np.ndarray: Updated financial asset values
        """
        return (
            current_wealth_in_other_financial_assets
            + new_wealth_in_other_financial_assets
            - used_up_wealth_in_other_financial_assets
        )

    @staticmethod
    def compute_wealth_in_deposits(
        current_wealth_in_deposits: np.ndarray,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        current_interest_paid: np.ndarray,
        price_paid_for_property: np.ndarray,
        debt_installments: np.ndarray,
        new_loans: np.ndarray,
        new_real_wealth: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate deposits with all flows.

        Args:
            current_wealth_in_deposits (np.ndarray): Current deposits
            new_wealth_in_deposits (np.ndarray): New deposits
            used_up_wealth_in_deposits (np.ndarray): Used deposits
            current_interest_paid (np.ndarray): Interest payments
            price_paid_for_property (np.ndarray): Property purchases
            debt_installments (np.ndarray): Debt payments
            new_loans (np.ndarray): New borrowing
            new_real_wealth (np.ndarray): New real assets
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Updated deposit values
        """
        return (
            current_wealth_in_deposits
            + new_wealth_in_deposits
            - used_up_wealth_in_deposits
            - current_interest_paid
            - price_paid_for_property
            - debt_installments
            + new_loans
            - tau_cf * np.maximum(0.0, new_real_wealth)
        )


class DefaultWealthSetter(WealthSetter):
    """Default implementation of household wealth management.

    Implements wealth management through:
    - Model-based allocation
    - Priority-based usage
    - Asset tracking
    - Flow management
    """

    def distribute_new_wealth(
        self,
        new_wealth: np.ndarray,
        model: Optional[Any],
        ts: TimeSeries,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate new wealth using model predictions.

        Args:
            new_wealth (np.ndarray): New wealth to allocate
            model (Optional[Any]): Allocation model
            ts (TimeSeries): Time series data

        Returns:
            Tuple[np.ndarray, np.ndarray]: New deposits and other assets
        """
        assert len(self.independents) > 0
        x = np.stack(
            [ts.current(ind.lower()) for ind in self.independents],
            axis=1,
        )
        non_zero = x.sum(axis=0) != 0.0
        x[:, non_zero] /= x.sum(axis=0)[non_zero]
        pred_deposit_fraction = model.predict(x)
        pred_deposit_fraction[pred_deposit_fraction > 1.0] = 1.0
        pred_deposit_fraction[pred_deposit_fraction < 0.0] = 0.0

        return (
            pred_deposit_fraction * new_wealth,
            (1 - pred_deposit_fraction) * new_wealth,
        )

    def use_up_wealth(
        self,
        used_up_wealth: np.ndarray,
        current_wealth_in_deposits: np.ndarray,
        current_wealth_in_other_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use wealth with priority on other assets.

        Args:
            used_up_wealth (np.ndarray): Wealth to be used
            current_wealth_in_deposits (np.ndarray): Current deposits
            current_wealth_in_other_financial_assets (np.ndarray): Other assets

        Returns:
            Tuple[np.ndarray, np.ndarray]: Used deposits and other assets
        """
        used_up_wealth_in_other_financial_assets = np.minimum(current_wealth_in_other_financial_assets, used_up_wealth)
        used_up_wealth_in_deposits = np.minimum(
            current_wealth_in_deposits,
            used_up_wealth - used_up_wealth_in_other_financial_assets,
        )
        return (
            used_up_wealth_in_deposits,
            used_up_wealth_in_other_financial_assets,
        )

    def compute_wealth_in_other_real_assets(
        self,
        current_wealth_in_other_real_assets: np.ndarray,
        current_investment_in_other_real_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate real assets with depreciation.

        Args:
            current_wealth_in_other_real_assets (np.ndarray): Current assets
            current_investment_in_other_real_assets (np.ndarray): New investment

        Returns:
            np.ndarray: Updated real asset values
        """
        return (
            1 - self.other_real_assets_depreciation_rate
        ) * current_wealth_in_other_real_assets + current_investment_in_other_real_assets

    def compute_wealth_in_other_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        new_wealth_in_other_financial_assets: np.ndarray,
        used_up_wealth_in_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate financial assets with flows.

        Args:
            current_wealth_in_other_financial_assets (np.ndarray): Current assets
            new_wealth_in_other_financial_assets (np.ndarray): New assets
            used_up_wealth_in_other_financial_assets (np.ndarray): Used assets

        Returns:
            np.ndarray: Updated financial asset values
        """
        return (
            current_wealth_in_other_financial_assets
            + new_wealth_in_other_financial_assets
            - used_up_wealth_in_other_financial_assets
        )

    @staticmethod
    def compute_wealth_in_deposits(
        current_wealth_in_deposits: np.ndarray,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        current_interest_paid: np.ndarray,
        price_paid_for_property: np.ndarray,
        debt_installments: np.ndarray,
        new_loans: np.ndarray,
        new_real_wealth: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate deposits with all flows.

        Args:
            current_wealth_in_deposits (np.ndarray): Current deposits
            new_wealth_in_deposits (np.ndarray): New deposits
            used_up_wealth_in_deposits (np.ndarray): Used deposits
            current_interest_paid (np.ndarray): Interest payments
            price_paid_for_property (np.ndarray): Property purchases
            debt_installments (np.ndarray): Debt payments
            new_loans (np.ndarray): New borrowing
            new_real_wealth (np.ndarray): New real assets
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Updated deposit values
        """
        return (
            current_wealth_in_deposits
            + new_wealth_in_deposits
            - used_up_wealth_in_deposits
            - current_interest_paid
            - price_paid_for_property
            - debt_installments
            + new_loans
            - tau_cf * np.maximum(0.0, new_real_wealth)
        )

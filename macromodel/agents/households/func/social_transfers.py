"""Household social transfer determination implementation.

This module implements social transfer allocation through:
- Transfer amount calculation
- Household-specific distribution
- Model-based predictions
- Equal allocation options

The implementation handles:
- Transfer budget allocation
- Household characteristics
- Model-driven predictions
- Distribution normalization
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class SocialTransfersSetter(ABC):
    """Abstract base class for social transfer allocation.

    Defines interface for determining transfer amounts based on:
    - Total transfer budget
    - Household characteristics
    - Model predictions
    - Distribution rules

    Attributes:
        independents (list[str]): Independent variables for transfer calculation
    """

    def __init__(self, independents: list[str]):
        self.independents = independents

    @abstractmethod
    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        """Calculate household social transfers.

        Args:
            n_households (int): Number of households
            total_other_social_transfers (float): Total transfer budget
            current_independents (np.ndarray): Current independent variables
            initial_independents (np.ndarray): Initial independent variables
            model (Optional[Any]): Prediction model

        Returns:
            np.ndarray: Transfer amounts by household
        """
        pass


class EqualSocialTransfersSetter(SocialTransfersSetter):
    """Simple transfer implementation using equal allocation.

    Distributes total transfer budget equally among all households.
    Used for scenarios where household-specific allocation is not needed.
    """

    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        """Return equal transfer amounts for all households.

        Args:
            n_households (int): Number of households
            total_other_social_transfers (float): Total transfer budget
            current_independents (np.ndarray): Current independent variables
            initial_independents (np.ndarray): Initial independent variables
            model (Optional[Any]): Prediction model

        Returns:
            np.ndarray: Uniform transfer amount array
        """
        return np.full(n_households, total_other_social_transfers / n_households)


class ConstantSocialTransfersSetter(SocialTransfersSetter):
    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        # x = (x - x.min()) / (x.max() - x.min())  # noqa
        initial_independents /= initial_independents.sum(axis=0)
        pred_transfers = model.predict(initial_independents)
        pred_transfers[pred_transfers < 0] = 0.0
        pred_transfers /= np.sum(pred_transfers)
        return pred_transfers * total_other_social_transfers


class DefaultSocialTransfersSetter(SocialTransfersSetter):
    """Default implementation of social transfer allocation.

    Implements transfer determination through:
    - Model-based predictions
    - Variable normalization
    - Distribution adjustment
    """

    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        """Calculate transfers using default behavior.

        Determines transfers through:
        - Variable normalization
        - Model prediction
        - Budget allocation

        Args:
            n_households (int): Number of households
            total_other_social_transfers (float): Total transfer budget
            current_independents (np.ndarray): Current independent variables
            initial_independents (np.ndarray): Initial independent variables
            model (Optional[Any]): Prediction model

        Returns:
            np.ndarray: Transfer amounts by household
        """
        current_independents /= current_independents.sum(axis=0)
        pred_transfers = model.predict(current_independents)
        pred_transfers[pred_transfers < 0] = 0.0
        pred_transfers /= np.sum(pred_transfers)
        return pred_transfers * total_other_social_transfers

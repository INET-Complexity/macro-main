"""Household saving rate determination implementation.

This module implements household saving behavior through:
- Saving rate calculation
- Income-based adjustments
- Model-based predictions
- Average rate application

The implementation handles:
- Household-specific rates
- Income-based variations
- Model-driven predictions
- Rate normalization
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class SavingRatesSetter(ABC):
    """Abstract base class for household saving rate behavior.

    Defines interface for determining saving rates based on:
    - Household characteristics
    - Income levels
    - Model predictions
    - Average rates

    Attributes:
        independents (list[str]): Independent variables for rate calculation
    """

    def __init__(self, independents: list[str]):
        self.independents = independents

    @abstractmethod
    def get_saving_rates(
        self,
        n_households: int,
        average_saving_rate: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        """Calculate household saving rates.

        Args:
            n_households (int): Number of households
            average_saving_rate (float): Average saving rate
            current_independents (np.ndarray): Current independent variables
            initial_independents (np.ndarray): Initial independent variables
            model (Optional[Any]): Prediction model

        Returns:
            np.ndarray: Saving rates by household
        """
        pass


class AverageSavingRatesSetter(SavingRatesSetter):
    """Simple saving rate implementation using average rates.

    Applies the same average saving rate to all households.
    Used for scenarios where household-specific rates are not needed.
    """

    def get_saving_rates(
        self,
        n_households: int,
        average_saving_rate: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        """Return average saving rate for all households.

        Args:
            n_households (int): Number of households
            average_saving_rate (float): Average saving rate
            current_independents (np.ndarray): Current independent variables
            initial_independents (np.ndarray): Initial independent variables
            model (Optional[Any]): Prediction model

        Returns:
            np.ndarray: Uniform saving rate array
        """
        return np.full(n_households, average_saving_rate)


class ConstantSavingRatesSetter(SavingRatesSetter):
    def get_saving_rates(
        self,
        n_households: int,
        average_saving_rate: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        # x = (x - x.min()) / (x.max() - x.min())  # noqa
        initial_independents = initial_independents.astype(float)
        initial_independents /= initial_independents.sum(axis=0)
        pred_sr = model.predict(initial_independents)
        pred_sr[pred_sr > 1.0] = 1.0
        pred_sr[pred_sr < 0.0] = 0.0
        return pred_sr


class DefaultSavingRatesSetter(SavingRatesSetter):
    """Default implementation of household saving rate behavior.

    Implements saving rate determination through:
    - Model-based predictions
    - Variable normalization
    - Rate bounds enforcement
    """

    def get_saving_rates(
        self,
        n_households: int,
        average_saving_rate: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        """Calculate saving rates using default behavior.

        Determines rates through:
        - Variable normalization
        - Model prediction
        - Rate bounding

        Args:
            n_households (int): Number of households
            average_saving_rate (float): Average saving rate
            current_independents (np.ndarray): Current independent variables
            initial_independents (np.ndarray): Initial independent variables
            model (Optional[Any]): Prediction model

        Returns:
            np.ndarray: Saving rates by household
        """
        current_independents /= current_independents.sum(axis=0)
        pred_sr = model.predict(current_independents)
        pred_sr[pred_sr > 1.0] = 1.0
        pred_sr[pred_sr < 0.0] = 0.0
        return pred_sr

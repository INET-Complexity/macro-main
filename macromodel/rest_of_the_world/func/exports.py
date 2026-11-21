"""Rest of the World export determination module.

This module implements various approaches for determining Rest of the World
export supply. It provides several strategies:

1. Autoregressive:
   - Time series based forecasting
   - Historical pattern extrapolation
   - Consistency with target imports

2. Growth-based:
   - Production index adjustments
   - Dynamic supply scaling
   - Growth rate responses

3. Exogenous:
   - Externally provided export paths
   - Historical calibration
   - Direct volume specification

Each approach implements different economic assumptions about how
external supply responds to domestic conditions.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from macromodel.forecaster.forecaster import ImplementedAutoregForecaster  # noqa
from macromodel.forecaster.forecaster import ManualAutoregForecaster


class RoWExportsSetter(ABC):
    """Abstract base class for Rest of World export determination.

    Provides interface for computing export supply based on various
    economic factors and historical patterns.

    Attributes:
        consistency (float): Consistency parameter (0 or 1)
        fixed_total_exports (Optional[np.ndarray]): Pre-computed exports
    """

    def __init__(self, consistency: float):
        """Initialize export setter.

        Args:
            consistency (float): Consistency parameter (must be 0 or 1)
        """
        self.consistency = consistency
        self.fixed_total_exports = None

        assert self.consistency == 0.0 or self.consistency == 1.0

    @abstractmethod
    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        current_time: int,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        """Compute desired export volumes.

        Args:
            historic_total_real_exports (np.ndarray): Past export volumes
            historic_total_real_exports_during (np.ndarray): Calibration exports
            current_time (int): Current period
            initial_desired_exports (np.ndarray): Initial export targets
            model (Optional[Any]): Export forecasting model
            aggregate_country_production_index (float): Production level
            adjustment_speed (float): Response parameter
            assume_zero_noise (bool): Whether to suppress randomness

        Returns:
            np.ndarray: Computed export volumes
        """
        pass


class AutoregressiveRoWExportsSetter(RoWExportsSetter):
    """Autoregressive export determination.

    Uses time series methods to forecast exports based on historical
    patterns, with optional consistency with target imports.
    """

    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        current_time: int,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        """Compute exports using autoregressive forecasting.

        Uses AR(1) model to predict exports, with adjustments for:
        - Zero history cases
        - Target import consistency
        - Initial distribution weighting

        Args:
            historic_total_real_exports (np.ndarray): Past export volumes
            historic_total_real_exports_during (np.ndarray): Calibration exports
            current_time (int): Current period
            initial_desired_exports (np.ndarray): Initial export targets
            model (Optional[Any]): Export forecasting model
            aggregate_country_production_index (float): Production level
            adjustment_speed (float): Response parameter
            assume_zero_noise (bool): Whether to suppress randomness

        Returns:
            np.ndarray: Computed export volumes
        """
        if historic_total_real_exports[-1] == 0.0:
            return np.zeros(initial_desired_exports.shape)

        # Fitting based on target exports
        if self.consistency == 1.0:
            if self.fixed_total_exports is None:
                self.fixed_total_exports = np.exp(
                    ManualAutoregForecaster().forecast(
                        data=np.log(historic_total_real_exports[: -current_time - 1]),
                        t=20,
                        assume_zero_noise=assume_zero_noise,
                    )
                )
            exports = self.fixed_total_exports[current_time]  # check!

        # Fitting based on historic consumption
        else:
            exports = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_real_exports),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        # Weighted by prices
        return np.maximum(
            0.0,
            exports * initial_desired_exports / initial_desired_exports.sum(),
        )


class GrowthRoWExportsSetter(RoWExportsSetter):
    """Growth-based export determination.

    Computes exports based on production index effects on
    initial export targets.
    """

    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        current_time: int,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        """Compute exports using growth effects.

        Adjusts initial export targets based on:
        - Production index deviations
        - Adjustment speed parameter

        Args:
            historic_total_real_exports (np.ndarray): Past export volumes
            historic_total_real_exports_during (np.ndarray): Calibration exports
            current_time (int): Current period
            initial_desired_exports (np.ndarray): Initial export targets
            model (Optional[Any]): Export forecasting model
            aggregate_country_production_index (float): Production level
            adjustment_speed (float): Response parameter
            assume_zero_noise (bool): Whether to suppress randomness

        Returns:
            np.ndarray: Computed export volumes
        """
        return np.maximum(
            0.0,
            (1.0 + adjustment_speed * (aggregate_country_production_index - 1.0)) * initial_desired_exports,
        )


class ExogenousRoWExportsSetter(RoWExportsSetter):
    """Exogenous export determination.

    Uses externally provided export paths with initial distribution weighting.
    """

    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        current_time: int,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        """Compute exports using exogenous paths.

        Uses calibration period exports adjusted for:
        - Initial export distribution

        Args:
            historic_total_real_exports (np.ndarray): Past export volumes
            historic_total_real_exports_during (np.ndarray): Calibration exports
            current_time (int): Current period
            initial_desired_exports (np.ndarray): Initial export targets
            model (Optional[Any]): Export forecasting model
            aggregate_country_production_index (float): Production level
            adjustment_speed (float): Response parameter
            assume_zero_noise (bool): Whether to suppress randomness

        Returns:
            np.ndarray: Computed export volumes
        """
        return (
            historic_total_real_exports_during[current_time] * initial_desired_exports / initial_desired_exports.sum()
        )


"""
class DefaultRoWExportsSetter(RoWExportsSetter):
    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        if model is None:
            return initial_desired_exports
        return np.maximum(
            0.0, model.predict([[0]])[0] * initial_desired_exports
        )
"""

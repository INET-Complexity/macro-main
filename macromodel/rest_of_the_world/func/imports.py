"""Rest of the World import determination module.

This module implements various approaches for determining Rest of the World
import demand. It provides several strategies:

1. Autoregressive:
   - Time series based forecasting
   - Historical pattern extrapolation
   - Consistency with target exports

2. Inflation-based:
   - Price level adjustments
   - Production index effects
   - Dynamic demand scaling

3. Exogenous:
   - Externally provided import paths
   - Historical calibration
   - Direct volume specification

Each approach implements different economic assumptions about how
external demand responds to domestic conditions.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from macromodel.forecaster.forecaster import (
    ImplementedAutoregForecaster,  # noqa
    ManualAutoregForecaster,
)


class RoWImportsSetter(ABC):
    """Abstract base class for Rest of World import determination.

    Provides interface for computing import demand based on various
    economic factors and historical patterns.

    Attributes:
        consistency (float): Consistency parameter (0 or 1)
        fixed_total_imports (Optional[np.ndarray]): Pre-computed imports
    """

    def __init__(self, consistency: float):
        """Initialize import setter.

        Args:
            consistency (float): Consistency parameter (must be 0 or 1)
        """
        self.consistency = max(0.0, min(1.0, consistency))
        self.fixed_total_imports = None

        assert self.consistency == 0.0 or self.consistency == 1.0

    @abstractmethod
    def compute_imports(
        self,
        historic_total_real_imports: np.ndarray,
        historic_total_real_imports_during: np.ndarray,
        current_time: int,
        initial_desired_imports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        """Compute desired import volumes.

        Args:
            historic_total_real_imports (np.ndarray): Past import volumes
            historic_total_real_imports_during (np.ndarray): Calibration imports
            current_time (int): Current period
            initial_desired_imports (np.ndarray): Initial import targets
            model (Optional[Any]): Import forecasting model
            aggregate_country_production_index (float): Production level
            aggregate_country_price_index (float): Price level
            adjustment_speed (float): Response parameter
            assume_zero_noise (bool): Whether to suppress randomness

        Returns:
            np.ndarray: Computed import volumes
        """
        pass


class AutoregressiveRoWImportsSetter(RoWImportsSetter):
    """Autoregressive import determination.

    Uses time series methods to forecast imports based on historical
    patterns, with optional consistency with target exports.
    """

    def compute_imports(
        self,
        historic_total_real_imports: np.ndarray,
        historic_total_real_imports_during: np.ndarray,
        current_time: int,
        initial_desired_imports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        """Compute imports using autoregressive forecasting.

        Uses AR(1) model to predict imports, with adjustments for:
        - Zero history cases
        - Target export consistency
        - Price level effects

        Args:
            historic_total_real_imports (np.ndarray): Past import volumes
            historic_total_real_imports_during (np.ndarray): Calibration imports
            current_time (int): Current period
            initial_desired_imports (np.ndarray): Initial import targets
            model (Optional[Any]): Import forecasting model
            aggregate_country_production_index (float): Production level
            aggregate_country_price_index (float): Price level
            adjustment_speed (float): Response parameter
            assume_zero_noise (bool): Whether to suppress randomness

        Returns:
            np.ndarray: Computed import volumes
        """
        if historic_total_real_imports[-1] == 0.0:
            return np.zeros(initial_desired_imports.shape)

        # Fitting based on target exports
        if self.consistency == 1.0:
            if self.fixed_total_imports is None:
                self.fixed_total_imports = np.exp(
                    ManualAutoregForecaster().forecast(
                        data=np.log(historic_total_real_imports[: -current_time - 1]),
                        t=20,
                        assume_zero_noise=assume_zero_noise,
                    )
                )
            imports = self.fixed_total_imports[current_time]  # check!

        # Fitting based on historic consumption
        else:
            imports = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_real_imports),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        # Weighted by prices
        return np.maximum(
            0.0,
            (1.0 + adjustment_speed * (aggregate_country_price_index - 1.0))
            * imports
            * initial_desired_imports
            / initial_desired_imports.sum(),
        )


class InflationRoWImportsSetter(RoWImportsSetter):
    """Inflation-based import determination.

    Computes imports based on price and production index effects on
    initial import targets.
    """

    def compute_imports(
        self,
        historic_total_real_imports: np.ndarray,
        historic_total_real_imports_during: np.ndarray,
        current_time: int,
        initial_desired_imports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        """Compute imports using price and production effects.

        Adjusts initial import targets based on:
        - Production index deviations
        - Price level changes
        - Adjustment speed parameter

        Args:
            historic_total_real_imports (np.ndarray): Past import volumes
            historic_total_real_imports_during (np.ndarray): Calibration imports
            current_time (int): Current period
            initial_desired_imports (np.ndarray): Initial import targets
            model (Optional[Any]): Import forecasting model
            aggregate_country_production_index (float): Production level
            aggregate_country_price_index (float): Price level
            adjustment_speed (float): Response parameter
            assume_zero_noise (bool): Whether to suppress randomness

        Returns:
            np.ndarray: Computed import volumes
        """
        return np.maximum(
            0.0,
            (1.0 + adjustment_speed * (aggregate_country_production_index - 1.0))
            * (1.0 + adjustment_speed * (aggregate_country_price_index - 1.0))
            * initial_desired_imports,
        )


class ExogenousRoWImportsSetter(RoWImportsSetter):
    """Exogenous import determination.

    Uses externally provided import paths with price level adjustments.
    """

    def compute_imports(
        self,
        historic_total_real_imports: np.ndarray,
        historic_total_real_imports_during: np.ndarray,
        current_time: int,
        initial_desired_imports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        """Compute imports using exogenous paths.

        Uses calibration period imports adjusted for:
        - Current price levels
        - Initial import distribution

        Args:
            historic_total_real_imports (np.ndarray): Past import volumes
            historic_total_real_imports_during (np.ndarray): Calibration imports
            current_time (int): Current period
            initial_desired_imports (np.ndarray): Initial import targets
            model (Optional[Any]): Import forecasting model
            aggregate_country_production_index (float): Production level
            aggregate_country_price_index (float): Price level
            adjustment_speed (float): Response parameter
            assume_zero_noise (bool): Whether to suppress randomness

        Returns:
            np.ndarray: Computed import volumes
        """
        return (
            aggregate_country_price_index
            * historic_total_real_imports_during[current_time]
            * initial_desired_imports
            / initial_desired_imports.sum()
        )

"""Rest of the World price determination module.

This module implements approaches for determining Rest of the World
prices in international markets. It provides mechanisms for:

1. Price Setting:
   - Initial price adjustment
   - Domestic price level response
   - Dynamic price convergence

2. Price Dynamics:
   - Inflation-based updating
   - Speed of adjustment
   - Price floor enforcement

The module implements inflation-based price setting that ensures
price level convergence while maintaining positive prices.
"""

from abc import ABC, abstractmethod

import numpy as np


class RoWPriceSetter(ABC):
    """Abstract base class for Rest of World price determination.

    Provides interface for computing ROW prices based on domestic
    price levels and adjustment parameters.
    """

    @abstractmethod
    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        """Compute ROW prices.

        Args:
            initial_price (np.ndarray): Base prices
            aggregate_country_price_index (float): Domestic price level
            adjustment_speed (float): Price adjustment parameter

        Returns:
            np.ndarray: Computed ROW prices
        """
        pass


class InflationRoWPriceSetter(RoWPriceSetter):
    """Inflation-based price determination implementation.

    Adjusts prices based on domestic price level changes while
    ensuring a minimum positive price level.
    """

    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        """Compute prices using inflation adjustment.

        Adjusts initial prices based on:
        - Domestic price level deviations
        - Adjustment speed parameter
        - Minimum price floor (0.001)

        Args:
            initial_price (np.ndarray): Base prices
            aggregate_country_price_index (float): Domestic price level
            adjustment_speed (float): Price adjustment parameter

        Returns:
            np.ndarray: Adjusted prices with minimum floor
        """
        return np.maximum(
            1e-3,
            (1.0 + adjustment_speed * (aggregate_country_price_index - 1.0)) * initial_price,
        )

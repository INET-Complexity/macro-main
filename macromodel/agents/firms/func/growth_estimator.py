from abc import ABC, abstractmethod

import numpy as np


class GrowthEstimator(ABC):
    """Abstract base class for estimating firm growth rates.

    This class defines strategies for calculating firm-specific growth rates
    based on market conditions including:
    - Price positioning relative to sector averages
    - Supply-demand balance
    - Market share dynamics

    Growth estimates are used to inform:
    - Production planning
    - Capacity decisions
    - Investment strategies
    """

    @abstractmethod
    def compute_growth(
        self,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
    ) -> np.ndarray:
        """Calculate growth rates for each firm based on market conditions.

        Determines appropriate growth rates considering:
        - Price competitiveness within sectors
        - Supply-demand imbalances
        - Market positioning

        Args:
            prev_average_good_prices (np.ndarray): Previous period's average
                prices by sector
            prev_firm_prices (np.ndarray): Previous period's prices by firm
            prev_supply (np.ndarray): Previous period's supply by firm
            prev_demand (np.ndarray): Previous period's demand by firm
            current_firm_sectors (np.ndarray): Sector ID for each firm

        Returns:
            np.ndarray: Estimated growth rates by firm
        """
        pass


class DefaultGrowthEstimator(GrowthEstimator):
    """Default implementation of firm growth estimation.

    This class implements a strategy that determines growth based on:
    1. Price position relative to sector average
    2. Supply-demand balance
    3. Market dynamics

    Growth is positive when:
    - Firm price >= sector average AND demand > supply
    - Firm price < sector average AND demand <= supply

    The magnitude of growth/decline is based on the demand/supply ratio.
    """

    def compute_growth(
        self,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
    ) -> np.ndarray:
        """Calculate growth rates using the default market-based strategy.

        The method:
        1. Maps sector average prices to firms
        2. Identifies firms with favorable market positions
        3. Calculates growth rates based on demand/supply ratios

        Growth is allowed when either:
        - High price (>= sector avg) and excess demand
        - Low price (< sector avg) and excess supply

        Args:
            prev_average_good_prices (np.ndarray): Previous period's average
                prices by sector
            prev_firm_prices (np.ndarray): Previous period's prices by firm
            prev_supply (np.ndarray): Previous period's supply by firm
            prev_demand (np.ndarray): Previous period's demand by firm
            current_firm_sectors (np.ndarray): Sector ID for each firm

        Returns:
            np.ndarray: Growth rates by firm, where positive values indicate
                expansion and negative values indicate contraction
        """
        average_price_by_firm = prev_average_good_prices[current_firm_sectors]
        firm_growth_rates = np.zeros_like(prev_firm_prices)
        ind_canvas = np.logical_or(
            np.logical_and(
                prev_supply <= prev_demand,
                prev_firm_prices >= average_price_by_firm,
            ),
            np.logical_and(
                prev_supply > prev_demand,
                prev_firm_prices < average_price_by_firm,
            ),
        )
        firm_growth_rates[ind_canvas] = (
            np.divide(
                prev_demand[ind_canvas],
                prev_supply[ind_canvas],
                out=np.ones_like(prev_demand[ind_canvas]),
                where=prev_supply[ind_canvas] != 0.0,
            )
            - 1.0
        )
        return firm_growth_rates

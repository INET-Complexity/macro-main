"""Rest of the World inflation module.

This module implements approaches for determining Rest of the World
inflation rates. It provides mechanisms for:

1. Inflation Determination:
   - PPI-based estimation
   - International price convergence
   - Global inflation dynamics

2. Price Level Adjustment:
   - Domestic-foreign price linkages
   - Inflation pass-through
   - Price level convergence

The module currently implements a default approach that assumes
international price level convergence through PPI inflation rates.
"""

from abc import ABC, abstractmethod


class RoWInflationSetter(ABC):
    """Abstract base class for Rest of World inflation determination.

    Provides interface for computing ROW inflation rates based on
    domestic price developments.
    """

    @abstractmethod
    def compute_inflation(self, average_country_ppi_inflation: float) -> float:
        """Compute ROW inflation rate.

        Args:
            average_country_ppi_inflation (float): Average domestic PPI inflation

        Returns:
            float: Computed ROW inflation rate
        """
        pass


class DefaultRoWInflationSetter(RoWInflationSetter):
    """Default inflation determination implementation.

    Assumes international price level convergence by setting ROW
    inflation equal to average domestic PPI inflation.
    """

    def compute_inflation(self, average_country_ppi_inflation: float) -> float:
        """Compute ROW inflation from PPI.

        Sets ROW inflation equal to average domestic PPI inflation,
        implementing price level convergence.

        Args:
            average_country_ppi_inflation (float): Average domestic PPI inflation

        Returns:
            float: ROW inflation rate equal to PPI inflation
        """
        return average_country_ppi_inflation

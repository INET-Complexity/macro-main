"""Rest of the World excess demand module.

This module implements strategies for determining excess demand constraints
in international trade. It provides two approaches:

1. Zero Excess Demand:
   - No excess demand allowed
   - Strict market clearing
   - Trade balance enforcement

2. Infinite Excess Demand:
   - Unconstrained excess demand
   - Flexible market adjustment
   - Demand-driven trade

The choice of excess demand regime affects how international markets
clear and how trade imbalances are handled.
"""

from abc import ABC, abstractmethod

import numpy as np


class ExcessDemandSetter(ABC):
    """Abstract base class for excess demand determination.

    Provides interface for setting maximum allowable excess demand
    in international trade markets.
    """

    @abstractmethod
    def set_maximum_excess_demand(
        self,
        n_exporters: int,
    ) -> np.ndarray:
        """Set maximum excess demand values.

        Args:
            n_exporters (int): Number of exporting agents

        Returns:
            np.ndarray: Maximum excess demand per exporter
        """
        pass


class ZeroExcessDemandSetter(ExcessDemandSetter):
    """Zero excess demand implementation.

    Enforces strict market clearing by disallowing any excess demand.
    This ensures trade balance in equilibrium.
    """

    def set_maximum_excess_demand(
        self,
        n_exporters: int,
    ) -> np.ndarray:
        """Set zero excess demand for all exporters.

        Args:
            n_exporters (int): Number of exporting agents

        Returns:
            np.ndarray: Zero-valued array of length n_exporters
        """
        return np.zeros(n_exporters)


class InfinityExcessDemandSetter(ExcessDemandSetter):
    """Infinite excess demand implementation.

    Allows unconstrained excess demand, enabling flexible market
    adjustment and demand-driven trade flows.
    """

    def set_maximum_excess_demand(
        self,
        n_exporters: int,
    ) -> np.ndarray:
        """Set infinite excess demand for all exporters.

        Args:
            n_exporters (int): Number of exporting agents

        Returns:
            np.ndarray: Infinity-valued array of length n_exporters
        """
        return np.full(n_exporters, np.inf)

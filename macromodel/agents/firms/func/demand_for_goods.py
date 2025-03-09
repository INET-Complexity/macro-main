from abc import ABC, abstractmethod

import numpy as np


class DemandSetter(ABC):
    """Abstract base class for calculating realized demand for firms' products.

    This class defines strategies for determining actual demand by combining:
    - Realized sales (quantities actually sold in the market)
    - Excess demand (additional quantities that could not be satisfied)

    The total demand calculation helps firms understand their true market position,
    including both fulfilled and unfulfilled demand, which is crucial for:
    - Production planning
    - Capacity decisions
    - Market share analysis
    """

    @abstractmethod
    def compute_demand(
        self,
        sell_real: np.ndarray,
        excess_demand: np.ndarray,
    ) -> np.ndarray:
        """Calculate total realized demand for each firm's products.

        Args:
            sell_real (np.ndarray): Actual quantities sold by each firm
            excess_demand (np.ndarray): Additional quantities demanded but not fulfilled
                due to capacity constraints or stock limitations

        Returns:
            np.ndarray: Total demand for each firm's products, including both
                       fulfilled (sales) and unfulfilled (excess) demand
        """
        pass


class DefaultDemandSetter(DemandSetter):
    """Default implementation of demand calculation.

    This class implements a simple additive strategy that:
    1. Takes actual sales as the base demand
    2. Adds excess demand to represent total market interest
    3. Provides firms with a complete picture of their market position

    The total demand figure represents the maximum quantity that could have
    been sold if all demand could have been satisfied.
    """

    def compute_demand(
        self,
        sell_real: np.ndarray,
        excess_demand: np.ndarray,
    ) -> np.ndarray:
        """Calculate total demand using the default additive strategy.

        Simply sums actual sales and excess demand to get total market demand.
        This represents the full market interest in each firm's products,
        regardless of whether that demand could be satisfied.

        Args:
            sell_real (np.ndarray): Actual quantities sold by each firm
            excess_demand (np.ndarray): Unfulfilled demand quantities

        Returns:
            np.ndarray: Total demand as the sum of actual sales and excess demand
        """
        return sell_real + excess_demand

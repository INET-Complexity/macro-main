from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BoughtGoodsDistributor(ABC):
    """Abstract base class for distributing purchased goods between intermediate inputs and capital investments.

    This class defines the interface for strategies that determine how firms allocate their
    purchased goods between intermediate inputs (used in current production) and capital
    investments (used for future production capacity).

    Different implementations can prioritize intermediate inputs, capital investments,
    or use other allocation strategies based on economic conditions and firm preferences.
    """

    @abstractmethod
    def distribute_bought_goods(
        self,
        desired_intermediate_inputs: np.ndarray,
        desired_investment: np.ndarray,
        buy_real: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Distribute purchased goods between intermediate inputs and capital investments.

        Args:
            desired_intermediate_inputs (np.ndarray): Target quantities of intermediate inputs
                Shape (n_firms, n_industries) representing desired input quantities for each firm
            desired_investment (np.ndarray): Target quantities of capital investments
                Shape (n_firms, n_industries) representing desired investment quantities
            buy_real (np.ndarray): Actual quantities of goods purchased
                Shape (n_firms, n_industries) representing actual purchases to distribute

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing:
                1. Quantities allocated to intermediate inputs (n_firms, n_industries)
                2. Quantities allocated to capital investment (n_firms, n_industries)
        """
        pass


class BoughtGoodsDistributorIIPrio(BoughtGoodsDistributor):
    """Distributes purchased goods with priority given to intermediate inputs.

    This implementation prioritizes intermediate inputs over capital investments.
    It first satisfies intermediate input demands up to the available purchases,
    then allocates any remaining quantities to capital investments.

    This strategy reflects a preference for maintaining current production capacity
    over expanding future capacity.
    """

    def __init__(self):
        """Initialize the intermediate-input-priority distributor."""
        pass

    def distribute_bought_goods(
        self,
        desired_intermediate_inputs: np.ndarray,
        desired_investment: np.ndarray,
        buy_real: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Distribute goods with priority to intermediate inputs.

        First allocates goods to satisfy intermediate input demands, then uses any
        remaining quantities for capital investment.

        Args:
            desired_intermediate_inputs (np.ndarray): Target intermediate input quantities
            desired_investment (np.ndarray): Target capital investment quantities
            buy_real (np.ndarray): Actual quantities purchased to distribute

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                1. Allocated intermediate inputs (min of desired and available)
                2. Allocated capital investments (remaining after intermediate inputs)
        """
        return (
            np.minimum(desired_intermediate_inputs, buy_real),
            np.maximum(0.0, buy_real - desired_intermediate_inputs),
        )


class BoughtGoodsDistributorEvenly(BoughtGoodsDistributor):
    """Distributes purchased goods proportionally between inputs and investments.

    This implementation allocates purchased goods in proportion to their desired quantities,
    ensuring a balanced distribution between maintaining current production capacity
    (intermediate inputs) and expanding future capacity (capital investments).
    """

    def __init__(self):
        """Initialize the proportional distributor."""
        pass

    def distribute_bought_goods(
        self,
        desired_intermediate_inputs: np.ndarray,
        desired_investment: np.ndarray,
        buy_real: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Distribute goods proportionally based on desired quantities.

        Allocates purchased goods in proportion to the relative magnitudes of
        desired intermediate inputs and desired investments.

        Args:
            desired_intermediate_inputs (np.ndarray): Target intermediate input quantities
            desired_investment (np.ndarray): Target capital investment quantities
            buy_real (np.ndarray): Actual quantities purchased to distribute

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                1. Allocated intermediate inputs (proportional share)
                2. Allocated capital investments (remaining quantity)
        """
        real_intermediate_inputs = (
            desired_intermediate_inputs / (desired_intermediate_inputs + desired_investment) * buy_real
        )
        return (
            real_intermediate_inputs,
            (buy_real - real_intermediate_inputs),
        )

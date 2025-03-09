from abc import ABC, abstractmethod

import numpy as np


class FirmProfitsSetter(ABC):
    """Abstract base class for estimating future firm profits.

    This class defines strategies for projecting profits based on:
    - Current profit levels
    - Expected economic growth
    - Expected inflation rates

    The profit estimation process considers:
    - Real growth in business activity
    - Nominal price level changes
    - Firm-specific performance
    """

    @abstractmethod
    def compute_estimated_profits(
        self,
        current_profits: np.ndarray,
        estimated_growth: float,
        estimated_inflation: float,
    ) -> np.ndarray:
        """Calculate estimated future profits for each firm.

        Projects profits forward considering both real growth
        and nominal price changes.

        Args:
            current_profits (np.ndarray): Current profit levels by firm
            estimated_growth (float): Expected real growth rate
            estimated_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Estimated future profits by firm
        """
        pass


class DefaultFirmProfitsSetter(FirmProfitsSetter):
    """Default implementation of profit estimation.

    This class implements a simple projection that:
    - Applies real growth to current profits
    - Adjusts for expected inflation
    - Assumes uniform growth across firms
    - Maintains relative profit differentials
    """

    def compute_estimated_profits(
        self,
        current_profits: np.ndarray,
        estimated_growth: float,
        estimated_inflation: float,
    ) -> np.ndarray:
        """Calculate profit projections using simple growth model.

        Applies both real growth and inflation adjustments to
        current profits using the formula:
        future_profits = current_profits * (1 + g) * (1 + π)
        where g is real growth and π is inflation

        Args:
            current_profits (np.ndarray): Current profit levels by firm
            estimated_growth (float): Expected real growth rate
            estimated_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Projected future profits by firm
        """
        return (1 + estimated_growth) * (1 + estimated_inflation) * current_profits

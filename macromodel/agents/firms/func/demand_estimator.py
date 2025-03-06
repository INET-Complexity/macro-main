from abc import ABC, abstractmethod

import numpy as np


class DemandEstimator(ABC):
    """Abstract base class for estimating future demand for firms' products.

    This class defines strategies for projecting future demand based on:
    - Previous period demand
    - Overall economic growth expectations
    - Firm-specific growth estimates

    The estimation process uses two adjustment speeds:
    1. Sectoral: How quickly firms adjust to overall economic growth
    2. Firm-specific: How quickly firms adjust to their individual growth prospects

    Attributes:
        sectoral_growth_adjustment_speed (float): Rate at which firms adjust to overall economic growth
            Values closer to 1 mean faster adjustment to sectoral trends
        firm_growth_adjustment_speed (float): Rate at which firms adjust to firm-specific growth
            Values closer to 1 mean faster adjustment to individual conditions
            Clipped to range [0,1]
    """

    def __init__(
        self,
        sectoral_growth_adjustment_speed: float,
        firm_growth_adjustment_speed: float,
    ):
        """Initialize the demand estimator with adjustment speeds.

        Args:
            sectoral_growth_adjustment_speed (float): Speed of adjustment to overall economic growth
            firm_growth_adjustment_speed (float): Speed of adjustment to firm-specific growth
                Will be clipped to range [0,1]
        """
        self.sectoral_growth_adjustment_speed = sectoral_growth_adjustment_speed
        self.firm_growth_adjustment_speed = max(0.0, min(1.0, firm_growth_adjustment_speed))
        self.firm_growth_adjustment_speed = firm_growth_adjustment_speed

    @abstractmethod
    def compute_estimated_demand(
        self,
        previous_demand: np.ndarray,
        current_estimated_growth: float,
        estimated_growth_by_firm: np.ndarray,
    ) -> np.ndarray:
        """Calculate estimated future demand for each firm.

        Args:
            previous_demand (np.ndarray): Previous period demand for each firm
            current_estimated_growth (float): Overall economic growth rate estimate
            estimated_growth_by_firm (np.ndarray): Firm-specific growth rate estimates

        Returns:
            np.ndarray: Estimated future demand for each firm
        """
        pass


class DefaultDemandEstimator(DemandEstimator):
    """Default implementation of demand estimation.

    This class implements a demand estimation strategy that:
    1. Starts with previous period demand
    2. Adjusts for overall economic growth at the sectoral adjustment speed
    3. Further adjusts for firm-specific growth at the firm adjustment speed

    The final estimate combines both macro and micro level growth expectations
    in a multiplicative fashion.
    """

    def compute_estimated_demand(
        self,
        previous_demand: np.ndarray,
        current_estimated_growth: float,
        estimated_growth_by_firm: np.ndarray,
    ) -> np.ndarray:
        """Calculate estimated demand using the default strategy.

        Computes future demand as:
        previous_demand * (1 + sectoral_speed * overall_growth) * (1 + firm_speed * firm_growth)

        This formulation allows for:
        - Base demand from previous period
        - Sectoral growth effects with controlled adjustment speed
        - Firm-specific growth effects with controlled adjustment speed

        Args:
            previous_demand (np.ndarray): Previous period demand for each firm
            current_estimated_growth (float): Overall economic growth rate estimate
            estimated_growth_by_firm (np.ndarray): Firm-specific growth rate estimates

        Returns:
            np.ndarray: Estimated future demand for each firm, incorporating both
                       overall economic conditions and firm-specific factors
        """
        return (
            (1 + self.sectoral_growth_adjustment_speed * current_estimated_growth)
            * (1 + self.firm_growth_adjustment_speed * estimated_growth_by_firm)
            * previous_demand
        )

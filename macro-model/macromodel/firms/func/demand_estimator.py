from abc import ABC, abstractmethod

import numpy as np


class DemandEstimator(ABC):
    def __init__(
        self,
        sectoral_growth_adjustment_speed: float,
        firm_growth_adjustment_speed: float,
    ):
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
        pass


class DefaultDemandEstimator(DemandEstimator):
    def compute_estimated_demand(
        self,
        previous_demand: np.ndarray,
        current_estimated_growth: float,
        estimated_growth_by_firm: np.ndarray,
    ) -> np.ndarray:
        return (
            (1 + self.sectoral_growth_adjustment_speed * current_estimated_growth)
            * (1 + self.firm_growth_adjustment_speed * estimated_growth_by_firm)
            * previous_demand
        )

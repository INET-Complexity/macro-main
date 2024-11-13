from abc import ABC, abstractmethod

import numpy as np


class FirmProfitsSetter(ABC):
    @abstractmethod
    def compute_estimated_profits(
        self,
        current_profits: np.ndarray,
        estimated_growth: float,
        estimated_inflation: float,
    ) -> np.ndarray:
        pass


class DefaultFirmProfitsSetter(FirmProfitsSetter):
    def compute_estimated_profits(
        self,
        current_profits: np.ndarray,
        estimated_growth: float,
        estimated_inflation: float,
    ) -> np.ndarray:
        return (1 + estimated_growth) * (1 + estimated_inflation) * current_profits

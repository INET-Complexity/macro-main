import numpy as np

from abc import abstractmethod, ABC


class BankProfitsSetter(ABC):
    @abstractmethod
    def compute_estimated_profits(
        self,
        current_profits: np.ndarray,
        estimated_growth: float,
        estimated_inflation: float,
    ) -> np.ndarray:
        pass


class DefaultBankProfitsSetter(BankProfitsSetter):
    def compute_estimated_profits(
        self,
        current_profits: np.ndarray,
        estimated_growth: float,
        estimated_inflation: float,
    ) -> np.ndarray:
        return (
            (1 + estimated_growth) * (1 + estimated_inflation) * current_profits
        )

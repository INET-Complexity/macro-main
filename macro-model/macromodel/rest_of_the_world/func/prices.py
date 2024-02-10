import numpy as np
from abc import abstractmethod, ABC


class RoWPriceSetter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_price(
        self,
        previous_price: np.ndarray,
        previous_row_inflation: float,
    ) -> np.ndarray:
        pass


class ConstantRoWPriceSetter(RoWPriceSetter):
    def compute_price(
        self,
        previous_price: np.ndarray,
        previous_row_inflation: float,
    ) -> np.ndarray:
        return previous_price


class InflationRoWPriceSetter(RoWPriceSetter):
    def compute_price(
        self,
        previous_price: np.ndarray,
        previous_row_inflation: float,
    ) -> np.ndarray:
        return (1 + previous_row_inflation) * previous_price

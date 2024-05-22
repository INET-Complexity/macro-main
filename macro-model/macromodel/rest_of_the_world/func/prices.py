from abc import abstractmethod, ABC

import numpy as np


class RoWPriceSetter(ABC):
    @abstractmethod
    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        pass


class InflationRoWPriceSetter(RoWPriceSetter):
    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        return np.maximum(
            1e-3,
            (1.0 + adjustment_speed * (aggregate_country_price_index - 1.0))
            * initial_price,
        )

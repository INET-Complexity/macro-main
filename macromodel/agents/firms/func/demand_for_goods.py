from abc import ABC, abstractmethod

import numpy as np


class DemandSetter(ABC):
    @abstractmethod
    def compute_demand(
        self,
        sell_real: np.ndarray,
        excess_demand: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultDemandSetter(DemandSetter):
    def compute_demand(
        self,
        sell_real: np.ndarray,
        excess_demand: np.ndarray,
    ) -> np.ndarray:
        return sell_real + excess_demand

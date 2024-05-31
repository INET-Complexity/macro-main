import numpy as np

from abc import abstractmethod, ABC


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

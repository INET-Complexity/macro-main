import numpy as np
from abc import abstractmethod, ABC


class DesiredLabourSetter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_desired_labour(
        self,
        current_desired_production: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultDesiredLabourSetter(DesiredLabourSetter):
    def compute_desired_labour(
        self,
        current_desired_production: np.ndarray,
    ) -> np.ndarray:
        return current_desired_production

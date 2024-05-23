from abc import abstractmethod, ABC

import numpy as np


class DesiredLabourSetter(ABC):
    def __init__(
        self,
        consider_intermediate_inputs: float,
        consider_capital_inputs: float,
    ):
        self.consider_intermediate_inputs = max(0.0, min(1.0, consider_intermediate_inputs))
        self.consider_capital_inputs = max(0.0, min(1.0, consider_capital_inputs))

    @abstractmethod
    def compute_desired_labour(
        self,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultDesiredLabourSetter(DesiredLabourSetter):
    def compute_desired_labour(
        self,
        current_target_production: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
    ) -> np.ndarray:
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.consider_intermediate_inputs * (current_limiting_intermediate_inputs - current_target_production),
        )
        current_target_production = np.minimum(
            current_target_production,
            current_target_production
            + self.consider_capital_inputs * (current_limiting_capital_inputs - current_target_production),
        )
        return current_target_production

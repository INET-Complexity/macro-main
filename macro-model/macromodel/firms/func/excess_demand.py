from abc import ABC, abstractmethod

import numpy as np


class ExcessDemandSetter(ABC):
    def __init__(
        self,
        consider_intermediate_inputs: float,
        consider_capital_inputs: float,
        consider_labour_inputs: float,
    ):
        self.consider_intermediate_inputs = max(0.0, min(1.0, consider_intermediate_inputs))
        self.consider_intermediate_inputs = consider_intermediate_inputs
        self.consider_capital_inputs = max(0.0, min(1.0, consider_capital_inputs))
        self.consider_capital_inputs = consider_capital_inputs
        self.consider_labour_inputs = max(0.0, min(1.0, consider_labour_inputs))
        self.consider_labour_inputs = consider_labour_inputs

    @abstractmethod
    def set_maximum_excess_demand(
        self,
        current_production: np.ndarray,
        target_production: np.ndarray,
        limiting_intermediate_inputs: np.ndarray,
        limiting_capital_inputs: np.ndarray,
        limiting_labour_inputs: np.ndarray,
    ) -> np.ndarray:
        pass


class ConstrainedExcessDemandSetter(ExcessDemandSetter):
    def set_maximum_excess_demand(
        self,
        current_production: np.ndarray,
        target_production: np.ndarray,
        limiting_intermediate_inputs: np.ndarray,
        limiting_capital_inputs: np.ndarray,
        limiting_labour_inputs: np.ndarray,
    ) -> np.ndarray:
        target_production = np.minimum(
            target_production,
            target_production
            + self.consider_labour_inputs * ((limiting_labour_inputs - current_production) - target_production),
        )
        target_production = np.minimum(
            target_production,
            target_production
            + self.consider_intermediate_inputs
            * ((limiting_intermediate_inputs - current_production) - target_production),
        )
        target_production = np.minimum(
            target_production,
            target_production
            + self.consider_capital_inputs * ((limiting_capital_inputs - current_production) - target_production),
        )

        return target_production


class ZeroExcessDemandSetter(ExcessDemandSetter):
    def set_maximum_excess_demand(
        self,
        current_production: np.ndarray,
        target_production: np.ndarray,
        limiting_intermediate_inputs: np.ndarray,
        limiting_capital_inputs: np.ndarray,
        limiting_labour_inputs: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(current_production.shape)

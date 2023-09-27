import numpy as np

from abc import abstractmethod, ABC

from typing import Any, Optional


class GovernmentConsumptionSetter(ABC):
    @abstractmethod
    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        pass


class DefaultGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        if model is None:
            return previous_desired_government_consumption
        return model.predict([[0]])[0] * previous_desired_government_consumption

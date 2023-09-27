import numpy as np

from abc import abstractmethod, ABC

from typing import Any, Optional


class RoWExportsSetter(ABC):
    @abstractmethod
    def compute_exports(
        self,
        previous_desired_exports: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        pass


class DefaultRoWExportsSetter(RoWExportsSetter):
    def compute_exports(
        self,
        previous_desired_exports: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        if model is None:
            return previous_desired_exports
        return model.predict([[0]])[0] * previous_desired_exports

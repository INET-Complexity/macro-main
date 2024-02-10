import numpy as np

from abc import abstractmethod, ABC

from typing import Any, Optional


class RoWImportsSetter(ABC):
    @abstractmethod
    def compute_imports(
        self,
        previous_desired_imports: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        pass


class ConstantRoWImportsSetter(RoWImportsSetter):
    def compute_imports(
        self,
        previous_desired_imports: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        return previous_desired_imports


class DefaultRoWImportsSetter(RoWImportsSetter):
    def compute_imports(
        self,
        previous_desired_imports: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        if model is None:
            return previous_desired_imports
        return model.predict([[0]])[0] * previous_desired_imports

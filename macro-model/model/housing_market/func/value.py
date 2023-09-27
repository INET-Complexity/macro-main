import numpy as np

from abc import abstractmethod, ABC


class PropertyValueSetter(ABC):
    @abstractmethod
    def compute_value(self, current_property_values: np.ndarray) -> np.ndarray:
        pass


class DefaultPropertyValueSetter(PropertyValueSetter):
    def __init__(self, random_fluctuation_std: float):
        self.random_fluctuation_std = random_fluctuation_std

    def compute_value(self, current_property_values: np.ndarray) -> np.ndarray:
        return (
            1 + np.random.normal(0.0, self.random_fluctuation_std, current_property_values.shape)
        ) * current_property_values

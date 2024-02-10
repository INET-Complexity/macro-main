import numpy as np

from abc import abstractmethod, ABC


class RentSetter(ABC):
    def __init__(
        self,
        partial_rent_inflation_indexation: float,
        new_property_rent_markup: float,
        offered_rent_decrease: float,
    ):
        self.partial_rent_inflation_indexation = partial_rent_inflation_indexation
        self.new_property_rent_markup = new_property_rent_markup
        self.offered_rent_decrease = offered_rent_decrease

    @abstractmethod
    def compute_rent(
        self,
        current_rent: np.ndarray,
        historic_inflation: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_offered_rent_for_new_properties(
        self,
        property_value: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_offered_rent_for_existing_properties(self, current_offered_rent: np.ndarray) -> np.ndarray:
        pass


class ConstantRentSetter(RentSetter):
    def compute_rent(self, current_rent: np.ndarray, historic_inflation: np.ndarray) -> np.ndarray:
        return current_rent

    def compute_offered_rent_for_new_properties(
        self, property_value: np.ndarray, observed_fraction_rent_value: np.ndarray
    ) -> np.ndarray:
        return property_value

    def compute_offered_rent_for_existing_properties(self, current_offered_rent: np.ndarray) -> np.ndarray:
        return current_offered_rent


class DefaultRentSetter(RentSetter):
    def compute_rent(
        self,
        current_rent: np.ndarray,
        historic_inflation: np.ndarray,
    ) -> np.ndarray:
        return (
            1
            + np.maximum(
                0.0,
                self.partial_rent_inflation_indexation * historic_inflation[-1],
            )
        ) * current_rent

    def compute_offered_rent_for_new_properties(
        self,
        property_value: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
    ) -> np.ndarray:
        return (1 + self.new_property_rent_markup) * (
            observed_fraction_rent_value[0] * property_value + observed_fraction_rent_value[1]
        )

    def compute_offered_rent_for_existing_properties(self, current_offered_rent: np.ndarray) -> np.ndarray:
        return (1 - self.offered_rent_decrease) * current_offered_rent

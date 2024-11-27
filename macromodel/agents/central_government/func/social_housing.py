from abc import ABC, abstractmethod

import numpy as np


class SocialHousing(ABC):
    def __init__(self, rent_as_fraction_of_unemployment_rate: float):
        self.rent_as_fraction_of_unemployment_rate = rent_as_fraction_of_unemployment_rate

    @abstractmethod
    def compute_social_housing_rent(
        self,
        current_unemployment_benefits_by_individual: float,
        current_household_size: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultSocialHousing(SocialHousing):
    def compute_social_housing_rent(
        self,
        current_unemployment_benefits_by_individual: float,
        current_household_size: np.ndarray,
    ) -> np.ndarray:
        return (
            self.rent_as_fraction_of_unemployment_rate
            * current_unemployment_benefits_by_individual
            * current_household_size
        )

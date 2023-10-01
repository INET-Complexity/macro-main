import numpy as np

from abc import abstractmethod, ABC

from typing import Any, Optional


class SocialBenefitsSetter(ABC):
    @abstractmethod
    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_cpi_inflation: np.ndarray,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        pass

    @abstractmethod
    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_cpi_inflation: np.ndarray,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        pass


class DefaultSocialBenefitsSetter(SocialBenefitsSetter):
    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_cpi_inflation: np.ndarray,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        if model is None:
            return prev_unemployment_benefits
        pred = model.predict(np.array([[historic_cpi_inflation[-1], current_unemployment_rate]]))[0]
        return pred * prev_unemployment_benefits

    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_cpi_inflation: np.ndarray,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        if model is None:
            return prev_regular_transfer_to_households
        pred = model.predict(np.array([[historic_cpi_inflation[-1], current_unemployment_rate]]))[0]
        return pred * prev_regular_transfer_to_households

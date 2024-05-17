import numpy as np

from abc import abstractmethod, ABC

from typing import Any, Optional


class SocialBenefitsSetter(ABC):
    @abstractmethod
    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        pass

    @abstractmethod
    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        pass


class ConstantSocialBenefitsSetter(SocialBenefitsSetter):
    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        return prev_unemployment_benefits

    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        return prev_regular_transfer_to_households


class GrowthSocialBenefitsSetter(SocialBenefitsSetter):
    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        return max(1.0, 1 / (1 + current_estimated_growth)) * prev_unemployment_benefits

    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        return (1 + current_estimated_growth) * prev_regular_transfer_to_households


class DefaultSocialBenefitsSetter(SocialBenefitsSetter):
    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        if model is None:
            return prev_unemployment_benefits
        pred = model.predict(np.array([[historic_ppi_inflation[-1], current_unemployment_rate]]))[0]
        return pred

    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        if model is None:
            return prev_regular_transfer_to_households
        pred = model.predict(np.array([[historic_ppi_inflation[-1], current_unemployment_rate]]))[0]
        return pred

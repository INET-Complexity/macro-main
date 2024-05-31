import numpy as np

from abc import abstractmethod, ABC


class FinancialAssets(ABC):
    def __init__(self, income_from_fa_noise_std: float):
        self.income_from_fa_noise_std = income_from_fa_noise_std

    @abstractmethod
    def compute_expected_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultFinancialAssets(FinancialAssets):
    def compute_expected_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        return income_coefficient * current_other_financial_assets

    def compute_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        return (
            (
                1
                + np.random.normal(
                    0.0,
                    self.income_from_fa_noise_std,
                    initial_other_financial_assets.shape,
                )
            )
            * income_coefficient
            * current_other_financial_assets
        )

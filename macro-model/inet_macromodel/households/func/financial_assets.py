import numpy as np

from abc import abstractmethod, ABC


class FinancialAssets(ABC):
    @abstractmethod
    def compute_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        pass


class ConstantFinancialAssets(FinancialAssets):
    def compute_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        return income_coefficient * initial_other_financial_assets


class DefaultFinancialAssets(FinancialAssets):
    def compute_income(
        self,
        income_coefficient: float,
        initial_other_financial_assets: np.ndarray,
        current_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        return income_coefficient * current_other_financial_assets

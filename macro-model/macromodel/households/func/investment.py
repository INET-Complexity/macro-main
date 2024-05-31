import numpy as np

from abc import abstractmethod, ABC


class HouseholdInvestment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        income: np.ndarray,
        exogenous_total_investment: np.ndarray,
        current_time: int,
        investment_weights: np.ndarray,
        investment_rate: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        pass


class NoHouseholdInvestment(HouseholdInvestment):
    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        income: np.ndarray,
        exogenous_total_investment: np.ndarray,
        current_time: int,
        investment_weights: np.ndarray,
        investment_rate: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        return np.zeros((income.shape[0], investment_weights.shape[0]))


class DefaultHouseholdInvestment(HouseholdInvestment):
    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        income: np.ndarray,
        exogenous_total_investment: np.ndarray,
        current_time: int,
        investment_weights: np.ndarray,
        investment_rate: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        return 1.0 / (1 + tau_cf) * np.outer(investment_weights, investment_rate * income).T


class ExogenousHouseholdInvestment(HouseholdInvestment):
    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        income: np.ndarray,
        exogenous_total_investment: np.ndarray,
        current_time: int,
        investment_weights: np.ndarray,
        investment_rate: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        target_investment = np.maximum(
            0.0,
            (1.0 / (1 + tau_cf) * np.outer(investment_weights, investment_rate * income).T),
        )
        return (
            (1 + expected_inflation)
            * current_cpi
            / initial_cpi
            * 1.0
            / (1 + tau_cf)
            * exogenous_total_investment[current_time]
            * target_investment
            / target_investment.sum()
        )

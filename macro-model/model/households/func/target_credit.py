import numpy as np

from abc import abstractmethod, ABC

from typing import Tuple


class HouseholdTargetCredit(ABC):
    def __init__(self, consumption_expansion_quantile: float):
        self.consumption_expansion_quantile = consumption_expansion_quantile

    @abstractmethod
    def compute_target_payday_loans(
        self,
        target_consumption_before_ce: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_consumption_expansion_loans(
        self,
        current_income: np.ndarray,
        initial_income: np.ndarray,
        current_wealth_other_real_assets: np.ndarray,
        initial_wealth_other_real_assets: np.ndarray,
        target_consumption_before_ce: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def compute_target_mortgage(
        self,
        target_house_price: np.ndarray,
        target_consumption_before_ce: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultHouseholdTargetCredit(HouseholdTargetCredit):
    def compute_target_payday_loans(
        self,
        target_consumption_before_ce: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(
            0.0,
            target_consumption_before_ce.sum(axis=1) - (income - rent) - wealth_in_financial_assets,
        )

    def compute_consumption_expansion_loans(
        self,
        current_income: np.ndarray,
        initial_income: np.ndarray,
        current_wealth_other_real_assets: np.ndarray,
        initial_wealth_other_real_assets: np.ndarray,
        target_consumption_before_ce: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        additional_target_consumption = np.zeros_like(current_income)
        target_consumption_expansion_loans = np.zeros_like(current_income)

        # Households attempting to expand their consumption
        div = np.divide(
            current_wealth_other_real_assets,
            current_income,
            out=np.zeros_like(current_wealth_other_real_assets),
            where=current_income != 0.0,
        )
        quant = np.quantile(div, self.consumption_expansion_quantile)
        ind = np.logical_and(div < quant, current_income > 0.0)

        # Amount of additional purchase
        additional_target_consumption[ind] = np.maximum(
            0.0,
            current_income[ind] / initial_income[ind] * initial_wealth_other_real_assets[ind]
            - current_wealth_other_real_assets[ind],
        )

        # Target loans
        target_consumption_expansion_loans[ind] = np.maximum(
            0.0,
            additional_target_consumption[ind]
            - np.maximum(
                0.0,
                wealth_in_financial_assets[ind]
                - (target_consumption_before_ce.sum(axis=1)[ind] - (income[ind] - rent[ind])),
            ),
        )

        return additional_target_consumption, target_consumption_expansion_loans

    def compute_target_mortgage(
        self,
        target_house_price: np.ndarray,
        target_consumption_before_ce: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(
            0.0,
            target_house_price
            - np.maximum(
                0.0,
                wealth_in_financial_assets - (target_consumption_before_ce.sum(axis=1) - (income - rent)),
            ),
        )

from abc import ABC, abstractmethod

import numpy as np


class HouseholdTargetCredit(ABC):
    def __init__(
        self,
        down_payment_fraction: float,
    ):
        self.down_payment_fraction = down_payment_fraction

    @abstractmethod
    def compute_target_consumption_loans(
        self,
        target_consumption: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_target_mortgage(
        self,
        target_house_price: np.ndarray,
        target_consumption: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultHouseholdTargetCredit(HouseholdTargetCredit):
    def compute_target_consumption_loans(
        self,
        target_consumption: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(
            0.0,
            target_consumption.sum(axis=1) - (income - rent) - wealth_in_financial_assets,
        )

    def compute_target_mortgage(
        self,
        target_house_price: np.ndarray,
        target_consumption: np.ndarray,
        income: np.ndarray,
        rent: np.ndarray,
        wealth_in_financial_assets: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(
            0.0,
            target_house_price
            - self.down_payment_fraction
            * np.maximum(
                0.0,
                wealth_in_financial_assets - (target_consumption.sum(axis=1) - (income - rent)),
            ),
        )

import numpy as np

from inet_macromodel.timeseries import TimeSeries

from abc import abstractmethod, ABC

from typing import Tuple, Optional, Any


class WealthSetter(ABC):
    def __init__(self, other_real_assets_depreciation_rate: float):
        self.other_real_assets_depreciation_rate = other_real_assets_depreciation_rate

    @abstractmethod
    def distribute_new_wealth(
        self,
        new_wealth: np.ndarray,
        model: Optional[Any],
        independents: list[str],
        ts: TimeSeries,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def use_up_wealth(
        self,
        used_up_wealth: np.ndarray,
        current_wealth_in_deposits: np.ndarray,
        current_wealth_in_other_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def compute_wealth_in_other_real_assets(
        self,
        current_wealth_in_other_real_assets: np.ndarray,
        current_investment_in_other_real_assets: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_wealth_in_other_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        new_wealth_in_other_financial_assets: np.ndarray,
        used_up_wealth_in_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def compute_wealth_in_deposits(
        current_wealth_in_deposits: np.ndarray,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        current_interest_paid: np.ndarray,
        price_paid_for_property: np.ndarray,
        debt_installments: np.ndarray,
        new_loans: np.ndarray,
        new_real_wealth: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        pass


class DefaultWealthSetter(WealthSetter):
    def distribute_new_wealth(
        self,
        new_wealth: np.ndarray,
        model: Optional[Any],
        independents: list[str],
        ts: TimeSeries,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.stack(
            [ts.current(ind.lower()) for ind in independents],
            axis=1,
        )
        # x = (x - x.min()) / (x.max() - x.min())  # noqa
        x /= x.sum(axis=0)
        pred_deposit_fraction = model.predict(x)
        pred_deposit_fraction[pred_deposit_fraction > 1.0] = 1.0
        pred_deposit_fraction[pred_deposit_fraction < 0.0] = 0.0

        return pred_deposit_fraction * new_wealth, (1 - pred_deposit_fraction) * new_wealth

    def use_up_wealth(
        self,
        used_up_wealth: np.ndarray,
        current_wealth_in_deposits: np.ndarray,
        current_wealth_in_other_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        used_up_wealth_in_other_financial_assets = np.minimum(current_wealth_in_other_financial_assets, used_up_wealth)
        used_up_wealth_in_deposits = np.minimum(
            current_wealth_in_deposits,
            used_up_wealth - used_up_wealth_in_other_financial_assets,
        )
        return (
            used_up_wealth_in_deposits,
            used_up_wealth_in_other_financial_assets,
        )

    def compute_wealth_in_other_real_assets(
        self,
        current_wealth_in_other_real_assets: np.ndarray,
        current_investment_in_other_real_assets: np.ndarray,
    ) -> np.ndarray:
        return (
            1 - self.other_real_assets_depreciation_rate
        ) * current_wealth_in_other_real_assets + current_investment_in_other_real_assets

    def compute_wealth_in_other_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        new_wealth_in_other_financial_assets: np.ndarray,
        used_up_wealth_in_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        return (
            current_wealth_in_other_financial_assets
            + new_wealth_in_other_financial_assets
            - used_up_wealth_in_other_financial_assets
        )

    @staticmethod
    def compute_wealth_in_deposits(
        current_wealth_in_deposits: np.ndarray,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        current_interest_paid: np.ndarray,
        price_paid_for_property: np.ndarray,
        debt_installments: np.ndarray,
        new_loans: np.ndarray,
        new_real_wealth: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        return (
            current_wealth_in_deposits
            + new_wealth_in_deposits
            - used_up_wealth_in_deposits
            - current_interest_paid
            - price_paid_for_property
            - debt_installments
            + new_loans
            - tau_cf * np.maximum(0.0, new_real_wealth)
        )

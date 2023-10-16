import numpy as np

from abc import abstractmethod, ABC

from typing import Any, Optional


class SavingRatesSetter(ABC):
    def __init__(self, independents: list[str]):
        self.independents = independents

    @abstractmethod
    def get_saving_rates(
        self,
        n_households: int,
        average_saving_rate: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        pass


class AverageSavingRatesSetter(SavingRatesSetter):
    def get_saving_rates(
        self,
        n_households: int,
        average_saving_rate: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        return np.full(n_households, average_saving_rate)


class ConstantSavingRatesSetter(SavingRatesSetter):
    def get_saving_rates(
        self,
        n_households: int,
        average_saving_rate: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        # x = (x - x.min()) / (x.max() - x.min())  # noqa
        initial_independents = initial_independents.astype(float)
        initial_independents /= initial_independents.sum(axis=0)
        pred_sr = model.predict(initial_independents)
        pred_sr[pred_sr > 1.0] = 1.0
        pred_sr[pred_sr < 0.0] = 0.0
        return pred_sr


class DefaultSavingRatesSetter(SavingRatesSetter):
    def get_saving_rates(
        self,
        n_households: int,
        average_saving_rate: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        # x = (x - x.min()) / (x.max() - x.min())  # noqa
        current_independents /= current_independents.sum(axis=0)
        pred_sr = model.predict(current_independents)
        pred_sr[pred_sr > 1.0] = 1.0
        pred_sr[pred_sr < 0.0] = 0.0
        return pred_sr

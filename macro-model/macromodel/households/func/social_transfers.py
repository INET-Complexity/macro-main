from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class SocialTransfersSetter(ABC):
    def __init__(self, independents: list[str]):
        self.independents = independents

    @abstractmethod
    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        pass


class EqualSocialTransfersSetter(SocialTransfersSetter):
    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        return np.full(n_households, total_other_social_transfers / n_households)


class ConstantSocialTransfersSetter(SocialTransfersSetter):
    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        # x = (x - x.min()) / (x.max() - x.min())  # noqa
        initial_independents /= initial_independents.sum(axis=0)
        pred_transfers = model.predict(initial_independents)
        pred_transfers[pred_transfers < 0] = 0.0
        pred_transfers /= np.sum(pred_transfers)
        return pred_transfers * total_other_social_transfers


class DefaultSocialTransfersSetter(SocialTransfersSetter):
    def get_social_transfers(
        self,
        n_households: int,
        total_other_social_transfers: float,
        current_independents: np.ndarray,
        initial_independents: np.ndarray,
        model: Optional[Any],
    ) -> np.ndarray:
        # x = (x - x.min()) / (x.max() - x.min())  # noqa
        current_independents /= current_independents.sum(axis=0)
        pred_transfers = model.predict(current_independents)
        pred_transfers[pred_transfers < 0] = 0.0
        pred_transfers /= np.sum(pred_transfers)
        return pred_transfers * total_other_social_transfers

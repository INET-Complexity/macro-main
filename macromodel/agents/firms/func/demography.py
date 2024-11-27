from abc import ABC, abstractmethod

import numpy as np


class FirmDemography(ABC):
    @abstractmethod
    def handle_firm_insolvency(
        self,
        current_firm_is_insolvent: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        pass


class NoFirmDemography(FirmDemography):
    def handle_firm_insolvency(
        self,
        current_firm_is_insolvent: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        return np.full(current_firm_is_insolvent.shape, False)


class DefaultFirmDemography(FirmDemography):
    def handle_firm_insolvency(
        self,
        current_firm_is_insolvent: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_deposits: np.ndarray,
    ) -> None:
        return np.logical_and(current_firm_equity < 0, current_firm_deposits < 0)

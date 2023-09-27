import numpy as np

from abc import abstractmethod, ABC


class FirmDemography(ABC):
    @abstractmethod
    def handle_firm_insolvency(
        self,
        current_firm_is_insolvent: np.ndarray,
        current_firm_debts: np.ndarray,
        current_firm_deposits: np.ndarray,
    ) -> None:
        pass


class NoFirmDemography(FirmDemography):
    def handle_firm_insolvency(
        self,
        current_firm_is_insolvent: np.ndarray,
        current_firm_debts: np.ndarray,
        current_firm_deposits: np.ndarray,
    ) -> None:
        pass


class DefaultFirmDemography(FirmDemography):
    def handle_firm_insolvency(
        self,
        current_firm_is_insolvent: np.ndarray,
        current_firm_debts: np.ndarray,
        current_firm_deposits: np.ndarray,
    ) -> None:
        insolvent_firms = np.logical_and(current_firm_debts < 0, current_firm_deposits < 0)
        current_firm_is_insolvent[insolvent_firms] = True
        current_firm_debts[insolvent_firms] = 0.0
        current_firm_deposits[insolvent_firms] = 0.0

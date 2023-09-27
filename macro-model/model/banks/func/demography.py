import numpy as np

from abc import abstractmethod, ABC


class BankDemography(ABC):
    @abstractmethod
    def handle_bank_insolvency(
        self,
        current_bank_equity: np.ndarray,
        current_bank_deposits: np.ndarray,
        is_insolvent: np.ndarray,
    ) -> float:
        pass


class NoBankDemography(BankDemography):
    def handle_bank_insolvency(
        self,
        current_bank_equity: np.ndarray,
        current_bank_deposits: np.ndarray,
        is_insolvent: np.ndarray,
    ) -> float:
        return 0.0


class DefaultBankDemography(BankDemography):
    def handle_bank_insolvency(
        self,
        current_bank_equity: np.ndarray,
        current_bank_deposits: np.ndarray,
        is_insolvent: np.ndarray,
    ) -> float:
        insolvent_banks = np.logical_and(current_bank_equity < 0, current_bank_equity + current_bank_deposits < 0)
        is_insolvent[insolvent_banks] = True

        # Compute average equity of non-insolvent banks
        average_equity = np.mean(current_bank_equity[~insolvent_banks])

        # Inject equity
        equity_injection = np.maximum(0.0, average_equity - current_bank_equity[insolvent_banks]).sum()
        current_bank_equity[insolvent_banks] = average_equity
        current_bank_deposits[insolvent_banks] = 0.0

        return equity_injection

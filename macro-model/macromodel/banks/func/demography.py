from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BankDemography(ABC):
    def __init__(self, solvency_ratio: float):
        self.solvency_ratio = solvency_ratio

    @abstractmethod
    def handle_bank_insolvency(
        self,
        current_bank_equity: np.ndarray,
        current_bank_loans: np.ndarray,
        current_bank_deposits: np.ndarray,
        is_insolvent: np.ndarray,
    ) -> Tuple[float, float]:
        pass


class NoBankDemography(BankDemography):
    def handle_bank_insolvency(
        self,
        current_bank_equity: np.ndarray,
        current_bank_loans: np.ndarray,
        current_bank_deposits: np.ndarray,
        is_insolvent: np.ndarray,
    ) -> Tuple[float, float]:
        return 0.0, float(np.mean(current_bank_equity))


class DefaultBankDemography(BankDemography):
    def handle_bank_insolvency(
        self,
        current_bank_equity: np.ndarray,
        current_bank_loans: np.ndarray,
        current_bank_deposits: np.ndarray,
        is_insolvent: np.ndarray,
    ) -> Tuple[float, float]:
        insolvent_banks = (
            np.divide(
                current_bank_equity,
                current_bank_loans + current_bank_deposits,
                out=np.zeros(current_bank_equity.shape),
                where=current_bank_loans + current_bank_deposits != 0.0,
            )
            < self.solvency_ratio
        )
        is_insolvent[insolvent_banks] = True

        # Compute average equity of non-insolvent banks
        if np.sum(~insolvent_banks) > 0:
            average_equity = float(np.mean(current_bank_equity[~insolvent_banks]))
        else:
            average_equity = float(np.mean(current_bank_equity))
        equity_injection = np.maximum(0.0, average_equity - current_bank_equity[insolvent_banks]).sum()
        return equity_injection, average_equity

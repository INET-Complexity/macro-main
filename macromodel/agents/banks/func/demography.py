"""Bank demography and insolvency management.

This module implements strategies for managing bank demographics and
handling bank insolvency cases through:
- Solvency ratio monitoring
- Insolvency detection
- Equity injection calculation
- Bank restructuring

The strategies consider:
- Bank equity levels
- Loan portfolios
- Deposit bases
- Solvency thresholds
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BankDemography(ABC):
    """Abstract base class for bank demography management.

    This class defines strategies for handling bank demographics and
    insolvency through:
    - Solvency monitoring
    - Insolvency detection
    - Capital adequacy assessment
    - Restructuring decisions

    The strategies consider:
    - Equity to asset ratios
    - Minimum capital requirements
    - System-wide stability
    - Resolution mechanisms

    Attributes:
        solvency_ratio (float): Minimum required equity to assets ratio
    """

    def __init__(self, solvency_ratio: float):
        """Initialize bank demography manager.

        Args:
            solvency_ratio (float): Minimum required equity to assets ratio
        """
        self.solvency_ratio = solvency_ratio

    @abstractmethod
    def handle_bank_insolvency(
        self,
        current_bank_equity: np.ndarray,
        current_bank_loans: np.ndarray,
        current_bank_deposits: np.ndarray,
        is_insolvent: np.ndarray,
    ) -> Tuple[float, float]:
        """Handle insolvent banks.

        Args:
            current_bank_equity (np.ndarray): Current equity by bank
            current_bank_loans (np.ndarray): Current loans by bank
            current_bank_deposits (np.ndarray): Current deposits by bank
            is_insolvent (np.ndarray): Current insolvency status by bank

        Returns:
            Tuple[float, float]: (Total equity injection needed,
                Average equity of solvent banks)
        """
        pass


class NoBankDemography(BankDemography):
    """No-intervention bank demography strategy.

    This class implements a passive approach that:
    - Does not actively monitor solvency
    - Does not intervene in bank failures
    - Only tracks average equity levels
    - Maintains system as is
    """

    def handle_bank_insolvency(
        self,
        current_bank_equity: np.ndarray,
        current_bank_loans: np.ndarray,
        current_bank_deposits: np.ndarray,
        is_insolvent: np.ndarray,
    ) -> Tuple[float, float]:
        """Handle insolvency with no intervention.

        Returns system average equity without any intervention.

        Args:
            [same as parent class]

        Returns:
            Tuple[float, float]: (0.0 for no injection,
                System-wide average equity)
        """
        return 0.0, float(np.mean(current_bank_equity))


class DefaultBankDemography(BankDemography):
    """Default bank demography strategy.

    This class implements active bank monitoring and resolution:
    - Tracks equity to asset ratios
    - Identifies insolvent banks
    - Calculates needed equity injections
    - Manages bank restructuring

    The approach:
    - Uses solvency ratio threshold
    - Marks insolvent institutions
    - Computes recapitalization needs
    - Maintains system stability
    """

    def handle_bank_insolvency(
        self,
        current_bank_equity: np.ndarray,
        current_bank_loans: np.ndarray,
        current_bank_deposits: np.ndarray,
        is_insolvent: np.ndarray,
    ) -> Tuple[float, float]:
        """Handle insolvent banks actively.

        Identifies insolvent banks and calculates needed equity
        injections based on:
        - Current equity levels
        - Total assets (loans + deposits)
        - Solvency ratio requirement
        - System-wide average equity

        Args:
            [same as parent class]

        Returns:
            Tuple[float, float]: (Total equity injection needed,
                Average equity of solvent banks)
        """
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

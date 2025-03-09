from abc import ABC, abstractmethod

import numpy as np


class FirmDemography(ABC):
    """Abstract base class for managing firm demographics and survival.

    This class defines strategies for handling firm insolvency and survival in the economy.
    It determines which firms remain active based on their financial health, specifically:
    - Equity position (net worth)
    - Deposit balances (liquidity)

    The demography management is crucial for:
    - Maintaining realistic firm turnover
    - Handling business failures
    - Ensuring market discipline
    - Modeling creative destruction
    """

    @abstractmethod
    def handle_firm_insolvency(
        self,
        current_firm_is_insolvent: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        """Determine which firms become insolvent based on their financial position.

        Args:
            current_firm_is_insolvent (np.ndarray): Current insolvency status of firms
                Boolean array where True indicates already insolvent firms
            current_firm_equity (np.ndarray): Current equity (net worth) of firms
                Negative values indicate technical insolvency
            current_firm_deposits (np.ndarray): Current deposit balances of firms
                Negative values indicate overdrafts/liquidity problems

        Returns:
            np.ndarray: Boolean array indicating which firms are now insolvent
        """
        pass


class NoFirmDemography(FirmDemography):
    """Implementation that prevents any firm failures.

    This class implements a "no failure" policy where firms never become insolvent,
    regardless of their financial position. This can be useful for:
    - Model testing and debugging
    - Analyzing economy without firm turnover
    - Isolating effects of other economic mechanisms
    """

    def handle_firm_insolvency(
        self,
        current_firm_is_insolvent: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        """Always return False for all firms, preventing any insolvencies.

        Args:
            current_firm_is_insolvent (np.ndarray): Ignored in this implementation
            current_firm_equity (np.ndarray): Ignored in this implementation
            current_firm_deposits (np.ndarray): Ignored in this implementation

        Returns:
            np.ndarray: Array of False values, keeping all firms active
        """
        return np.full(current_firm_is_insolvent.shape, False)


class DefaultFirmDemography(FirmDemography):
    """Default implementation of firm demography management.

    This class implements a standard insolvency check where firms fail if they have:
    1. Negative equity (technical insolvency)
    2. Negative deposits (liquidity crisis)

    This represents a realistic approach where firms must maintain both:
    - Long-term solvency (positive net worth)
    - Short-term liquidity (positive cash position)
    """

    def handle_firm_insolvency(
        self,
        current_firm_is_insolvent: np.ndarray,
        current_firm_equity: np.ndarray,
        current_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        """Mark firms as insolvent if they have both negative equity and deposits.

        Implements a dual-condition test where firms must fail both:
        1. Balance sheet test (negative equity)
        2. Cash flow test (negative deposits)

        This approach reflects real-world insolvency criteria where firms
        must be both balance sheet insolvent and unable to pay debts.

        Args:
            current_firm_is_insolvent (np.ndarray): Current insolvency status
            current_firm_equity (np.ndarray): Equity positions
            current_firm_deposits (np.ndarray): Deposit balances

        Returns:
            np.ndarray: Boolean array where True indicates newly insolvent firms
        """
        return np.logical_and(current_firm_equity < 0, current_firm_deposits < 0)

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class TargetCreditSetter(ABC):
    """Abstract base class for determining firms' target credit requirements.

    This class defines strategies for calculating optimal credit demand based on:
    - Estimated deposits (available liquid funds)
    - Unconstrained costs for intermediate inputs (working capital needs)
    - Unconstrained costs for capital inputs (investment needs)

    The credit demand is split into:
    - Short-term credit: primarily for working capital and intermediate inputs
    - Long-term credit: primarily for capital investments
    """

    @abstractmethod
    def compute_target_credit(
        self,
        estimated_deposits: np.ndarray,
        unconstrained_target_intermediate_inputs_costs: np.ndarray,
        unconstrained_target_capital_inputs_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate target short-term and long-term credit for each firm.

        Args:
            estimated_deposits (np.ndarray): Available liquid funds for each firm
            unconstrained_target_intermediate_inputs_costs (np.ndarray): Desired spending
                on intermediate inputs (materials, supplies) without financial constraints
            unconstrained_target_capital_inputs_costs (np.ndarray): Desired spending
                on capital inputs (machinery, equipment) without financial constraints

        Returns:
            Tuple[np.ndarray, np.ndarray]: Target short-term and long-term credit amounts
                First array is short-term credit for working capital
                Second array is long-term credit for investments
        """
        pass


class DefaultTargetCreditSetter(TargetCreditSetter):
    """Default implementation of credit demand calculation.

    This class implements a hierarchical credit demand strategy that:
    1. First allocates deposits to cover intermediate input costs
    2. Any shortfall becomes short-term credit demand
    3. Remaining deposits (if any) are applied to capital input costs
    4. Any remaining shortfall becomes long-term credit demand

    This approach prioritizes working capital needs over investment financing,
    reflecting typical business financial management practices.
    """

    def compute_target_credit(
        self,
        estimated_deposits: np.ndarray,
        unconstrained_target_intermediate_inputs_costs: np.ndarray,
        unconstrained_target_capital_inputs_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate credit demand using the default hierarchical strategy.

        The method follows these steps:
        1. Calculate short-term credit as max(0, intermediate costs - deposits)
        2. Calculate remaining deposits after short-term needs
        3. Calculate long-term credit as max(0, capital costs - remaining deposits)

        Args:
            estimated_deposits (np.ndarray): Available liquid funds for each firm
            unconstrained_target_intermediate_inputs_costs (np.ndarray): Desired intermediate
                input spending without financial constraints
            unconstrained_target_capital_inputs_costs (np.ndarray): Desired capital
                input spending without financial constraints

        Returns:
            Tuple[np.ndarray, np.ndarray]: Target short-term and long-term credit amounts
                First array is short-term credit for working capital
                Second array is long-term credit for investments
        """
        target_short_term_credit = np.maximum(
            0.0,
            unconstrained_target_intermediate_inputs_costs - estimated_deposits,
        )
        target_long_term_credit = np.maximum(
            0.0,
            unconstrained_target_capital_inputs_costs - (estimated_deposits - target_short_term_credit),
        )
        return target_short_term_credit, target_long_term_credit


class SimpleTargetCreditSetter(TargetCreditSetter):
    """Simplified implementation of credit demand calculation.

    This class implements a basic credit demand strategy where:
    - No short-term credit is requested
    - Long-term credit is only requested to cover negative deposits

    This approach is useful for:
    - Model testing and validation
    - Scenarios where firms rely primarily on equity financing
    - Simplified economic models without complex credit markets
    """

    def compute_target_credit(
        self,
        estimated_deposits: np.ndarray,
        unconstrained_target_intermediate_inputs_costs: np.ndarray,
        unconstrained_target_capital_inputs_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate credit demand using the simplified strategy.

        Sets short-term credit to zero and only requests long-term credit
        to cover negative deposits, ignoring actual input costs.

        Args:
            estimated_deposits (np.ndarray): Available liquid funds for each firm
            unconstrained_target_intermediate_inputs_costs (np.ndarray): Desired intermediate
                input spending (unused in this implementation)
            unconstrained_target_capital_inputs_costs (np.ndarray): Desired capital
                input spending (unused in this implementation)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Target short-term and long-term credit amounts
                First array is always zeros (no short-term credit)
                Second array is max(0, -deposits) (minimal long-term credit)
        """
        return np.zeros_like(estimated_deposits), np.maximum(0.0, -estimated_deposits)

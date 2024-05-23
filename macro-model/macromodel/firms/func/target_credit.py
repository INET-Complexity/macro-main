import numpy as np

from abc import abstractmethod, ABC

from typing import Tuple


class TargetCreditSetter(ABC):
    @abstractmethod
    def compute_target_credit(
        self,
        estimated_deposits: np.ndarray,
        unconstrained_target_intermediate_inputs_costs: np.ndarray,
        unconstrained_target_capital_inputs_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class DefaultTargetCreditSetter(TargetCreditSetter):
    def compute_target_credit(
        self,
        estimated_deposits: np.ndarray,
        unconstrained_target_intermediate_inputs_costs: np.ndarray,
        unconstrained_target_capital_inputs_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
    def compute_target_credit(
        self,
        estimated_deposits: np.ndarray,
        unconstrained_target_intermediate_inputs_costs: np.ndarray,
        unconstrained_target_capital_inputs_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(estimated_deposits), np.maximum(0.0, -estimated_deposits)

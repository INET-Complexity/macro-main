from abc import ABC, abstractmethod

import numpy as np


class TargetIntermediateInputsSetter(ABC):
    def __init__(
        self,
        target_intermediate_inputs_fraction: float,
        credit_gap_fraction: float,
    ):
        self.target_intermediate_inputs_fraction = target_intermediate_inputs_fraction
        self.credit_gap_fraction = credit_gap_fraction

    @abstractmethod
    def compute_unconstrained_target_intermediate_inputs(
        self,
        current_target_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        prev_intermediate_inputs_stock: np.ndarray,
        initial_intermediate_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_target_intermediate_inputs(
        self,
        unconstrained_target_intermediate_inputs: np.ndarray,
        target_short_term_credit: np.ndarray,
        received_short_term_credit: np.ndarray,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> np.ndarray:
        pass


class FinancialTargetIntermediateInputsSetter(TargetIntermediateInputsSetter):
    def compute_unconstrained_target_intermediate_inputs(
        self,
        current_target_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        prev_intermediate_inputs_stock: np.ndarray,
        initial_intermediate_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
    ) -> np.ndarray:
        target_intermediate_inputs = np.divide(
            current_target_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros(intermediate_inputs_productivity_matrix.shape),
            where=intermediate_inputs_productivity_matrix != 0.0,
        )

        # Take current stock of intermediate inputs into accounts
        target_intermediate_inputs = np.maximum(
            0.0,
            target_intermediate_inputs
            - self.target_intermediate_inputs_fraction
            * (
                prev_intermediate_inputs_stock
                - (
                    (
                        np.divide(
                            prev_production,
                            initial_production,
                            out=np.zeros(prev_production.shape),
                            where=initial_production != 0.0,
                        )
                    )[:, None]
                    * initial_intermediate_inputs_stock
                )
            ),
        )

        return target_intermediate_inputs

    def compute_target_intermediate_inputs(
        self,
        unconstrained_target_intermediate_inputs: np.ndarray,
        target_short_term_credit: np.ndarray,
        received_short_term_credit: np.ndarray,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> np.ndarray:
        return np.maximum(
            0.0,
            unconstrained_target_intermediate_inputs
            - self.credit_gap_fraction
            * (target_short_term_credit - received_short_term_credit)[:, None]
            / ((1 + expected_inflation) * previous_good_prices),
        )

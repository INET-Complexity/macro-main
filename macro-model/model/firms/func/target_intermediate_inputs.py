import numpy as np

from abc import abstractmethod, ABC


class TargetIntermediateInputsSetter(ABC):
    def __init__(self, target_intermediate_inputs_fraction: float):
        self.target_intermediate_inputs_fraction = target_intermediate_inputs_fraction

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
    ) -> np.ndarray:
        pass


class UnconstrainedTargetIntermediateInputsSetter(TargetIntermediateInputsSetter):
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
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
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
                            out=np.zeros_like(prev_production),
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
    ) -> np.ndarray:
        return unconstrained_target_intermediate_inputs


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
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
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
                            out=np.zeros_like(prev_production),
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
    ) -> np.ndarray:
        return (
            unconstrained_target_intermediate_inputs
            * np.divide(
                received_short_term_credit,
                target_short_term_credit,
                out=np.ones_like(received_short_term_credit),
                where=target_short_term_credit != 0.0,
            )[:, None]
        )

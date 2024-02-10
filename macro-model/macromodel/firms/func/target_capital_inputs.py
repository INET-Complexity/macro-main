import numpy as np

from abc import abstractmethod, ABC


class TargetCapitalInputsSetter(ABC):
    def __init__(self, target_capital_inputs_fraction: float):
        self.target_capital_inputs_fraction = target_capital_inputs_fraction

    @abstractmethod
    def compute_unconstrained_target_capital_inputs(
        self,
        current_target_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_target_capital_inputs(
        self,
        unconstrained_target_capital_inputs: np.ndarray,
        target_long_term_credit: np.ndarray,
        received_long_term_credit: np.ndarray,
    ) -> np.ndarray:
        pass


class UnconstrainedTargetCapitalInputsSetter(TargetCapitalInputsSetter):
    def compute_unconstrained_target_capital_inputs(
        self,
        current_target_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
    ) -> np.ndarray:
        target_capital_inputs = np.multiply(
            current_target_production[:, None],
            capital_inputs_depreciation_matrix,
            out=np.zeros_like(capital_inputs_depreciation_matrix),
        )

        # Take current stock of capital inputs into accounts
        target_capital_inputs = np.maximum(
            0.0,
            target_capital_inputs
            - self.target_capital_inputs_fraction
            * (
                prev_capital_inputs_stock
                - (
                    (
                        np.divide(
                            prev_production,
                            initial_production,
                            out=np.zeros_like(prev_production),
                            where=initial_production != 0.0,
                        )
                    )[:, None]
                    * initial_capital_inputs_stock
                )
            ),
        )

        return target_capital_inputs

    def compute_target_capital_inputs(
        self,
        unconstrained_target_capital_inputs: np.ndarray,
        target_long_term_credit: np.ndarray,
        received_long_term_credit: np.ndarray,
    ) -> np.ndarray:
        return unconstrained_target_capital_inputs


class FinancialTargetCapitalInputsSetter(TargetCapitalInputsSetter):
    def compute_unconstrained_target_capital_inputs(
        self,
        current_target_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
    ) -> np.ndarray:
        target_capital_inputs = np.multiply(
            current_target_production[:, None],
            capital_inputs_depreciation_matrix,
            out=np.zeros_like(capital_inputs_depreciation_matrix),
        )

        # Take current stock of capital inputs into accounts
        target_capital_inputs = np.maximum(
            0.0,
            target_capital_inputs
            - self.target_capital_inputs_fraction
            * (
                prev_capital_inputs_stock
                - (
                    (
                        np.divide(
                            prev_production,
                            initial_production,
                            out=np.zeros_like(prev_production),
                            where=initial_production != 0.0,
                        )
                    )[:, None]
                    * initial_capital_inputs_stock
                )
            ),
        )

        return target_capital_inputs

    def compute_target_capital_inputs(
        self,
        unconstrained_target_capital_inputs: np.ndarray,
        target_long_term_credit: np.ndarray,
        received_long_term_credit: np.ndarray,
    ) -> np.ndarray:
        return (
            unconstrained_target_capital_inputs
            * np.divide(
                received_long_term_credit,
                target_long_term_credit,
                out=np.ones_like(received_long_term_credit),
                where=target_long_term_credit != 0.0,
            )[:, None]
        )

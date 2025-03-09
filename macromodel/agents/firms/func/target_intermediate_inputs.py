from abc import ABC, abstractmethod

import numpy as np


class TargetIntermediateInputsSetter(ABC):
    """Abstract base class for determining firms' target intermediate input requirements.

    This class defines strategies for calculating optimal intermediate input demand based on:
    - Production targets and productivity
    - Current stock levels and historical usage
    - Financial constraints and credit availability

    The calculation is done in two steps:
    1. Computing unconstrained targets based on production needs
    2. Adjusting targets based on financial constraints

    Attributes:
        target_intermediate_inputs_fraction (float): Fraction of existing stock
            considered available for future production
        credit_gap_fraction (float): How much to reduce input targets when
            facing credit constraints (between 0 and 1)
    """

    def __init__(
        self,
        target_intermediate_inputs_fraction: float,
        credit_gap_fraction: float,
    ) -> None:
        """Initialize the target intermediate inputs setter.

        Args:
            target_intermediate_inputs_fraction (float): Fraction of existing stock
                considered available for future production
            credit_gap_fraction (float): How much to reduce input targets when
                facing credit constraints (between 0 and 1)
        """
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
        """Calculate unconstrained target intermediate inputs for each firm.

        This represents the ideal amount of intermediate inputs needed,
        before considering financial constraints.

        Args:
            current_target_production (np.ndarray): Target production levels
            intermediate_inputs_productivity_matrix (np.ndarray): Matrix showing how
                efficiently each input type contributes to production
            prev_intermediate_inputs_stock (np.ndarray): Current stock of
                intermediate inputs by type
            initial_intermediate_inputs_stock (np.ndarray): Initial stock levels
                of intermediate inputs by type
            prev_production (np.ndarray): Previous period's production levels
            initial_production (np.ndarray): Initial production levels

        Returns:
            np.ndarray: Unconstrained target intermediate inputs by firm and type
        """
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
        """Calculate financially constrained target intermediate inputs.

        Adjusts the unconstrained targets based on credit availability
        and financial constraints.

        Args:
            unconstrained_target_intermediate_inputs (np.ndarray): Ideal input
                quantities before financial constraints
            target_short_term_credit (np.ndarray): Desired short-term credit
            received_short_term_credit (np.ndarray): Actually received credit
            previous_good_prices (np.ndarray): Previous period's input prices
            expected_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Financially constrained target intermediate inputs
        """
        pass


class FinancialTargetIntermediateInputsSetter(TargetIntermediateInputsSetter):
    """Implementation of intermediate input targeting with financial considerations.

    This class implements a strategy that:
    1. Calculates base input needs from production targets and productivity
    2. Adjusts for existing stock levels relative to historical usage
    3. Further adjusts based on credit constraints and price expectations

    The approach ensures that input targets are:
    - Sufficient for planned production
    - Efficient in stock management
    - Financially feasible
    """

    def compute_unconstrained_target_intermediate_inputs(
        self,
        current_target_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        prev_intermediate_inputs_stock: np.ndarray,
        initial_intermediate_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
    ) -> np.ndarray:
        """Calculate unconstrained intermediate input targets with stock adjustment.

        The method:
        1. Converts production targets to input needs using productivity matrix
        2. Adjusts for current stock levels relative to historical usage
        3. Ensures non-negative targets

        Args:
            current_target_production (np.ndarray): Target production levels
            intermediate_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients matrix
            prev_intermediate_inputs_stock (np.ndarray): Current stock levels
            initial_intermediate_inputs_stock (np.ndarray): Reference stock levels
            prev_production (np.ndarray): Current production levels
            initial_production (np.ndarray): Reference production levels

        Returns:
            np.ndarray: Unconstrained target intermediate inputs by firm and type
        """
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
        """Calculate financially constrained intermediate input targets.

        Adjusts unconstrained targets downward based on:
        1. Credit gap (difference between desired and received credit)
        2. Expected price changes
        3. Previous period prices

        Args:
            unconstrained_target_intermediate_inputs (np.ndarray): Ideal input
                quantities before financial constraints
            target_short_term_credit (np.ndarray): Desired short-term credit
            received_short_term_credit (np.ndarray): Actually received credit
            previous_good_prices (np.ndarray): Previous period's input prices
            expected_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Financially constrained target intermediate inputs,
                adjusted for credit availability and price expectations
        """
        return np.maximum(
            0.0,
            unconstrained_target_intermediate_inputs
            - self.credit_gap_fraction
            * (target_short_term_credit - received_short_term_credit)[:, None]
            / ((1 + expected_inflation) * previous_good_prices),
        )

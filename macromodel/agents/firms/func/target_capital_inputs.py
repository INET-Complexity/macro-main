from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class TargetCapitalInputsSetter(ABC):
    """Abstract base class for determining firms' capital input targets.

    This class defines strategies for calculating optimal capital input
    demand based on:
    - Production targets and depreciation
    - Current capital stock levels
    - Historical usage patterns
    - Financial constraints

    The capital targeting process considers:
    - Replacement of depreciated capital
    - Capacity expansion needs
    - Credit availability
    - Price expectations

    Attributes:
        target_capital_inputs_fraction (float): Fraction of existing capital
            stock considered available for future production
        credit_gap_fraction (float): How much to reduce capital targets
            when facing credit constraints (between 0 and 1)
    """

    def __init__(self, target_capital_inputs_fraction: float, credit_gap_fraction: float):
        """Initialize the target capital inputs setter.

        Args:
            target_capital_inputs_fraction (float): Fraction of existing capital
                stock considered available for future production
            credit_gap_fraction (float): How much to reduce capital targets
                when facing credit constraints (between 0 and 1)
        """
        self.target_capital_inputs_fraction = target_capital_inputs_fraction
        self.credit_gap_fraction = credit_gap_fraction

    @abstractmethod
    def compute_unconstrained_target_capital_inputs(
        self,
        current_target_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
        previous_good_prices: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
        extra_taxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate unconstrained target capital inputs for each firm.

        Determines ideal capital input needs before considering
        financial constraints, based on:
        - Production targets
        - Depreciation rates
        - Current stock levels
        - Historical usage patterns

        Args:
            current_target_production (np.ndarray): Target production levels
            capital_inputs_depreciation_matrix (np.ndarray): Depreciation
                rates for different capital types
            prev_capital_inputs_stock (np.ndarray): Current capital stock
                by type
            initial_capital_inputs_stock (np.ndarray): Initial capital stock
                levels by type
            prev_production (np.ndarray): Previous production levels
            initial_production (np.ndarray): Initial production levels
            previous_good_prices (np.ndarray): Previous period's input prices
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for inputs (not used in Leontief technology)
            extra_taxes (Optional[np.ndarray], optional): Additional taxes on inputs
                that may affect input decisions. Defaults to None.

        Returns:
            np.ndarray: Unconstrained target capital inputs by firm and type
        """
        pass

    @abstractmethod
    def compute_target_capital_inputs(
        self,
        unconstrained_target_capital_inputs: np.ndarray,
        target_long_term_credit: np.ndarray,
        received_long_term_credit: np.ndarray,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> np.ndarray:
        """Calculate financially constrained target capital inputs.

        Adjusts unconstrained targets based on:
        - Credit availability
        - Price expectations
        - Financial constraints

        Args:
            unconstrained_target_capital_inputs (np.ndarray): Ideal capital
                input quantities before financial constraints
            target_long_term_credit (np.ndarray): Desired long-term credit
            received_long_term_credit (np.ndarray): Actually received credit
            previous_good_prices (np.ndarray): Previous period's prices
            expected_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Financially constrained target capital inputs
        """
        pass


class FinancialTargetCapitalInputsSetter(TargetCapitalInputsSetter):
    """Implementation of capital input targeting with financial considerations.

    This class implements a strategy that:
    1. Calculates base capital needs from production and depreciation
    2. Adjusts for existing stock levels relative to historical usage
    3. Further adjusts based on credit constraints and price expectations

    The approach ensures that capital targets are:
    - Sufficient for planned production
    - Efficient in stock management
    - Financially feasible
    """

    def compute_unconstrained_target_capital_inputs(
        self,
        current_target_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
        previous_good_prices: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
        extra_taxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate unconstrained capital input targets with stock adjustment.

        The method:
        1. Calculates base capital needs from production and depreciation
        2. Adjusts for current stock levels relative to historical usage
        3. Ensures non-negative targets

        Args:
            current_target_production (np.ndarray): Target production levels
            capital_inputs_depreciation_matrix (np.ndarray): Depreciation rates
            prev_capital_inputs_stock (np.ndarray): Current stock levels
            initial_capital_inputs_stock (np.ndarray): Reference stock levels
            prev_production (np.ndarray): Current production levels
            initial_production (np.ndarray): Reference production levels
            previous_good_prices (np.ndarray): Previous period's input prices
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for inputs (not used in Leontief technology)
            extra_taxes (Optional[np.ndarray], optional): Additional taxes on inputs
                that may affect input decisions. Defaults to None.

        Returns:
            np.ndarray: Unconstrained target capital inputs by firm and type
        """
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
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> np.ndarray:
        """Calculate financially constrained capital input targets.

        Adjusts unconstrained targets downward based on:
        1. Credit gap (difference between desired and received credit)
        2. Expected price changes
        3. Previous period prices

        Args:
            unconstrained_target_capital_inputs (np.ndarray): Ideal capital
                input quantities before financial constraints
            target_long_term_credit (np.ndarray): Desired long-term credit
            received_long_term_credit (np.ndarray): Actually received credit
            previous_good_prices (np.ndarray): Previous period's prices
            expected_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Financially constrained target capital inputs,
                adjusted for credit availability and price expectations
        """
        return np.maximum(
            0.0,
            unconstrained_target_capital_inputs
            - self.credit_gap_fraction
            * (target_long_term_credit - received_long_term_credit)[:, None]
            / ((1 + expected_inflation) * previous_good_prices),
        )


class BundleWeightedTargetCapitalInputsSetter(FinancialTargetCapitalInputsSetter):
    """Implementation of capital input targeting with bundle-based weighting.

    This class extends the financial targeting approach by applying weights to the
    unconstrained targets based on:
    - Input prices
    - Depreciation coefficients
    - Substitution bundles

    The weighting mechanism allows for substitution between capital inputs within the same bundle
    based on relative prices and depreciation rates.
    """

    def __init__(
        self,
        target_capital_inputs_fraction: float,
        credit_gap_fraction: float,
        beta: float = 1.0,
    ) -> None:
        """Initialize the bundle-weighted target capital inputs setter.

        Args:
            target_capital_inputs_fraction (float): Fraction of existing capital
                stock considered available for future production
            credit_gap_fraction (float): How much to reduce capital targets when
                facing credit constraints (between 0 and 1)
            beta (float, optional): Parameter controlling the sensitivity of weights
                to price and depreciation differences. Defaults to 1.0.
        """
        super().__init__(target_capital_inputs_fraction, credit_gap_fraction)
        self.beta = beta

    def compute_unconstrained_target_capital_inputs(
        self,
        current_target_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        prev_capital_inputs_stock: np.ndarray,
        initial_capital_inputs_stock: np.ndarray,
        prev_production: np.ndarray,
        initial_production: np.ndarray,
        previous_good_prices: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
        extra_taxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate bundle-weighted unconstrained capital input targets.

        The method:
        1. Computes base unconstrained targets using the parent class method
        2. Calculates weights based on prices, depreciation, and bundles
        3. Applies the weights to the unconstrained targets

        Args:
            current_target_production (np.ndarray): Target production levels
            capital_inputs_depreciation_matrix (np.ndarray): Depreciation rates
            prev_capital_inputs_stock (np.ndarray): Current stock levels
            initial_capital_inputs_stock (np.ndarray): Reference stock levels
            prev_production (np.ndarray): Current production levels
            initial_production (np.ndarray): Reference production levels
            previous_good_prices (np.ndarray): Previous period's input prices
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for inputs
            extra_taxes (Optional[np.ndarray], optional): Additional taxes on inputs
                that may affect input decisions. Defaults to None.

        Returns:
            np.ndarray: Bundle-weighted unconstrained target capital inputs
        """

        if extra_taxes is None:
            extra_taxes = np.zeros_like(previous_good_prices)

        # Get base unconstrained targets from parent class
        base_targets = super().compute_unconstrained_target_capital_inputs(
            current_target_production,
            capital_inputs_depreciation_matrix,
            prev_capital_inputs_stock,
            initial_capital_inputs_stock,
            prev_production,
            initial_production,
            previous_good_prices,
            substitution_bundle_matrix,
            extra_taxes,
        )

        # Calculate average price (using nanmean to handle potential NaN values)
        avg_price = np.nanmean(previous_good_prices)

        # Calculate unnormalized weights
        # exp(-beta / avg_price * price[j] / depreciation_matrix[i,j])
        unnormalized_weights = np.exp(
            -self.beta / avg_price * (previous_good_prices + extra_taxes) / capital_inputs_depreciation_matrix
        )

        # Create bundle matrix C = M*M.transpose() and replace non-zero coefficients with 1
        n_industries = substitution_bundle_matrix.shape[0]
        bundle_matrix = np.zeros((n_industries, n_industries))

        # Compute C = M*M.transpose()
        bundle_matrix = np.dot(substitution_bundle_matrix, substitution_bundle_matrix.T)

        # Replace non-zero coefficients with 1
        bundle_matrix = np.where(bundle_matrix > 0, 1, 0)

        # Calculate normalization factors for each firm and input
        # sum_l b[i,l] * unnormalized_weights[i,l]
        normalization_factors = np.einsum("jl, il -> ij", bundle_matrix, unnormalized_weights)

        # Calculate normalized weights
        normalized_weights = np.divide(
            unnormalized_weights,
            normalization_factors,
            out=np.ones_like(unnormalized_weights),
            where=normalization_factors != 0,
        )

        # Apply weights to base targets
        weighted_targets = base_targets * normalized_weights

        return weighted_targets

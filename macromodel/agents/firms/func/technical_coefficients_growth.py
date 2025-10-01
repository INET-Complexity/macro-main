"""Technical coefficients productivity growth module.

This module implements mechanisms for improving the productivity of technical coefficients
(intermediate and capital input productivity matrices) through targeted investment.
Unlike TFP which affects all inputs uniformly, technical coefficient improvements are
input-specific, allowing firms to strategically improve efficiency for particular inputs.

Architecture note:
- Base technical coefficients are [n_industries x n_industries] matrices (shared by industry)
- Each firm has multipliers [n_firms x n_industries] that modify these base coefficients
- Effective coefficient for firm i in industry k using input j: base[k,j] * multiplier[i,j]
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class TechnicalCoefficientsGrowth(ABC):
    """Abstract base class for computing technical coefficients productivity growth.

    This class defines strategies for calculating how firms' input productivity
    multipliers improve over time based on targeted investments. The multipliers
    modify the base industry-level coefficients to create firm-specific productivity.

    Attributes:
        investment_effectiveness (float): How effectively investment translates to improvements (ψ)
        diminishing_returns_factor (float): Rate of diminishing returns (δ)
        cumulative_intermediate_improvements (np.ndarray): Tracks cumulative improvements
            for intermediate inputs to implement diminishing returns [n_firms x n_industries]
        cumulative_capital_improvements (np.ndarray): Tracks cumulative improvements
            for capital inputs to implement diminishing returns [n_firms x n_industries]
    """

    def __init__(
        self,
        n_firms: int,
        n_industries: int,
        investment_effectiveness: float = 0.1,
        diminishing_returns_factor: float = 0.5,
        **kwargs
    ):
        """Initialize technical coefficients growth tracker.

        Args:
            n_firms (int): Number of firms
            n_industries (int): Number of industries/input types
            investment_effectiveness (float): How effectively investment translates to improvements (ψ)
                Default: 0.1 (10% effectiveness)
            diminishing_returns_factor (float): Rate of diminishing returns (δ)
                Default: 0.5 (moderate diminishing returns)
            **kwargs: Additional parameters for specific implementations
        """
        # Store parameters as attributes
        self.investment_effectiveness = investment_effectiveness
        self.diminishing_returns_factor = diminishing_returns_factor

        # Initialize cumulative improvement trackers (per firm, per input type)
        self.cumulative_intermediate_improvements = np.zeros((n_firms, n_industries))
        self.cumulative_capital_improvements = np.zeros((n_firms, n_industries))

    @abstractmethod
    def compute_intermediate_multiplier_growth(
        self,
        current_multipliers: np.ndarray,
        base_coefficients: np.ndarray,
        firm_industries: np.ndarray,
        technical_investment: np.ndarray,
        production: np.ndarray,
        prices: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Calculate growth rates for intermediate input multipliers.

        Args:
            current_multipliers (np.ndarray): Current firm multipliers [n_firms x n_industries]
            base_coefficients (np.ndarray): Base industry coefficients [n_industries x n_industries]
            firm_industries (np.ndarray): Industry index for each firm [n_firms]
            technical_investment (np.ndarray): Investment in each input's productivity [n_firms x n_industries]
            production (np.ndarray): Current production levels [n_firms]
            prices (np.ndarray): Current input prices [n_industries]
            **kwargs: Additional parameters for specific implementations

        Returns:
            np.ndarray: Growth rates for each multiplier [n_firms x n_industries]
        """
        pass

    @abstractmethod
    def compute_capital_multiplier_growth(
        self,
        current_multipliers: np.ndarray,
        base_coefficients: np.ndarray,
        firm_industries: np.ndarray,
        technical_investment: np.ndarray,
        production: np.ndarray,
        prices: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Calculate growth rates for capital input multipliers.

        Args:
            current_multipliers (np.ndarray): Current firm multipliers [n_firms x n_industries]
            base_coefficients (np.ndarray): Base industry coefficients [n_industries x n_industries]
            firm_industries (np.ndarray): Industry index for each firm [n_firms]
            technical_investment (np.ndarray): Investment in each input's productivity [n_firms x n_industries]
            production (np.ndarray): Current production levels [n_firms]
            prices (np.ndarray): Current input prices [n_industries]
            **kwargs: Additional parameters for specific implementations

        Returns:
            np.ndarray: Growth rates for each multiplier [n_firms x n_industries]
        """
        pass

    def update_cumulative_improvements(
        self, intermediate_growth: np.ndarray, capital_growth: np.ndarray
    ) -> None:
        """Update cumulative improvement trackers after growth is applied.

        Args:
            intermediate_growth (np.ndarray): Applied growth rates for intermediate multipliers [n_firms x n_industries]
            capital_growth (np.ndarray): Applied growth rates for capital multipliers [n_firms x n_industries]
        """
        self.cumulative_intermediate_improvements += intermediate_growth
        self.cumulative_capital_improvements += capital_growth

    @staticmethod
    def update_multipliers(current_multipliers: np.ndarray, growth_rates: np.ndarray) -> np.ndarray:
        """Update multiplier matrix based on growth rates.

        Args:
            current_multipliers (np.ndarray): Current multiplier matrix [n_firms x n_industries]
            growth_rates (np.ndarray): Growth rates to apply [n_firms x n_industries]

        Returns:
            np.ndarray: Updated multiplier matrix
        """
        return current_multipliers * (1 + growth_rates)



class NoOpTechnicalGrowth(TechnicalCoefficientsGrowth):
    """No-operation technical growth implementation (static multipliers).

    This class keeps all multipliers at 1.0 (no improvements).
    Useful as a default when technical growth is not desired but the
    interface needs to be satisfied.
    """

    def __init__(self, n_firms: int, n_industries: int, **kwargs):
        """Initialize NoOpTechnicalGrowth (ignores all parameters).

        Args:
            n_firms (int): Number of firms
            n_industries (int): Number of industries/input types
            **kwargs: Additional parameters (all ignored)
        """
        super().__init__(n_firms, n_industries)
        # NoOp doesn't need any parameters

    def compute_intermediate_multiplier_growth(
        self,
        current_multipliers: np.ndarray,
        base_coefficients: np.ndarray,
        firm_industries: np.ndarray,
        technical_investment: np.ndarray,
        production: np.ndarray,
        prices: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Return zero growth (static multipliers).

        Args:
            current_multipliers (np.ndarray): Current multipliers (ignored)
            base_coefficients (np.ndarray): Base coefficients (ignored)
            firm_industries (np.ndarray): Firm industries (ignored)
            technical_investment (np.ndarray): Investment amounts (ignored)
            production (np.ndarray): Production levels (ignored)
            prices (np.ndarray): Input prices (ignored)
            **kwargs: Additional parameters (all ignored)

        Returns:
            np.ndarray: Zero growth rates [n_firms x n_industries]
        """
        return np.zeros_like(current_multipliers)

    def compute_capital_multiplier_growth(
        self,
        current_multipliers: np.ndarray,
        base_coefficients: np.ndarray,
        firm_industries: np.ndarray,
        technical_investment: np.ndarray,
        production: np.ndarray,
        prices: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Return zero growth (static multipliers).

        Args:
            current_multipliers (np.ndarray): Current multipliers (ignored)
            base_coefficients (np.ndarray): Base coefficients (ignored)
            firm_industries (np.ndarray): Firm industries (ignored)
            technical_investment (np.ndarray): Investment amounts (ignored)
            production (np.ndarray): Production levels (ignored)
            prices (np.ndarray): Input prices (ignored)
            **kwargs: Additional parameters (all ignored)

        Returns:
            np.ndarray: Zero growth rates [n_firms x n_industries]
        """
        return np.zeros_like(current_multipliers)


class SimpleTechnicalGrowth(TechnicalCoefficientsGrowth):
    """Simple technical coefficients growth implementation.

    Implements the formula from the productivity examination document:
    η_ij = ψ * (I_ij / C_ij_ref) * exp(-δ * Ω_ij)

    Where:
    - η_ij = productivity growth rate for firm i's multiplier on input j
    - ψ = investment effectiveness (stored as attribute)
    - I_ij = investment by firm i in improving input j
    - C_ij_ref = reference cost (price * quantity used)
    - δ = diminishing returns factor (stored as attribute)
    - Ω_ij = cumulative past improvements for firm i on input j
    """

    def __init__(
        self,
        n_firms: int,
        n_industries: int,
        investment_effectiveness: float = 0.1,
        diminishing_returns_factor: float = 0.5,
        **kwargs
    ):
        """Initialize SimpleTechnicalGrowth with parameters.

        Args:
            n_firms (int): Number of firms
            n_industries (int): Number of industries/input types
            investment_effectiveness (float): How effectively investment translates to improvements (ψ)
            diminishing_returns_factor (float): Rate of diminishing returns (δ)
            **kwargs: Additional parameters (for future extensions)
        """
        super().__init__(n_firms, n_industries, investment_effectiveness, diminishing_returns_factor, **kwargs)

    def compute_intermediate_multiplier_growth(
        self,
        current_multipliers: np.ndarray,
        base_coefficients: np.ndarray,
        firm_industries: np.ndarray,
        technical_investment: np.ndarray,
        production: np.ndarray,
        prices: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Calculate growth rates for intermediate input multipliers.

        Uses the formula: η_ij = ψ * (I_ij / C_ij_ref) * exp(-δ * Ω_ij)
        where C_ij_ref = price_j * quantity_ij

        Args:
            current_multipliers (np.ndarray): Current multipliers [n_firms x n_industries]
            base_coefficients (np.ndarray): Base coefficients [n_industries x n_industries]
            firm_industries (np.ndarray): Industry for each firm [n_firms]
            technical_investment (np.ndarray): Investment [n_firms x n_industries]
            production (np.ndarray): Current production [n_firms]
            prices (np.ndarray): Input prices [n_industries]
            **kwargs: Additional parameters

        Returns:
            np.ndarray: Growth rates for each multiplier [n_firms x n_industries]
        """
        n_firms, n_industries = current_multipliers.shape

        # Calculate effective coefficients for each firm
        # effective_coefficient[i,j] = base_coefficient[firm_industry[i], j] * multiplier[i,j]
        effective_coefficients = np.zeros((n_firms, n_industries))
        for i in range(n_firms):
            industry = firm_industries[i]
            effective_coefficients[i, :] = base_coefficients[industry, :] * current_multipliers[i, :]

        # Calculate quantities used: quantity_ij = production_i / effective_coefficient_ij
        quantities = np.zeros((n_firms, n_industries))
        positive_coeff = effective_coefficients > 0
        if np.any(positive_coeff):
            quantities[positive_coeff] = (
                production[:, np.newaxis][positive_coeff] / effective_coefficients[positive_coeff]
            )

        # Reference cost = price * quantity
        reference_costs = prices[np.newaxis, :] * quantities

        # Calculate growth rates where there's positive investment and reference cost
        growth_rates = np.zeros((n_firms, n_industries))
        valid_mask = (technical_investment > 0) & (reference_costs > 0)

        if np.any(valid_mask):
            # Investment intensity
            investment_intensity = np.zeros((n_firms, n_industries))
            investment_intensity[valid_mask] = technical_investment[valid_mask] / reference_costs[valid_mask]

            # Apply diminishing returns based on cumulative improvements
            diminishing_factor = np.exp(-self.diminishing_returns_factor * self.cumulative_intermediate_improvements)

            # Calculate growth rates
            growth_rates[valid_mask] = (
                self.investment_effectiveness * investment_intensity[valid_mask] * diminishing_factor[valid_mask]
            )

        return growth_rates

    def compute_capital_multiplier_growth(
        self,
        current_multipliers: np.ndarray,
        base_coefficients: np.ndarray,
        firm_industries: np.ndarray,
        technical_investment: np.ndarray,
        production: np.ndarray,
        prices: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Calculate growth rates for capital input multipliers.

        Uses the same formula as intermediate inputs.

        Args:
            current_multipliers (np.ndarray): Current multipliers [n_firms x n_industries]
            base_coefficients (np.ndarray): Base coefficients [n_industries x n_industries]
            firm_industries (np.ndarray): Industry for each firm [n_firms]
            technical_investment (np.ndarray): Investment [n_firms x n_industries]
            production (np.ndarray): Current production [n_firms]
            prices (np.ndarray): Input prices [n_industries]
            **kwargs: Additional parameters

        Returns:
            np.ndarray: Growth rates for each multiplier [n_firms x n_industries]
        """
        n_firms, n_industries = current_multipliers.shape

        # Calculate effective coefficients for each firm
        # effective_coefficient[i,j] = base_coefficient[firm_industry[i], j] * multiplier[i,j]
        effective_coefficients = np.zeros((n_firms, n_industries))
        for i in range(n_firms):
            industry = firm_industries[i]
            effective_coefficients[i, :] = base_coefficients[industry, :] * current_multipliers[i, :]

        # Calculate quantities used
        quantities = np.zeros((n_firms, n_industries))
        positive_coeff = effective_coefficients > 0
        if np.any(positive_coeff):
            quantities[positive_coeff] = (
                production[:, np.newaxis][positive_coeff] / effective_coefficients[positive_coeff]
            )

        # Reference cost = price * quantity
        reference_costs = prices[np.newaxis, :] * quantities

        # Calculate growth rates
        growth_rates = np.zeros((n_firms, n_industries))
        valid_mask = (technical_investment > 0) & (reference_costs > 0)

        if np.any(valid_mask):
            investment_intensity = np.zeros((n_firms, n_industries))
            investment_intensity[valid_mask] = technical_investment[valid_mask] / reference_costs[valid_mask]

            diminishing_factor = np.exp(-self.diminishing_returns_factor * self.cumulative_capital_improvements)

            growth_rates[valid_mask] = (
                self.investment_effectiveness * investment_intensity[valid_mask] * diminishing_factor[valid_mask]
            )

        return growth_rates


class BundleAwareTechnicalGrowth(SimpleTechnicalGrowth):
    """Technical growth with bundle-aware investment effects.

    Extends SimpleTechnicalGrowth to account for substitution bundles.
    Within a bundle, improvements to one input can affect the effective
    productivity of the entire bundle through spillover effects.
    """

    def __init__(
        self,
        n_firms: int,
        n_industries: int,
        investment_effectiveness: float = 0.1,
        diminishing_returns_factor: float = 0.5,
        substitution_bundles: Optional[np.ndarray] = None,
        bundle_spillover: float = 0.2,
        **kwargs
    ):
        """Initialize with bundle information.

        Args:
            n_firms (int): Number of firms
            n_industries (int): Number of industries/input types
            investment_effectiveness (float): How effectively investment translates to improvements
            diminishing_returns_factor (float): Rate of diminishing returns
            substitution_bundles (Optional[np.ndarray]): Bundle matrix [n_industries x n_bundles]
                indicating which inputs belong to which substitution bundles
            bundle_spillover (float): Fraction of improvement that spills over to bundle members
            **kwargs: Additional parameters
        """
        super().__init__(n_firms, n_industries, investment_effectiveness, diminishing_returns_factor)
        self.substitution_bundles = substitution_bundles
        self.bundle_spillover = bundle_spillover

    def apply_bundle_effects(self, growth_rates: np.ndarray) -> np.ndarray:
        """Apply bundle spillover effects to growth rates.

        When one input in a bundle improves, it can have positive spillovers
        to other inputs in the same bundle (representing complementarities).

        Args:
            growth_rates (np.ndarray): Base growth rates [n_firms x n_industries]

        Returns:
            np.ndarray: Adjusted growth rates with bundle effects
        """
        if self.substitution_bundles is None:
            return growth_rates

        adjusted_rates = growth_rates.copy()

        # For each bundle
        for bundle_idx in range(self.substitution_bundles.shape[1]):
            bundle_members = self.substitution_bundles[:, bundle_idx] > 0

            if np.any(bundle_members):
                # Calculate average improvement in the bundle for each firm
                for firm_idx in range(growth_rates.shape[0]):
                    firm_bundle_rates = growth_rates[firm_idx, bundle_members]
                    if len(firm_bundle_rates) > 0:
                        bundle_avg = np.mean(firm_bundle_rates)

                        # Apply spillover to all bundle members for this firm
                        for input_idx in np.where(bundle_members)[0]:
                            # Add spillover from other bundle members
                            spillover = self.bundle_spillover * (bundle_avg - growth_rates[firm_idx, input_idx])
                            adjusted_rates[firm_idx, input_idx] += max(0, spillover)

        return adjusted_rates

    def compute_intermediate_multiplier_growth(
        self,
        current_multipliers: np.ndarray,
        base_coefficients: np.ndarray,
        firm_industries: np.ndarray,
        technical_investment: np.ndarray,
        production: np.ndarray,
        prices: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Calculate growth rates with bundle effects.

        Args:
            current_multipliers (np.ndarray): Current multipliers
            base_coefficients (np.ndarray): Base coefficients
            firm_industries (np.ndarray): Firm industries
            technical_investment (np.ndarray): Investment in each input
            production (np.ndarray): Current production levels
            prices (np.ndarray): Current input prices
            **kwargs: Additional parameters

        Returns:
            np.ndarray: Growth rates with bundle effects
        """
        # Get base growth rates from parent class
        base_growth = super().compute_intermediate_multiplier_growth(
            current_multipliers,
            base_coefficients,
            firm_industries,
            technical_investment,
            production,
            prices,
            **kwargs,
        )

        # Apply bundle effects
        return self.apply_bundle_effects(base_growth)

    def compute_capital_multiplier_growth(
        self,
        current_multipliers: np.ndarray,
        base_coefficients: np.ndarray,
        firm_industries: np.ndarray,
        technical_investment: np.ndarray,
        production: np.ndarray,
        prices: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Calculate growth rates with bundle effects.

        Args:
            current_multipliers (np.ndarray): Current multipliers
            base_coefficients (np.ndarray): Base coefficients
            firm_industries (np.ndarray): Firm industries
            technical_investment (np.ndarray): Investment in each input
            production (np.ndarray): Current production levels
            prices (np.ndarray): Current input prices
            **kwargs: Additional parameters

        Returns:
            np.ndarray: Growth rates with bundle effects
        """
        # Get base growth rates from parent class
        base_growth = super().compute_capital_multiplier_growth(
            current_multipliers,
            base_coefficients,
            firm_industries,
            technical_investment,
            production,
            prices,
            **kwargs,
        )

        # Apply bundle effects
        return self.apply_bundle_effects(base_growth)
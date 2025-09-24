from abc import ABC, abstractmethod

import numpy as np


class ProductivityGrowth(ABC):
    """Abstract base class for computing Total Factor Productivity (TFP) growth.

    This class defines strategies for calculating how firms' productivity
    grows over time based on:
    - Base/exogenous growth rates
    - Investment in productivity improvements
    - Stochastic shocks (optional)

    TFP affects all inputs uniformly, allowing firms to produce more output
    with the same amount of inputs.
    """

    @abstractmethod
    def compute_tfp_growth(
        self,
        current_tfp: np.ndarray,
        production: np.ndarray,
        productivity_investment: np.ndarray,
        base_growth_rate: float,
        investment_elasticity: float,
        **kwargs,
    ) -> np.ndarray:
        """Calculate TFP growth rates for firms.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers for each firm
            production (np.ndarray): Current production levels
            productivity_investment (np.ndarray): Investment in productivity improvements
            base_growth_rate (float): Exogenous TFP growth rate (e.g., 0.0025 for 0.25% quarterly)
            investment_elasticity (float): Returns to scale parameter for investment (typically 0.3-0.5)
            **kwargs: Additional parameters for specific implementations

        Returns:
            np.ndarray: TFP growth rates for each firm
        """
        pass

    @staticmethod
    def update_tfp(current_tfp: np.ndarray, tfp_growth_rates: np.ndarray) -> np.ndarray:
        """Update TFP multipliers based on growth rates.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers
            tfp_growth_rates (np.ndarray): Growth rates to apply

        Returns:
            np.ndarray: Updated TFP multipliers
        """
        return current_tfp * (1 + tfp_growth_rates)


class NoOpTFPGrowth(ProductivityGrowth):
    """No-operation TFP growth implementation (static TFP).

    This class implements a static TFP where there is no growth.
    Useful as a default when TFP growth is not desired but the
    interface needs to be satisfied.
    """

    def __init__(self, **kwargs):
        """Initialize NoOpTFPGrowth (ignores all parameters)."""
        # NoOp doesn't need any parameters, but accept them for compatibility
        pass

    def compute_tfp_growth(
        self,
        current_tfp: np.ndarray,
        production: np.ndarray,
        productivity_investment: np.ndarray,
        base_growth_rate: float,
        investment_elasticity: float,
        **kwargs,
    ) -> np.ndarray:
        """Return zero TFP growth (static TFP).

        Args:
            current_tfp (np.ndarray): Current TFP multipliers for each firm
            production (np.ndarray): Current production levels
            productivity_investment (np.ndarray): Investment in productivity improvements
            base_growth_rate (float): Ignored - no growth applied
            investment_elasticity (float): Ignored - no growth applied
            **kwargs: Additional parameters (all ignored)

        Returns:
            np.ndarray: Zero growth rates for each firm
        """
        # Return zeros - no TFP growth
        return np.zeros_like(current_tfp)


class SimpleTFPGrowth(ProductivityGrowth):
    """Simple deterministic TFP growth implementation.

    TFP grows at a constant base rate plus investment-driven improvements:
    g_TFP = base_growth + φ * (Investment/Production)^α
    """

    def __init__(self, investment_effectiveness: float = 0.1, **kwargs):
        """Initialize SimpleTFPGrowth with investment effectiveness parameter.

        Args:
            investment_effectiveness (float): How effectively investment translates to TFP growth
            **kwargs: Additional parameters (for future extensions)
        """
        self.investment_effectiveness = investment_effectiveness

    def compute_tfp_growth(
        self,
        current_tfp: np.ndarray,
        production: np.ndarray,
        productivity_investment: np.ndarray,
        base_growth_rate: float,
        investment_elasticity: float,
        **kwargs,
    ) -> np.ndarray:
        """Calculate TFP growth with base rate and investment effects.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers for each firm
            production (np.ndarray): Current production levels
            productivity_investment (np.ndarray): Investment in productivity improvements
            base_growth_rate (float): Exogenous TFP growth rate
            investment_elasticity (float): Returns to scale parameter (α)

        Returns:
            np.ndarray: TFP growth rates for each firm
        """
        # Base growth applies to all firms
        tfp_growth = np.full_like(current_tfp, base_growth_rate)

        # Add investment-driven growth where production > 0
        positive_production = production > 0
        if np.any(positive_production):
            # Use stored investment effectiveness parameter
            investment_effectiveness = self.investment_effectiveness

            # Calculate investment intensity (Investment/Production)
            # Only consider positive investments for productivity growth
            investment_intensity = np.zeros_like(production)
            positive_investment = productivity_investment > 0
            valid_firms = positive_production & positive_investment

            if np.any(valid_firms):
                investment_intensity[valid_firms] = productivity_investment[valid_firms] / production[valid_firms]

                # Apply diminishing returns with elasticity parameter
                investment_contribution = investment_effectiveness * np.power(
                    investment_intensity, investment_elasticity
                )

                tfp_growth += investment_contribution

        return tfp_growth


class StochasticTFPGrowth(ProductivityGrowth):
    """TFP growth with stochastic shocks.

    Extends simple TFP growth with random productivity shocks:
    g_TFP = base_growth + φ * (Investment/Production)^α + ε
    where ε ~ N(0, σ²)
    """

    def __init__(self, investment_effectiveness: float = 0.1, shock_std: float = 0.01, **kwargs):
        """Initialize StochasticTFPGrowth with parameters.

        Args:
            investment_effectiveness (float): How effectively investment translates to TFP growth
            shock_std (float): Standard deviation of productivity shocks
            **kwargs: Additional parameters (for future extensions)
        """
        self.investment_effectiveness = investment_effectiveness
        self.shock_std = shock_std

    def compute_tfp_growth(
        self,
        current_tfp: np.ndarray,
        production: np.ndarray,
        productivity_investment: np.ndarray,
        base_growth_rate: float,
        investment_elasticity: float,
        **kwargs,
    ) -> np.ndarray:
        """Calculate TFP growth with base rate, investment effects, and stochastic shocks.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers for each firm
            production (np.ndarray): Current production levels
            productivity_investment (np.ndarray): Investment in productivity improvements
            base_growth_rate (float): Exogenous TFP growth rate
            investment_elasticity (float): Returns to scale parameter (α)
            **kwargs: Additional parameters including:
                - shock_std (float): Standard deviation of productivity shocks
                - investment_effectiveness (float): Investment effectiveness parameter

        Returns:
            np.ndarray: TFP growth rates for each firm
        """
        # Start with base growth
        tfp_growth = np.full_like(current_tfp, base_growth_rate)

        # Add investment-driven growth
        positive_production = production > 0
        if np.any(positive_production):
            investment_effectiveness = self.investment_effectiveness

            # Only consider positive investments for productivity growth
            investment_intensity = np.zeros_like(production)
            positive_investment = productivity_investment > 0
            valid_firms = positive_production & positive_investment

            if np.any(valid_firms):
                investment_intensity[valid_firms] = productivity_investment[valid_firms] / production[valid_firms]

                investment_contribution = investment_effectiveness * np.power(
                    investment_intensity, investment_elasticity
                )

                tfp_growth += investment_contribution

        # Add stochastic shocks
        shock_std = self.shock_std
        if shock_std > 0:
            shocks = np.random.normal(0, shock_std, size=current_tfp.shape)
            tfp_growth += shocks

        return tfp_growth


class SectoralTFPGrowth(ProductivityGrowth):
    """Sector-specific TFP growth implementation.

    Different sectors can have different base growth rates and investment
    effectiveness parameters.
    """

    def __init__(
        self,
        investment_effectiveness: float = 0.1,
        sector_base_growth: dict = None,
        sector_effectiveness: dict = None,
        **kwargs,
    ):
        """Initialize SectoralTFPGrowth with parameters.

        Args:
            investment_effectiveness (float): Default investment effectiveness
            sector_base_growth (dict): Base growth rate by sector
            sector_effectiveness (dict): Investment effectiveness by sector
            **kwargs: Additional parameters (for future extensions)
        """
        self.investment_effectiveness = investment_effectiveness
        self.sector_base_growth = sector_base_growth or {}
        self.sector_effectiveness = sector_effectiveness or {}

    def compute_tfp_growth(
        self,
        current_tfp: np.ndarray,
        production: np.ndarray,
        productivity_investment: np.ndarray,
        base_growth_rate: float,
        investment_elasticity: float,
        **kwargs,
    ) -> np.ndarray:
        """Calculate TFP growth with sector-specific parameters.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers for each firm
            production (np.ndarray): Current production levels
            productivity_investment (np.ndarray): Investment in productivity improvements
            base_growth_rate (float): Default exogenous TFP growth rate
            investment_elasticity (float): Returns to scale parameter (α)
            **kwargs: Additional parameters including:
                - sector_ids (np.ndarray): Sector ID for each firm
                - sector_base_growth (dict): Base growth rate by sector
                - sector_effectiveness (dict): Investment effectiveness by sector

        Returns:
            np.ndarray: TFP growth rates for each firm
        """
        # Get sector-specific parameters
        sector_ids = kwargs.get("sector_ids")
        sector_base_growth = self.sector_base_growth
        sector_effectiveness = self.sector_effectiveness

        # Initialize growth rates
        tfp_growth = np.zeros_like(current_tfp)

        # Apply sector-specific base growth
        if sector_ids is not None:
            for sector in np.unique(sector_ids):
                sector_mask = sector_ids == sector
                sector_base = sector_base_growth.get(sector, base_growth_rate)
                tfp_growth[sector_mask] = sector_base
        else:
            tfp_growth[:] = base_growth_rate

        # Add investment-driven growth
        positive_production = production > 0
        if np.any(positive_production):
            # Only consider positive investments for productivity growth
            investment_intensity = np.zeros_like(production)
            positive_investment = productivity_investment > 0
            valid_firms = positive_production & positive_investment

            if np.any(valid_firms):
                investment_intensity[valid_firms] = productivity_investment[valid_firms] / production[valid_firms]

                # Apply sector-specific effectiveness if available
                if sector_ids is not None:
                    for sector in np.unique(sector_ids):
                        sector_mask = (sector_ids == sector) & valid_firms
                        if np.any(sector_mask):
                            effectiveness = sector_effectiveness.get(sector, 0.1)

                            sector_contribution = effectiveness * np.power(
                                investment_intensity[sector_mask], investment_elasticity
                            )
                            tfp_growth[sector_mask] += sector_contribution
                else:
                    effectiveness = self.investment_effectiveness
                    investment_contribution = effectiveness * np.power(investment_intensity, investment_elasticity)
                    tfp_growth += investment_contribution

        return tfp_growth

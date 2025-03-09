"""Policy rate determination for central bank monetary policy.

This module implements various strategies for setting policy interest
rates, including:
- Constant rate maintenance
- Taylor-rule based adjustments
- Growth and inflation targeting
- Interest rate smoothing

The policy rate setting considers:
- Inflation gap from target
- Economic growth rates
- Previous policy rates
- Monetary policy parameters
"""

from abc import ABC, abstractmethod


class PolicyRate(ABC):
    """Abstract base class for determining policy interest rates.

    This class defines strategies for setting monetary policy rates
    based on:
    - Inflation developments
    - Economic growth
    - Policy objectives
    - Previous rate levels

    The rate setting process considers:
    - Price stability targets
    - Economic growth goals
    - Policy transmission lags
    - Financial stability
    """

    @abstractmethod
    def compute_rate(
        self,
        prev_rate: float,
        inflation: float,
        growth: float,
        central_bank_states: dict[str, float],
    ) -> float:
        """Calculate the appropriate policy interest rate.

        Determines policy rate considering:
        - Previous rate level
        - Current inflation
        - Economic growth
        - Policy parameters

        Args:
            prev_rate (float): Previous period's policy rate
            inflation (float): Current inflation rate
            growth (float): Current economic growth rate
            central_bank_states (dict[str, float]): Policy parameters including:
                - targeted_inflation_rate: Inflation target
                - rho: Interest rate smoothing parameter
                - r_star: Natural real interest rate
                - xi_pi: Inflation gap response coefficient
                - xi_gamma: Output growth response coefficient

        Returns:
            float: New policy interest rate
        """
        pass


class ConstantPolicyRate(PolicyRate):
    """Implementation of constant policy rate strategy.

    This class maintains unchanged policy rates by:
    - Keeping rates at previous levels
    - Ignoring inflation developments
    - Disregarding growth rates
    - Maintaining policy stance

    This approach is useful for:
    - Model testing and validation
    - Policy transmission analysis
    - Baseline scenario creation
    """

    def compute_rate(
        self,
        prev_rate: float,
        inflation: float,
        growth: float,
        central_bank_states: dict[str, float],
    ) -> float:
        """Keep policy rate constant.

        Returns the same rate regardless of economic conditions.

        Args:
            [same as parent class]

        Returns:
            float: Previous policy rate (unchanged)
        """
        return prev_rate


class PolednaPolicyRate(PolicyRate):
    """Implementation of Poledna et al. monetary policy rule.

    This class implements a Taylor-type rule that:
    - Responds to inflation gaps
    - Considers economic growth
    - Smooths interest rates
    - Maintains non-negative rates

    The approach provides:
    - Systematic policy responses
    - Price stability focus
    - Growth considerations
    - Policy predictability

    """

    def compute_rate(
        self,
        prev_rate: float,
        inflation: float,
        growth: float,
        central_bank_states: dict[str, float],
    ) -> float:
        """Calculate policy rate using Poledna et al. rule.

        Implements a Taylor-type rule with:
        - Interest rate smoothing (rho parameter)
        - Inflation gap response (xi_pi parameter)
        - Growth response (xi_gamma parameter)
        - Zero lower bound constraint

        Args:
            [same as parent class]

        Returns:
            float: New policy rate based on rule calculation,
                constrained to be non-negative
        """
        return max(
            0.0,
            central_bank_states["rho"] * prev_rate
            + (1 - central_bank_states["rho"])
            * (
                central_bank_states["r_star"]
                + central_bank_states["targeted_inflation_rate"]
                + central_bank_states["xi_pi"] * (inflation - central_bank_states["targeted_inflation_rate"])
                + central_bank_states["xi_gamma"] * growth
            ),
        )

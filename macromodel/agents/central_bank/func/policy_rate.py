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
        cpi_yoy_inflation: float | None = None,
        output_gap: float | None = None,
        time_unit: int = 1,
        shock: float = 0.0,
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
            cpi_yoy_inflation (float | None): Optional year-over-year CPI inflation
                used by rules that target annual CPI inflation directly.
            output_gap (float | None): Optional output gap used by rules that react
                to activity relative to trend rather than raw growth.
            time_unit (int): Simulation period length in months.
            shock (float): Additive policy shock.

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
        cpi_yoy_inflation: float | None = None,
        output_gap: float | None = None,
        time_unit: int = 1,
        shock: float = 0.0,
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
        cpi_yoy_inflation: float | None = None,
        output_gap: float | None = None,
        time_unit: int = 1,
        shock: float = 0.0,
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


class SmoothTaylorRule(PolicyRate):
    """Implementation of a smoothed Taylor rule with annual CPI inflation.

    This rule follows:

        i_t = rho * i_{t-1}
            + (1-rho) * (r_star + pi_star + phi_pi * (pi_t - pi_star) + phi_q * q_t)
            + epsilon_t

    Design choices for this model implementation:
    - `pi_t` is observed year-over-year CPI inflation.
    - `r_star`, `targeted_inflation_rate`, `phi_pi`, and `phi_q` are interpreted as
      annual-policy parameters.
    - The model stores and applies policy rates in per-period units, so the final
      annual policy rate is converted to the simulation frequency using `time_unit`.
    """

    @staticmethod
    def _periods_per_year(time_unit: int) -> int:
        """Convert the model time unit in months to integer periods per year."""
        if time_unit <= 0 or 12 % time_unit != 0:
            raise ValueError("SmoothTaylorRule requires `time_unit` to be a positive divisor of 12.")
        return 12 // time_unit

    def compute_rate(
        self,
        prev_rate: float,
        inflation: float,
        growth: float,
        central_bank_states: dict[str, float],
        cpi_yoy_inflation: float | None = None,
        output_gap: float | None = None,
        time_unit: int = 1,
        shock: float = 0.0,
    ) -> float:
        """Calculate the per-period policy rate from annual Taylor-rule inputs.

        Args:
            [same as parent class]

        Returns:
            float: New policy rate in per-period units, constrained to be non-negative.
        """
        periods_per_year = self._periods_per_year(time_unit)
        annual_prev_rate = prev_rate * periods_per_year
        annual_inflation = inflation if cpi_yoy_inflation is None else cpi_yoy_inflation
        current_output_gap = 0.0 if output_gap is None else output_gap

        annual_rate = (
            central_bank_states["rho"] * annual_prev_rate
            + (1 - central_bank_states["rho"])
            * (
                central_bank_states["r_star"]
                + central_bank_states["targeted_inflation_rate"]
                + central_bank_states["phi_pi"] * (annual_inflation - central_bank_states["targeted_inflation_rate"])
                + central_bank_states["phi_q"] * current_output_gap
            )
            + shock
        )
        return max(0.0, annual_rate / periods_per_year)

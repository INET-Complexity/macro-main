from abc import ABC, abstractmethod

import numpy as np


class ProductivityInvestmentPlanner(ABC):
    """Abstract base class for planning firms' productivity investments.

    This class defines strategies for determining optimal investment in
    productivity improvements based on:
    - Expected returns from TFP growth
    - Available financial resources
    - Hurdle rates and risk preferences
    - Market conditions and demand expectations

    The investment planning follows the framework in Section 3 of the
    productivity examination document, implementing the decision logic
    for allocating resources to productivity improvements.

    Attributes:
        hurdle_rate (float): Minimum required rate of return (ρ_min)
        max_investment_fraction (float): Maximum investment as fraction of output (θ)
        investment_effectiveness (float): How investment translates to TFP growth (φ)
        investment_elasticity (float): Returns to scale parameter (α)
    """

    def __init__(
        self,
        hurdle_rate: float = 0.15,
        max_investment_fraction: float = 0.1,
        investment_effectiveness: float = 0.1,
        investment_elasticity: float = 0.3,
    ):
        """Initialize the productivity investment planner.

        Args:
            hurdle_rate (float): Minimum required return on investment
            max_investment_fraction (float): Max investment as fraction of output
            investment_effectiveness (float): TFP growth effectiveness parameter
            investment_elasticity (float): Diminishing returns parameter
        """
        self.hurdle_rate = hurdle_rate
        self.max_investment_fraction = max_investment_fraction
        self.investment_effectiveness = investment_effectiveness
        self.investment_elasticity = investment_elasticity

    @abstractmethod
    def plan_productivity_investment(
        self,
        current_tfp: np.ndarray,
        current_production: np.ndarray,
        current_unit_costs: np.ndarray,
        available_cash: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Plan productivity investment for each firm.

        Determines optimal investment in productivity improvements based on
        expected cost savings, financial constraints, and strategic considerations.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers
            current_production (np.ndarray): Current production levels
            current_unit_costs (np.ndarray): Current unit costs of production
            available_cash (np.ndarray): Cash available for investment
            **kwargs: Additional parameters for specific implementations

        Returns:
            np.ndarray: Planned productivity investment for each firm
        """
        pass

    def compute_expected_tfp_growth(
        self, productivity_investment: np.ndarray, current_production: np.ndarray
    ) -> np.ndarray:
        """Calculate expected TFP growth from investment.

        Uses the formula: g_TFP = φ * (I_TFP/Y)^α

        Args:
            productivity_investment (np.ndarray): Investment amounts
            current_production (np.ndarray): Current production levels

        Returns:
            np.ndarray: Expected TFP growth rates
        """
        investment_intensity = np.divide(
            productivity_investment,
            current_production,
            out=np.zeros_like(productivity_investment),
            where=current_production > 0,
        )
        return self.investment_effectiveness * np.power(investment_intensity, self.investment_elasticity)

    def compute_hurdle_adjusted_value(
        self,
        productivity_investment: np.ndarray,
        current_production: np.ndarray,
        current_unit_costs: np.ndarray,
    ) -> np.ndarray:
        """Calculate hurdle-adjusted present value from productivity investment.

        Following investment_decision.md logic:
        - TFP growth factor: A_{t+1}/A_t = φ * (I/Y)^α
        - Cost reduction per period: tfp_growth * unit_cost * Y
        - Hurdle-adjusted present value: cost_reduction * (1+r_h)/r_h

        This gives the present value of perpetual cost savings, discounted at
        the hurdle rate. Investment is profitable if this exceeds investment cost.

        Args:
            productivity_investment (np.ndarray): Investment amounts
            current_production (np.ndarray): Current production levels
            current_unit_costs (np.ndarray): Current unit costs of production

        Returns:
            np.ndarray: Hurdle-adjusted present value of cost savings
        """
        # Expected TFP growth factor (φ * (I/Y)^α)
        tfp_growth = self.compute_expected_tfp_growth(productivity_investment, current_production)

        # Cost reduction per period = tfp_growth * unit_cost * production
        cost_reduction_per_period = tfp_growth * current_unit_costs * current_production

        # Present value using hurdle rate as discount rate
        # PV = cost_reduction * (1 + r_h) / r_h
        hurdle_discount_factor = (1 + self.hurdle_rate) / self.hurdle_rate

        return cost_reduction_per_period * hurdle_discount_factor

    def compute_investment_budget(
        self,
        available_cash: np.ndarray,
        current_production: np.ndarray,
        max_cash_fraction: float = 0.5,
    ) -> np.ndarray:
        """Calculate available budget for productivity investment.

        Implements constraint: B_i(t) = min{κ*Cash_i(t), θ*Y_i(t)}

        Args:
            available_cash (np.ndarray): Available cash balances
            current_production (np.ndarray): Current production levels
            max_cash_fraction (float): Maximum fraction of cash to use (κ)

        Returns:
            np.ndarray: Available investment budget for each firm
        """
        cash_constraint = max_cash_fraction * np.maximum(0, available_cash)
        output_constraint = self.max_investment_fraction * current_production
        return np.minimum(cash_constraint, output_constraint)

    def compute_expected_prices(self, current_prices: np.ndarray, estimated_inflation: float) -> np.ndarray:
        """Calculate expected future prices using inflation estimate.

        Uses the same approach as elsewhere in the model:
        E[p(t+1)] = p(t) * (1 + estimated_inflation)

        Args:
            current_prices (np.ndarray): Current prices
            estimated_inflation (float): Expected inflation rate from economy

        Returns:
            np.ndarray: Expected future prices
        """
        return current_prices * (1 + estimated_inflation)


class NoProductivityInvestmentPlanner(ProductivityInvestmentPlanner):
    """No productivity investment implementation (for backward compatibility).

    This implementation always returns zero investment, maintaining the
    model's behavior when productivity investment is not enabled.
    """

    def plan_productivity_investment(
        self,
        current_tfp: np.ndarray,
        current_production: np.ndarray,
        current_unit_costs: np.ndarray,
        available_cash: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Return zero productivity investment for all firms.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers
            current_production (np.ndarray): Current production levels
            current_unit_costs (np.ndarray): Current unit costs of production
            available_cash (np.ndarray): Cash available for investment
            **kwargs: Additional parameters (ignored)

        Returns:
            np.ndarray: Zero investment for all firms
        """
        return np.zeros_like(current_production)


class SimpleProductivityInvestmentPlanner(ProductivityInvestmentPlanner):
    """Simple rule-based productivity investment planning.

    This implementation uses a fixed fraction of available budget for
    productivity investment, subject to hurdle rate constraints.
    """

    def __init__(
        self,
        hurdle_rate: float = 0.15,
        max_investment_fraction: float = 0.1,
        investment_effectiveness: float = 0.1,
        investment_elasticity: float = 0.3,
        investment_propensity: float = 0.2,
    ):
        """Initialize the simple productivity investment planner.

        Args:
            hurdle_rate (float): Minimum required return on investment
            max_investment_fraction (float): Max investment as fraction of output
            investment_effectiveness (float): TFP growth effectiveness parameter
            investment_elasticity (float): Diminishing returns parameter
            investment_propensity (float): Fraction of budget to invest
        """
        super().__init__(
            hurdle_rate,
            max_investment_fraction,
            investment_effectiveness,
            investment_elasticity,
        )
        self.investment_propensity = investment_propensity

    def plan_productivity_investment(
        self,
        current_tfp: np.ndarray,
        current_production: np.ndarray,
        current_unit_costs: np.ndarray,
        available_cash: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Plan productivity investment using simple rules.

        Invests a fixed fraction of available budget if expected cost savings
        exceed the hurdle rate.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers
            current_production (np.ndarray): Current production levels
            current_unit_costs (np.ndarray): Current unit costs of production
            available_cash (np.ndarray): Cash available for investment
            **kwargs: Additional parameters

        Returns:
            np.ndarray: Planned productivity investment for each firm
        """
        # Compute available budget
        budget = self.compute_investment_budget(available_cash, current_production)

        # Candidate investment (fraction of budget)
        candidate_investment = self.investment_propensity * budget

        # Compute hurdle-adjusted present value of cost savings
        hurdle_value = self.compute_hurdle_adjusted_value(
            candidate_investment, current_production, current_unit_costs
        )

        # Investment is profitable if hurdle-adjusted value exceeds investment cost
        profitable = hurdle_value > candidate_investment

        # Only invest where profitable
        productivity_investment = np.where(profitable, candidate_investment, 0)

        return productivity_investment


class OptimalProductivityInvestmentPlanner(ProductivityInvestmentPlanner):
    """Optimal productivity investment planning based on expected returns.

    This implementation solves for the optimal investment level that
    maximizes expected returns subject to budget and hurdle rate constraints.
    """

    def __init__(
        self,
        hurdle_rate: float = 0.15,
        max_investment_fraction: float = 0.1,
        investment_effectiveness: float = 0.1,
        investment_elasticity: float = 0.3,
        search_steps: int = 20,
    ):
        """Initialize the optimal productivity investment planner.

        Args:
            hurdle_rate (float): Minimum required return on investment
            max_investment_fraction (float): Max investment as fraction of output
            investment_effectiveness (float): TFP growth effectiveness parameter
            investment_elasticity (float): Diminishing returns parameter
            search_steps (int): Number of steps in optimization search
        """
        super().__init__(
            hurdle_rate,
            max_investment_fraction,
            investment_effectiveness,
            investment_elasticity,
        )
        self.search_steps = search_steps

    def plan_productivity_investment(
        self,
        current_tfp: np.ndarray,
        current_production: np.ndarray,
        current_unit_costs: np.ndarray,
        available_cash: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Plan productivity investment by optimizing expected returns.

        Searches for the investment level that maximizes net present value
        from cost savings subject to constraints.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers
            current_production (np.ndarray): Current production levels
            current_unit_costs (np.ndarray): Current unit costs of production
            available_cash (np.ndarray): Cash available for investment
            **kwargs: Additional parameters

        Returns:
            np.ndarray: Optimal productivity investment for each firm
        """
        n_firms = len(current_production)
        optimal_investment = np.zeros(n_firms)

        # Compute constraints
        budget = self.compute_investment_budget(available_cash, current_production)

        # For each firm, find optimal investment level
        for i in range(n_firms):
            if budget[i] <= 0 or current_production[i] <= 0 or current_unit_costs[i] <= 0:
                continue

            # Search over possible investment levels
            # Start with zero investment as baseline (NPV = 0)
            best_npv = 0
            best_investment = 0

            investment_levels = np.linspace(0, budget[i], self.search_steps)

            for investment in investment_levels:
                if investment == 0:
                    continue

                # Compute hurdle-adjusted value for this investment level
                inv_array = np.array([investment])
                prod_array = np.array([current_production[i]])
                costs_array = np.array([current_unit_costs[i]])

                hurdle_value = self.compute_hurdle_adjusted_value(
                    inv_array, prod_array, costs_array
                )[0]

                # NPV: hurdle-adjusted value minus investment cost
                npv = hurdle_value - investment

                # Only invest if NPV is positive and better than current best
                if npv > best_npv:
                    best_npv = npv
                    best_investment = investment

            optimal_investment[i] = best_investment

        return optimal_investment
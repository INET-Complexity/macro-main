from abc import ABC, abstractmethod
from typing import Optional

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
        n_firms: int,
        hurdle_rate: float | list[float] = 0.15,
        max_investment_fraction: float | list[float] = 0.1,
        investment_effectiveness: float | list[float] = 0.1,
        investment_elasticity: float | list[float] = 0.3,
        # New parameters for technical coefficient investment
        tfp_investment_share: float | list[float] = 0.4,
        technical_investment_effectiveness: float | list[float] = 0.15,
        technical_diminishing_returns: float | list[float] = 0.5,
        price_weight: float | list[float] = 0.4,
        usage_weight: float | list[float] = 0.3,
        potential_weight: float | list[float] = 0.3,
    ):
        """Initialize the productivity investment planner.

        Args:
            n_firms (int): Number of firms (required to convert scalars to arrays)
            hurdle_rate (float | list[float]): Minimum required return on investment
            max_investment_fraction (float | list[float]): Max investment as fraction of output
            investment_effectiveness (float | list[float]): TFP growth effectiveness parameter
            investment_elasticity (float | list[float]): Diminishing returns parameter
            tfp_investment_share (float | list[float]): Share of budget allocated to TFP vs technical
            technical_investment_effectiveness (float | list[float]): Technical coefficient growth effectiveness
            technical_diminishing_returns (float | list[float]): Exponential diminishing returns factor
            price_weight (float | list[float]): Weight for price-based input targeting
            usage_weight (float | list[float]): Weight for usage-based input targeting
            potential_weight (float | list[float]): Weight for improvement potential targeting
        """
        self.n_firms = n_firms

        # Convert all parameters to numpy arrays with shape (n_firms,)
        self.hurdle_rate = self._to_array(hurdle_rate, n_firms)
        self.max_investment_fraction = self._to_array(max_investment_fraction, n_firms)
        self.investment_effectiveness = self._to_array(investment_effectiveness, n_firms)
        self.investment_elasticity = self._to_array(investment_elasticity, n_firms)
        # Technical coefficient parameters
        self.tfp_investment_share = self._to_array(tfp_investment_share, n_firms)
        self.technical_investment_effectiveness = self._to_array(technical_investment_effectiveness, n_firms)
        self.technical_diminishing_returns = self._to_array(technical_diminishing_returns, n_firms)
        self.price_weight = self._to_array(price_weight, n_firms)
        self.usage_weight = self._to_array(usage_weight, n_firms)
        self.potential_weight = self._to_array(potential_weight, n_firms)

    def _to_array(self, value: float | list[float], n_firms: int) -> np.ndarray:
        """Convert scalar or list to numpy array with shape (n_firms,).

        Args:
            value (float | list[float]): Scalar or list of values
            n_firms (int): Number of firms

        Returns:
            np.ndarray: Array with shape (n_firms,)
        """
        if isinstance(value, (list, tuple)):
            arr = np.array(value)
            if len(arr) != n_firms:
                raise ValueError(f"Parameter list length {len(arr)} does not match n_firms {n_firms}")
            return arr
        else:
            return np.full(n_firms, value)

    @abstractmethod
    def plan_productivity_investment(
        self,
        current_tfp: np.ndarray,
        current_production: np.ndarray,
        current_unit_costs: np.ndarray,
        available_cash: np.ndarray,
        current_prices: np.ndarray,
        n_industries: int,
        input_usage: np.ndarray,
        current_tech_multipliers: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plan productivity investment for each firm.

        Determines optimal investment in productivity improvements based on
        expected cost savings, financial constraints, and strategic considerations.
        Also allocates the investment between TFP and technical coefficient improvements.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers [n_firms]
            current_production (np.ndarray): Current production levels [n_firms]
            current_unit_costs (np.ndarray): Current unit costs of production [n_firms]
            available_cash (np.ndarray): Cash available for investment [n_firms]
            current_prices (np.ndarray): Current market prices [n_industries]
            n_industries (int): Number of industries
            input_usage (np.ndarray): Input usage by firms [n_firms x n_industries]
            current_tech_multipliers (np.ndarray): Current technical multipliers [n_firms x n_industries]
            substitution_bundle_matrix (np.ndarray): Bundle matrix [n_industries x n_bundles]

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
                - Total planned productivity investment for each firm [n_firms]
                - TFP investment portion for each firm [n_firms]
                - Technical investment by input [n_firms x n_industries]
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
        max_cash_fraction: float = 0.1,
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

    def allocate_productivity_investment(
        self,
        total_investment: np.ndarray,
        current_prices: np.ndarray,
        input_usage: np.ndarray,
        current_tech_multipliers: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
        bundle_significance_threshold: float = 0.1,
        arbitrage_intensity: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Allocate total investment between TFP and technical improvements.

        Args:
            total_investment (np.ndarray): Total productivity investment [n_firms]
            current_prices (np.ndarray): Current market prices [n_industries]
            input_usage (np.ndarray): Input usage by firms [n_firms x n_industries]
            current_tech_multipliers (np.ndarray): Current technical multipliers [n_firms x n_industries]
            substitution_bundle_matrix (np.ndarray): Bundle matrix [n_industries x n_bundles]
            bundle_significance_threshold (float): Min fraction of spending for bundle arbitrage
            arbitrage_intensity (float): Strength of arbitrage adjustment

        Returns:
            tuple[np.ndarray, np.ndarray]: (tfp_investment, technical_investment_by_input)
        """
        n_firms, n_industries = input_usage.shape

        # Split between TFP and technical based on configured share
        tfp_investment = self.tfp_investment_share * total_investment
        technical_budget = (1.0 - self.tfp_investment_share) * total_investment

        # Compute base input priorities for technical investment
        priorities = self._compute_input_priorities(current_prices, input_usage, current_tech_multipliers)

        # Apply bundle arbitrage (always - bundles exist even if singleton)
        priorities = self._apply_bundle_arbitrage(
            priorities,
            current_prices,
            input_usage,
            substitution_bundle_matrix,
            bundle_significance_threshold,
            arbitrage_intensity,
        )

        # Normalize priorities
        priority_sums = priorities.sum(axis=1, keepdims=True)
        normalized_priorities = np.divide(
            priorities, priority_sums, out=np.zeros_like(priorities), where=priority_sums > 0
        )

        # Distribute technical budget across inputs
        technical_investment = technical_budget[:, np.newaxis] * normalized_priorities

        return tfp_investment, technical_investment

    def _compute_input_priorities(
        self,
        current_prices: np.ndarray,
        input_usage: np.ndarray,
        current_tech_multipliers: np.ndarray,
    ) -> np.ndarray:
        """Compute investment priorities for each input type.

        Combines price, usage, and improvement potential factors.
        """
        n_firms, n_industries = input_usage.shape

        # Price-based priority: expensive inputs get higher weight
        avg_price: np.ndarray = np.mean(current_prices)  # type: ignore
        relative_prices = current_prices / (avg_price + 1e-10)

        # Usage-based priority: heavily used inputs get higher weight
        total_usage_per_firm = input_usage.sum(axis=1, keepdims=True)
        relative_usage = np.divide(
            input_usage, total_usage_per_firm, out=np.zeros_like(input_usage), where=total_usage_per_firm > 0
        )

        # Improvement potential: lower efficiency gets higher weight
        improvement_potential = 1.0 / np.maximum(0.5, current_tech_multipliers)

        # Combine factors using configuration weights (now firm-specific)
        priorities = (
            self.price_weight[:, np.newaxis] * relative_prices[np.newaxis, :]
            + self.usage_weight[:, np.newaxis] * relative_usage
            + self.potential_weight[:, np.newaxis] * improvement_potential
        )

        return priorities

    def _apply_bundle_arbitrage(
        self,
        base_priorities: np.ndarray,
        current_prices: np.ndarray,
        input_usage: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
        bundle_significance_threshold: float = 0.1,
        arbitrage_intensity: float = 2.0,
    ) -> np.ndarray:
        """Apply bundle-aware arbitrage to investment priorities.

        For significant bundles (high spending), boost investment in cheaper alternatives
        and reduce investment in expensive inputs within the bundle.

        Args:
            base_priorities (np.ndarray): Base priority scores [n_firms x n_industries]
            current_prices (np.ndarray): Current market prices [n_industries]
            input_usage (np.ndarray): Input usage by firms [n_firms x n_industries]
            substitution_bundle_matrix (np.ndarray): Bundle matrix [n_industries x n_bundles]
            bundle_significance_threshold (float): Min fraction of total spending for bundle arbitrage
            arbitrage_intensity (float): Strength of arbitrage adjustment

        Returns:
            np.ndarray: Adjusted priority scores [n_firms x n_industries]
        """
        n_firms, n_industries = base_priorities.shape
        n_bundles = substitution_bundle_matrix.shape[1]

        adjusted_priorities = base_priorities.copy()

        # Calculate spending on each input for each firm
        input_spending = input_usage * current_prices[np.newaxis, :]  # [n_firms x n_industries]
        total_spending_per_firm = input_spending.sum(axis=1, keepdims=True)  # [n_firms x 1]

        # Process each bundle
        for bundle_idx in range(n_bundles):
            # Find industries in this bundle (non-zero weights)
            bundle_mask = substitution_bundle_matrix[:, bundle_idx] > 0
            bundle_industries = np.where(bundle_mask)[0]

            if len(bundle_industries) <= 1:
                continue  # Skip singleton bundles - no arbitrage opportunity

            # Calculate bundle spending for each firm
            bundle_spending = input_spending[:, bundle_industries].sum(axis=1, keepdims=True)  # [n_firms x 1]
            bundle_spending_fraction = np.divide(
                bundle_spending,
                total_spending_per_firm,
                out=np.zeros_like(bundle_spending),
                where=total_spending_per_firm > 0,
            )

            # Only apply arbitrage for firms where this bundle is significant
            significant_firms = (bundle_spending_fraction >= bundle_significance_threshold).flatten()

            if not np.any(significant_firms):
                continue  # No firms spend significantly on this bundle

            # Calculate average price within this bundle
            bundle_prices = current_prices[bundle_industries]
            bundle_avg_price = np.mean(bundle_prices)

            # Apply arbitrage: cheaper inputs get boosted, expensive inputs get penalized
            relative_prices = bundle_prices / bundle_avg_price
            arbitrage_multiplier = np.power(1.0 / relative_prices, arbitrage_intensity)

            # Apply to significant firms only
            for firm_idx in np.where(significant_firms)[0]:
                adjusted_priorities[firm_idx, bundle_industries] *= arbitrage_multiplier

        return adjusted_priorities

    def compute_combined_hurdle_value(
        self,
        total_investment: np.ndarray,
        current_production: np.ndarray,
        current_unit_costs: np.ndarray,
        tfp_investment: np.ndarray,
        technical_investment: np.ndarray,
        input_costs: np.ndarray,
        current_coefficients: np.ndarray,
        cumulative_improvements: np.ndarray,
    ) -> np.ndarray:
        """Calculate hurdle-adjusted present value from combined investments.

        Evaluates both TFP and technical coefficient returns.

        Args:
            total_investment (np.ndarray): Total investment amounts
            current_production (np.ndarray): Current production levels
            current_unit_costs (np.ndarray): Current unit costs of production
            tfp_investment (np.ndarray): TFP investment portion
            technical_investment (np.ndarray): Technical investment by input [n_firms x n_industries]
            input_costs (np.ndarray): Cost of each input type [n_firms x n_industries]
            current_coefficients (np.ndarray): Current technical coefficients [n_firms x n_industries]
            cumulative_improvements (np.ndarray): Cumulative past improvements [n_firms x n_industries]

        Returns:
            np.ndarray: Hurdle-adjusted present value of combined cost savings
        """
        # TFP component
        tfp_growth = self.compute_expected_tfp_growth(tfp_investment, current_production)
        tfp_cost_savings = tfp_growth * current_unit_costs * current_production

        # Technical component - vectorized across all inputs
        # Calculate investment intensity for each input
        investment_intensity = np.divide(
            technical_investment,
            current_production[:, np.newaxis] * current_coefficients,
            out=np.zeros_like(technical_investment),
            where=(current_production[:, np.newaxis] * current_coefficients) > 0,
        )

        # Apply exponential diminishing returns (firm-specific parameters)
        diminishing_factor = np.exp(-self.technical_diminishing_returns[:, np.newaxis] * cumulative_improvements)

        # Effective technical growth with diminishing returns (firm-specific parameters)
        technical_growth = (
            self.technical_investment_effectiveness[:, np.newaxis]
            * np.power(investment_intensity, self.investment_elasticity[:, np.newaxis])
            * diminishing_factor
        )

        # Cost savings from technical improvements
        technical_cost_reductions = technical_growth * input_costs
        tech_cost_savings = np.sum(technical_cost_reductions, axis=1)

        # Combined present value
        total_cost_savings = tfp_cost_savings + tech_cost_savings
        hurdle_discount_factor = (1 + self.hurdle_rate) / self.hurdle_rate

        return total_cost_savings * hurdle_discount_factor


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
        current_prices: np.ndarray,
        n_industries: int,
        input_usage: np.ndarray,
        current_tech_multipliers: np.ndarray,
        substitution_bundle_matrix: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return zero productivity investment for all firms.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers
            current_production (np.ndarray): Current production levels
            current_unit_costs (np.ndarray): Current unit costs of production
            available_cash (np.ndarray): Cash available for investment
            current_prices (np.ndarray): Current market prices by industry
            n_industries (int): Number of industries
            input_usage (np.ndarray): Used intermediate inputs [n_firms x n_industries]
            current_tech_multipliers (np.ndarray): Technical coefficient multipliers [n_firms x n_industries]
            substitution_bundle_matrix (np.ndarray | None): Substitution bundle matrix

        Returns:
            tuple: Zero investments (total, TFP, technical by input)
        """
        n_firms = len(current_production)

        total_investment = np.zeros(n_firms)
        tfp_investment = np.zeros(n_firms)
        technical_investment = np.zeros((n_firms, n_industries))

        return total_investment, tfp_investment, technical_investment


class SimpleProductivityInvestmentPlanner(ProductivityInvestmentPlanner):
    """Simple rule-based productivity investment planning.

    This implementation uses a fixed fraction of available budget for
    productivity investment, subject to hurdle rate constraints.
    """

    def __init__(
        self,
        n_firms: int,
        hurdle_rate: float | list[float] = 0.15,
        max_investment_fraction: float | list[float] = 0.1,
        investment_effectiveness: float | list[float] = 0.1,
        investment_elasticity: float | list[float] = 0.3,
        investment_propensity: float | list[float] = 0.2,
        # New parameters for technical coefficient investment
        tfp_investment_share: float | list[float] = 0.4,
        technical_investment_effectiveness: float | list[float] = 0.15,
        technical_diminishing_returns: float | list[float] = 0.5,
        price_weight: float | list[float] = 0.4,
        usage_weight: float | list[float] = 0.3,
        potential_weight: float | list[float] = 0.3,
    ):
        """Initialize the simple productivity investment planner.

        Args:
            n_firms (int): Number of firms
            hurdle_rate (float | list[float]): Minimum required return on investment
            max_investment_fraction (float | list[float]): Max investment as fraction of output
            investment_effectiveness (float | list[float]): TFP growth effectiveness parameter
            investment_elasticity (float | list[float]): Diminishing returns parameter
            investment_propensity (float | list[float]): Fraction of budget to invest
        """
        super().__init__(
            n_firms,
            hurdle_rate,
            max_investment_fraction,
            investment_effectiveness,
            investment_elasticity,
            tfp_investment_share,
            technical_investment_effectiveness,
            technical_diminishing_returns,
            price_weight,
            usage_weight,
            potential_weight,
        )
        self.investment_propensity = self._to_array(investment_propensity, n_firms)

    def plan_productivity_investment(
        self,
        current_tfp: np.ndarray,
        current_production: np.ndarray,
        current_unit_costs: np.ndarray,
        available_cash: np.ndarray,
        current_prices: np.ndarray,
        n_industries: int,
        input_usage: np.ndarray,
        current_tech_multipliers: np.ndarray,
        substitution_bundle_matrix: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plan productivity investment using simple rules.

        Invests a fixed fraction of available budget if expected cost savings
        exceed the hurdle rate. Allocates between TFP and technical investments.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers
            current_production (np.ndarray): Current production levels
            current_unit_costs (np.ndarray): Current unit costs of production
            available_cash (np.ndarray): Cash available for investment
            current_prices (np.ndarray): Current market prices by industry
            n_industries (int): Number of industries
            input_usage (np.ndarray): Used intermediate inputs [n_firms x n_industries]
            current_tech_multipliers (np.ndarray): Technical coefficient multipliers [n_firms x n_industries]
            substitution_bundle_matrix (np.ndarray | None): Substitution bundle matrix

        Returns:
            tuple: (total_investment, tfp_investment, technical_investment_by_input)
        """
        # Compute available budget
        budget = self.compute_investment_budget(available_cash, current_production)

        # Candidate investment (fraction of budget)
        candidate_investment = self.investment_propensity * budget

        # Compute hurdle-adjusted present value of cost savings
        hurdle_value = self.compute_hurdle_adjusted_value(candidate_investment, current_production, current_unit_costs)

        # Investment is profitable if hurdle-adjusted value exceeds investment cost
        profitable = hurdle_value > candidate_investment

        # Only invest where profitable
        total_investment = np.where(profitable, candidate_investment, 0)

        if substitution_bundle_matrix is not None:
            # Allocate between TFP and technical using bundle-aware logic
            tfp_investment, technical_investment = self.allocate_productivity_investment(
                total_investment,
                current_prices,
                input_usage,
                current_tech_multipliers,
                substitution_bundle_matrix,
            )
        else:
            # Fallback to simple allocation without bundle logic
            tfp_investment = self.tfp_investment_share * total_investment
            technical_budget = (1.0 - self.tfp_investment_share) * total_investment
            technical_investment = np.zeros((len(current_production), n_industries))
            # Distribute technical budget evenly across inputs for simplicity
            if n_industries > 0:
                technical_investment[:, :] = technical_budget[:, np.newaxis] / n_industries

        return total_investment, tfp_investment, technical_investment


class OptimalProductivityInvestmentPlanner(ProductivityInvestmentPlanner):
    """Optimal productivity investment planning based on expected returns.

    This implementation solves for the optimal investment level that
    maximizes expected returns subject to budget and hurdle rate constraints.
    """

    def __init__(
        self,
        n_firms: int,
        hurdle_rate: float | list[float] = 0.15,
        max_investment_fraction: float | list[float] = 0.1,
        investment_effectiveness: float | list[float] = 0.1,
        investment_elasticity: float | list[float] = 0.3,
        search_steps: int = 20,
        # New parameters for technical coefficient investment
        tfp_investment_share: float | list[float] = 0.4,
        technical_investment_effectiveness: float | list[float] = 0.15,
        technical_diminishing_returns: float | list[float] = 0.5,
        price_weight: float | list[float] = 0.4,
        usage_weight: float | list[float] = 0.3,
        potential_weight: float | list[float] = 0.3,
    ):
        """Initialize the optimal productivity investment planner.

        Args:
            n_firms (int): Number of firms
            hurdle_rate (float | list[float]): Minimum required return on investment
            max_investment_fraction (float | list[float]): Max investment as fraction of output
            investment_effectiveness (float | list[float]): TFP growth effectiveness parameter
            investment_elasticity (float | list[float]): Diminishing returns parameter
            search_steps (int): Number of steps in optimization search
        """
        super().__init__(
            n_firms,
            hurdle_rate,
            max_investment_fraction,
            investment_effectiveness,
            investment_elasticity,
            tfp_investment_share,
            technical_investment_effectiveness,
            technical_diminishing_returns,
            price_weight,
            usage_weight,
            potential_weight,
        )
        self.search_steps = search_steps

    def plan_productivity_investment(
        self,
        current_tfp: np.ndarray,
        current_production: np.ndarray,
        current_unit_costs: np.ndarray,
        available_cash: np.ndarray,
        current_prices: np.ndarray,
        n_industries: int,
        input_usage: np.ndarray,
        current_tech_multipliers: np.ndarray,
        substitution_bundle_matrix: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plan productivity investment by optimizing expected returns.

        Searches for the investment level that maximizes net present value
        from cost savings subject to constraints. Allocates between TFP and technical.

        Args:
            current_tfp (np.ndarray): Current TFP multipliers
            current_production (np.ndarray): Current production levels
            current_unit_costs (np.ndarray): Current unit costs of production
            available_cash (np.ndarray): Cash available for investment
            current_prices (np.ndarray): Current market prices by industry
            n_industries (int): Number of industries
            input_usage (np.ndarray): Used intermediate inputs [n_firms x n_industries]
            current_tech_multipliers (np.ndarray): Technical coefficient multipliers [n_firms x n_industries]
            substitution_bundle_matrix (np.ndarray | None): Substitution bundle matrix

        Returns:
            tuple: (total_investment, tfp_investment, technical_investment_by_input)
        """
        n_firms = len(current_production)
        total_investment = np.zeros(n_firms)

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

                hurdle_value = self.compute_hurdle_adjusted_value(inv_array, prod_array, costs_array)[0]

                # NPV: hurdle-adjusted value minus investment cost
                npv = hurdle_value - investment

                # Only invest if NPV is positive and better than current best
                if npv > best_npv:
                    best_npv = npv
                    best_investment = investment

            total_investment[i] = best_investment

        if substitution_bundle_matrix is not None:
            # Allocate between TFP and technical using bundle-aware logic
            tfp_investment, technical_investment = self.allocate_productivity_investment(
                total_investment,
                current_prices,
                input_usage,
                current_tech_multipliers,
                substitution_bundle_matrix,
            )
        else:
            # Fallback to simple allocation without bundle logic
            tfp_investment = self.tfp_investment_share * total_investment
            technical_budget = (1.0 - self.tfp_investment_share) * total_investment
            technical_investment = np.zeros((len(current_production), n_industries))
            # Distribute technical budget evenly across inputs for simplicity
            if n_industries > 0:
                technical_investment[:, :] = technical_budget[:, np.newaxis] / n_industries

        return total_investment, tfp_investment, technical_investment

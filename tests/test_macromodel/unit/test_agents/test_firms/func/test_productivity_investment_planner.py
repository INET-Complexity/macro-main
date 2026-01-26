import numpy as np
import pytest

from macromodel.agents.firms.func.productivity_investment_planner import (
    NoProductivityInvestmentPlanner,
    OptimalProductivityInvestmentPlanner,
    SimpleProductivityInvestmentPlanner,
)


def create_test_params(n_firms, n_industries=18):
    """Create default test parameters for productivity investment planning."""
    return {
        "current_prices": np.ones(n_industries),
        "n_industries": n_industries,
        "input_usage": np.ones((n_firms, n_industries)),
        "current_tech_multipliers": np.ones((n_firms, n_industries)),
        "substitution_bundle_matrix": None,
    }


class TestNoProductivityInvestmentPlanner:
    """Test the NoProductivityInvestmentPlanner implementation."""

    def test_always_returns_zero_investment(self):
        """Test that no investment planner always returns zero."""
        n_firms = 5
        planner = NoProductivityInvestmentPlanner(n_firms=n_firms)

        current_tfp = np.ones(n_firms)
        current_production = np.full(n_firms, 100.0)
        current_unit_costs = np.full(n_firms, 5.0)
        available_cash = np.full(n_firms, 1000.0)

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(n_firms),
        )

        # Should return all zeros
        expected_1d = np.zeros(n_firms)
        expected_2d = np.zeros((n_firms, technical_investment.shape[1]))  # Match actual shape
        assert np.allclose(total_investment, expected_1d)
        assert np.allclose(tfp_investment, expected_1d)
        assert np.allclose(technical_investment, expected_2d)

    def test_ignores_all_parameters(self):
        """Test that no investment planner ignores all input parameters."""
        planner = NoProductivityInvestmentPlanner(n_firms=1)

        # Test with extreme values
        current_tfp = np.array([10.0])
        current_production = np.array([1000.0])
        current_unit_costs = np.array([100.0])
        available_cash = np.array([10000.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        assert np.allclose(total_investment, [0.0])
        assert np.allclose(tfp_investment, [0.0])
        assert np.allclose(technical_investment, [0.0])


class TestSimpleProductivityInvestmentPlanner:
    """Test the SimpleProductivityInvestmentPlanner implementation."""

    def test_basic_investment_calculation_with_cost_savings(self):
        """Test basic investment calculation using cost savings logic."""
        planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.10,
            max_investment_fraction=0.1,
            investment_effectiveness=0.1,
            investment_elasticity=0.3,
            investment_propensity=0.5,
            n_firms=3,
        )

        n_firms = 3
        current_tfp = np.ones(n_firms)
        current_production = np.array([100.0, 200.0, 150.0])
        current_unit_costs = np.array([8.0, 12.0, 10.0])  # Unit costs for cost savings calculation
        available_cash = np.array([500.0, 800.0, 600.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(n_firms),
        )

        # All values should be non-negative
        assert np.all(total_investment >= 0)
        assert np.all(tfp_investment >= 0)
        assert np.all(technical_investment >= 0)
        # Should not exceed maximum investment constraints
        max_investment = np.minimum(0.5 * available_cash, 0.1 * current_production)
        assert np.all(total_investment <= max_investment + 1e-10)

    def test_higher_unit_costs_encourage_investment(self):
        """Test that firms with higher unit costs invest more (higher returns from cost savings)."""
        planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.05,  # Low hurdle rate to allow investment
            investment_propensity=0.5,
            investment_effectiveness=0.1,
            n_firms=2,
        )

        # Two identical firms except for unit costs
        current_tfp = np.array([1.0, 1.0])
        current_production = np.array([100.0, 100.0])
        current_unit_costs = np.array([5.0, 15.0])  # Second firm has higher costs
        available_cash = np.array([1000.0, 1000.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(2),
        )

        # Firm with higher unit costs should invest more (higher returns from cost reduction)
        assert total_investment[1] >= total_investment[0]

    def test_hurdle_rate_constraint_with_cost_savings(self):
        """Test that investments are rejected when cost savings don't meet hurdle rate."""
        planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.50,  # Very high hurdle rate (50%)
            max_investment_fraction=0.1,
            investment_effectiveness=0.00,  # no effectiveness
            investment_elasticity=0.3,
            investment_propensity=0.5,
            n_firms=1,
        )

        current_tfp = np.array([1.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([2.0])  # Low unit costs = low savings potential
        available_cash = np.array([500.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        # High hurdle rate + low effectiveness + low unit costs should lead to no investment
        assert np.allclose(total_investment, [0.0])
        assert np.allclose(tfp_investment, [0.0])
        assert np.allclose(technical_investment, [0.0])

    def test_budget_constraints(self):
        """Test that investment respects budget constraints."""
        planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.05,  # Low hurdle rate
            max_investment_fraction=0.2,  # Allow more investment relative to output
            investment_propensity=1.0,  # Try to invest all available budget
            n_firms=1,
        )

        current_tfp = np.array([1.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([20.0])  # High unit costs = high savings potential
        available_cash = np.array([5.0])  # Very limited cash

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        # Investment should be limited by cash constraint
        cash_constraint = 0.5 * available_cash[0]  # Default max_cash_fraction = 0.5
        output_constraint = 0.2 * current_production[0]
        expected_budget = min(cash_constraint, output_constraint)

        # Investment should not exceed the more restrictive constraint
        assert total_investment[0] <= expected_budget + 1e-10

    def test_zero_unit_costs_no_investment(self):
        """Test that zero unit costs lead to no investment (no cost savings possible)."""
        planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.05,
            investment_propensity=1.0,
            n_firms=1,
        )

        current_tfp = np.array([1.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([0.0])  # Zero unit costs
        available_cash = np.array([500.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        # No unit costs means no cost savings possible
        assert np.allclose(total_investment, [0.0])
        assert np.allclose(tfp_investment, [0.0])
        assert np.allclose(technical_investment, [0.0])

    def test_zero_cash_handling(self):
        """Test behavior when firms have no cash available."""
        planner = SimpleProductivityInvestmentPlanner(n_firms=1)

        current_tfp = np.array([1.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([10.0])
        available_cash = np.array([0.0])  # No cash available

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        assert np.allclose(total_investment, [0.0])
        assert np.allclose(tfp_investment, [0.0])
        assert np.allclose(technical_investment, [0.0])

    def test_hurdle_value_calculation_correctness(self):
        """Test that the hurdle-adjusted value calculation is mathematically correct."""
        planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.15,
            investment_effectiveness=0.1,
            investment_elasticity=0.5,
            n_firms=1,
        )

        productivity_investment = np.array([10.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([20.0])

        # Calculate expected value manually following investment_decision.md
        investment_intensity = productivity_investment / current_production  # 0.1
        tfp_growth = 0.1 * np.power(investment_intensity, 0.5)  # 0.1 * sqrt(0.1) ≈ 0.0316
        cost_reduction_per_period = tfp_growth * current_unit_costs * current_production
        hurdle_discount = (1 + 0.15) / 0.15  # (1 + r_h) / r_h
        expected_value = cost_reduction_per_period * hurdle_discount

        # Test the planner's calculation
        computed_value = planner.compute_hurdle_adjusted_value(
            productivity_investment, current_production, current_unit_costs
        )

        assert np.allclose(computed_value, expected_value)


class TestOptimalProductivityInvestmentPlanner:
    """Test the OptimalProductivityInvestmentPlanner implementation."""

    def test_optimal_investment_basic(self):
        """Test basic optimal investment calculation."""
        planner = OptimalProductivityInvestmentPlanner(
            hurdle_rate=0.10,
            search_steps=50,  # More steps for better optimization
            n_firms=1,
        )

        current_tfp = np.array([1.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([15.0])  # Good cost savings potential
        available_cash = np.array([500.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        assert np.all(total_investment >= 0)
        assert np.all(tfp_investment >= 0)
        assert np.all(technical_investment >= 0)
        # Should respect budget constraint
        budget = planner.compute_investment_budget(available_cash, current_production)
        assert total_investment[0] <= budget[0] + 1e-10

    def test_optimal_better_than_simple(self):
        """Test that optimal planner finds better solutions than simple planner."""
        # Use same parameters for fair comparison
        common_params = {
            "hurdle_rate": 0.15,
            "max_investment_fraction": 0.1,
            "investment_effectiveness": 0.1,
            "investment_elasticity": 0.3,
            "n_firms": 1,
        }

        optimal_planner = OptimalProductivityInvestmentPlanner(
            **common_params,
            search_steps=30,
        )

        simple_planner = SimpleProductivityInvestmentPlanner(
            **common_params,
            investment_propensity=0.5,
        )

        # Set up scenario with clear optimization potential
        current_tfp = np.array([1.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([12.0])
        available_cash = np.array([500.0])

        optimal_total, optimal_tfp, optimal_tech = optimal_planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        simple_total, simple_tfp, simple_tech = simple_planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        # Calculate NPV for both solutions
        def calculate_npv(investment):
            if investment == 0:
                return 0
            hurdle_value = optimal_planner.compute_hurdle_adjusted_value(
                np.array([investment]), current_production, current_unit_costs
            )[0]
            return hurdle_value - investment

        optimal_npv = calculate_npv(optimal_total[0])
        simple_npv = calculate_npv(simple_total[0])

        # Optimal should achieve at least as good NPV as simple
        assert optimal_npv >= simple_npv - 1e-6

    def test_multiple_firms_optimization(self):
        """Test optimization with multiple firms."""
        planner = OptimalProductivityInvestmentPlanner(
            hurdle_rate=0.12,
            search_steps=20,
            n_firms=4,
        )

        n_firms = 4
        current_tfp = np.ones(n_firms)
        current_production = np.array([80.0, 120.0, 100.0, 150.0])
        current_unit_costs = np.array([8.0, 12.0, 10.0, 15.0])
        available_cash = np.array([400.0, 600.0, 500.0, 800.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(n_firms),
        )

        assert len(total_investment) == n_firms
        assert len(tfp_investment) == n_firms
        assert len(technical_investment) == n_firms
        assert np.all(total_investment >= 0)
        assert np.all(tfp_investment >= 0)
        assert np.all(technical_investment >= 0)

        # Each firm's investment should respect individual budget constraints
        budgets = planner.compute_investment_budget(available_cash, current_production)
        assert np.all(total_investment <= budgets + 1e-10)

    def test_no_investment_when_unprofitable(self):
        """Test that optimal planner invests nothing when unprofitable."""
        planner = OptimalProductivityInvestmentPlanner(
            hurdle_rate=0.50,  # Very high hurdle rate
            investment_effectiveness=0.00,  # no effectiveness
            n_firms=1,
        )

        current_tfp = np.array([1.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([2.0])  # Low costs = limited savings potential
        available_cash = np.array([200.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        # Should find no profitable investment
        assert np.allclose(total_investment, [0.0])
        assert np.allclose(tfp_investment, [0.0])
        assert np.allclose(technical_investment, [0.0])


class TestProductivityInvestmentPlannerUtilities:
    """Test utility methods shared across all planners."""

    def test_compute_expected_tfp_growth(self):
        """Test TFP growth calculation from investment."""
        planner = SimpleProductivityInvestmentPlanner(
            investment_effectiveness=0.1,
            investment_elasticity=0.3,
            n_firms=3,
        )

        productivity_investment = np.array([10.0, 20.0, 0.0])
        current_production = np.array([100.0, 200.0, 150.0])

        tfp_growth = planner.compute_expected_tfp_growth(productivity_investment, current_production)

        # Calculate expected manually
        investment_intensity = productivity_investment / current_production
        expected = 0.1 * np.power(investment_intensity, 0.3)

        assert np.allclose(tfp_growth, expected)

    def test_compute_hurdle_adjusted_value_with_cost_savings(self):
        """Test hurdle-adjusted value calculation using cost savings logic."""
        planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.15,
            investment_effectiveness=0.1,
            investment_elasticity=0.3,
            n_firms=2,
        )

        productivity_investment = np.array([5.0, 10.0])
        current_production = np.array([100.0, 200.0])
        current_unit_costs = np.array([10.0, 20.0])

        hurdle_values = planner.compute_hurdle_adjusted_value(
            productivity_investment, current_production, current_unit_costs
        )

        # Hurdle values should be positive for positive inputs
        assert np.all(hurdle_values >= 0)
        # Firm with higher costs and production should have higher value
        assert hurdle_values[1] > hurdle_values[0]

    def test_compute_investment_budget(self):
        """Test investment budget calculation."""
        planner = SimpleProductivityInvestmentPlanner(
            max_investment_fraction=0.15,
            n_firms=3,
        )

        available_cash = np.array([1000.0, 500.0, 2000.0])
        current_production = np.array([100.0, 200.0, 80.0])
        max_cash_fraction = 0.4

        budget = planner.compute_investment_budget(available_cash, current_production, max_cash_fraction)

        # Should be minimum of cash and output constraints
        cash_constraint = max_cash_fraction * available_cash
        output_constraint = 0.15 * current_production
        expected = np.minimum(cash_constraint, output_constraint)

        assert np.allclose(budget, expected)

    def test_zero_production_handling(self):
        """Test graceful handling of zero production."""
        planner = SimpleProductivityInvestmentPlanner(
            n_firms=2,
        )

        current_tfp = np.array([1.0, 1.0])
        current_production = np.array([100.0, 0.0])  # Second firm has zero production
        current_unit_costs = np.array([10.0, 12.0])
        available_cash = np.array([500.0, 300.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(2),
        )

        # No NaN or inf values
        assert not np.isnan(total_investment).any()
        assert not np.isinf(total_investment).any()
        assert not np.isnan(tfp_investment).any()
        assert not np.isinf(tfp_investment).any()
        assert not np.isnan(technical_investment).any()
        assert not np.isinf(technical_investment).any()
        # Zero production firm should get zero investment
        assert total_investment[1] == 0.0
        assert tfp_investment[1] == 0.0
        assert (technical_investment[1] == 0.0).all()  # type: ignore

    def test_negative_cash_handling(self):
        """Test handling of negative cash balances."""
        planner = SimpleProductivityInvestmentPlanner(
            n_firms=1,
        )

        current_tfp = np.array([1.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([10.0])
        available_cash = np.array([-100.0])  # Negative cash

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(1),
        )

        # Should handle negative cash gracefully (no investment possible)
        assert np.allclose(total_investment, [0.0])
        assert np.allclose(tfp_investment, [0.0])
        assert np.allclose(technical_investment, [0.0])

    def test_empty_arrays(self):
        """Test with empty input arrays."""
        planner = SimpleProductivityInvestmentPlanner(n_firms=0)

        current_tfp = np.array([])
        current_production = np.array([])
        current_unit_costs = np.array([])
        available_cash = np.array([])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(0),
        )

        assert len(total_investment) == 0
        assert len(tfp_investment) == 0
        assert len(technical_investment) == 0
        assert isinstance(total_investment, np.ndarray)
        assert isinstance(tfp_investment, np.ndarray)
        assert isinstance(technical_investment, np.ndarray)

    def test_cost_savings_vs_production_scaling(self):
        """Test that hurdle values scale properly with production and unit costs."""
        planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.15,
            investment_effectiveness=0.1,
            investment_elasticity=0.5,
            n_firms=2,
        )

        # Same investment intensity for both firms
        productivity_investment = np.array([5.0, 10.0])
        current_production = np.array([100.0, 200.0])  # Second firm produces 2x more
        current_unit_costs = np.array([10.0, 10.0])  # Same unit costs

        hurdle_values = planner.compute_hurdle_adjusted_value(
            productivity_investment, current_production, current_unit_costs
        )

        # With same investment intensity and unit costs, values should scale with production
        # (since total cost savings = cost_savings_per_period * production)
        ratio = hurdle_values[1] / hurdle_values[0]
        assert abs(ratio - 2.0) < 0.1  # Should be approximately 2x

    def test_investment_effectiveness_impact(self):
        """Test that investment effectiveness parameter affects hurdle values correctly."""
        low_effectiveness_planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.15,
            investment_effectiveness=0.05,
            investment_elasticity=0.3,
            n_firms=1,
        )
        high_effectiveness_planner = SimpleProductivityInvestmentPlanner(
            hurdle_rate=0.15,
            investment_effectiveness=0.15,
            investment_elasticity=0.3,
            n_firms=1,
        )

        productivity_investment = np.array([10.0])
        current_production = np.array([100.0])
        current_unit_costs = np.array([15.0])

        low_values = low_effectiveness_planner.compute_hurdle_adjusted_value(
            productivity_investment, current_production, current_unit_costs
        )
        high_values = high_effectiveness_planner.compute_hurdle_adjusted_value(
            productivity_investment, current_production, current_unit_costs
        )

        # Higher effectiveness should lead to higher hurdle-adjusted values
        assert high_values[0] > low_values[0]


class TestHeterogeneousParameters:
    """Test heterogeneous parameter functionality across firms."""

    def test_heterogeneous_investment_effectiveness(self):
        """Test that different firms can have different investment effectiveness."""
        n_firms = 3
        planner = SimpleProductivityInvestmentPlanner(
            n_firms=n_firms,
            investment_effectiveness=[0.05, 0.10, 0.15],  # Different effectiveness per firm
            hurdle_rate=0.01,  # Low hurdle rate to ensure investment
            investment_propensity=1.0,
        )

        # Verify parameters are arrays with correct values
        assert isinstance(planner.investment_effectiveness, np.ndarray)
        assert len(planner.investment_effectiveness) == n_firms
        assert np.allclose(planner.investment_effectiveness, [0.05, 0.10, 0.15])

        # Test that effectiveness affects investment decisions differently
        current_tfp = np.ones(n_firms)
        current_production = np.array([100.0, 100.0, 100.0])  # Same production
        current_unit_costs = np.array([10.0, 10.0, 10.0])  # Same costs
        available_cash = np.array([1000.0, 1000.0, 1000.0])  # Same cash

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(n_firms),
        )

        # Firms with higher effectiveness should get higher expected returns
        # and potentially invest more (depending on optimization)
        assert np.all(total_investment >= 0)
        assert len(total_investment) == n_firms

    def test_heterogeneous_hurdle_rate(self):
        """Test that different firms can have different hurdle rates."""
        n_firms = 3
        planner = SimpleProductivityInvestmentPlanner(
            n_firms=n_firms,
            hurdle_rate=[0.05, 0.15, 0.30],  # Very different hurdle rates
            investment_effectiveness=0.1,
            investment_propensity=1.0,
        )

        # Verify parameters
        assert isinstance(planner.hurdle_rate, np.ndarray)
        assert np.allclose(planner.hurdle_rate, [0.05, 0.15, 0.30])

        current_tfp = np.ones(n_firms)
        current_production = np.array([100.0, 100.0, 100.0])
        current_unit_costs = np.array([10.0, 10.0, 10.0])
        available_cash = np.array([500.0, 500.0, 500.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(n_firms),
        )

        # Firm with low hurdle rate (0.05) should invest more than firm with high hurdle rate (0.30)
        # because more investments pass the hurdle
        assert total_investment[0] >= total_investment[2] or total_investment[2] == 0

    def test_mixed_heterogeneous_and_uniform(self):
        """Test mixing heterogeneous and uniform parameters."""
        n_firms = 4
        planner = SimpleProductivityInvestmentPlanner(
            n_firms=n_firms,
            hurdle_rate=[0.10, 0.12, 0.15, 0.20],  # Heterogeneous
            investment_effectiveness=0.1,  # Uniform
            investment_elasticity=[0.2, 0.3, 0.4, 0.5],  # Heterogeneous
            max_investment_fraction=0.15,  # Uniform
        )

        # Check parameter types and shapes
        assert isinstance(planner.hurdle_rate, np.ndarray)
        assert len(planner.hurdle_rate) == n_firms
        assert isinstance(planner.investment_effectiveness, np.ndarray)
        assert len(planner.investment_effectiveness) == n_firms
        assert np.allclose(planner.investment_effectiveness, 0.1)  # All same value
        assert isinstance(planner.investment_elasticity, np.ndarray)
        assert np.allclose(planner.investment_elasticity, [0.2, 0.3, 0.4, 0.5])

    def test_heterogeneous_technical_parameters(self):
        """Test heterogeneous parameters for technical coefficient investment."""
        n_firms = 3
        planner = SimpleProductivityInvestmentPlanner(
            n_firms=n_firms,
            technical_investment_effectiveness=[0.10, 0.15, 0.20],
            technical_diminishing_returns=[0.3, 0.5, 0.7],
            price_weight=[0.5, 0.4, 0.3],
            usage_weight=[0.3, 0.3, 0.4],
            potential_weight=[0.2, 0.3, 0.3],
            tfp_investment_share=0.5,  # 50/50 split
        )

        # Verify arrays
        assert np.allclose(planner.technical_investment_effectiveness, [0.10, 0.15, 0.20])
        assert np.allclose(planner.technical_diminishing_returns, [0.3, 0.5, 0.7])
        assert np.allclose(planner.price_weight, [0.5, 0.4, 0.3])
        assert np.allclose(planner.usage_weight, [0.3, 0.3, 0.4])
        assert np.allclose(planner.potential_weight, [0.2, 0.3, 0.3])

    def test_optimal_planner_with_heterogeneous_params(self):
        """Test that OptimalProductivityInvestmentPlanner works with heterogeneous parameters."""
        n_firms = 2
        planner = OptimalProductivityInvestmentPlanner(
            n_firms=n_firms,
            hurdle_rate=[0.10, 0.15],
            investment_effectiveness=[0.08, 0.12],
            search_steps=10,
        )

        # Verify parameters
        assert np.allclose(planner.hurdle_rate, [0.10, 0.15])
        assert np.allclose(planner.investment_effectiveness, [0.08, 0.12])

        current_tfp = np.ones(n_firms)
        current_production = np.array([100.0, 150.0])
        current_unit_costs = np.array([10.0, 12.0])
        available_cash = np.array([500.0, 600.0])

        total_investment, tfp_investment, technical_investment = planner.plan_productivity_investment(
            current_tfp=current_tfp,
            current_production=current_production,
            current_unit_costs=current_unit_costs,
            available_cash=available_cash,
            **create_test_params(n_firms),
        )

        # Should produce valid investment decisions
        assert len(total_investment) == n_firms
        assert np.all(total_investment >= 0)
        assert np.all(np.isfinite(total_investment))

    def test_parameter_validation_wrong_length(self):
        """Test that wrong length parameters raise an error."""
        with pytest.raises(ValueError, match="does not match n_firms"):
            SimpleProductivityInvestmentPlanner(
                n_firms=3,
                hurdle_rate=[0.10, 0.15],  # Only 2 values for 3 firms
            )

"""Test bundle-aware arbitrage functionality in productivity investment planners."""

import numpy as np

from macromodel.agents.firms.func.productivity_investment_planner import (
    SimpleProductivityInvestmentPlanner,
)
from macromodel.agents.firms.utils.create_bundle_matrix import create_bundle_matrix
from macromodel.configurations.firms_configuration import create_good_bundle


class TestBundleArbitrageInvestmentAllocation:
    """Test bundle-aware arbitrage in investment allocation."""

    def test_bundle_arbitrage_with_price_differences(self):
        """Test that bundle arbitrage favors cheaper inputs within significant bundles."""

        # Create investment planner with bundle arbitrage enabled
        planner = SimpleProductivityInvestmentPlanner(
            tfp_investment_share=0.0,  # 100% to technical for easier testing
            price_weight=0.5,
            usage_weight=0.3,
            potential_weight=0.2,
            n_firms=2,
        )

        # Setup test scenario: 2 firms, 4 industries
        n_firms, n_industries = 2, 4

        # Create bundle: industries 0,1 are in energy bundle, others are singleton bundles
        bundles = [[0, 1]]  # Energy bundle with oil (industry 0) and electricity (industry 1)
        good_bundle = create_good_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(good_bundle))

        # Set prices: oil expensive ($100), electricity cheap ($20)
        # Other industries have moderate prices
        current_prices = np.array([100.0, 20.0, 50.0, 60.0])  # oil, electricity, coal, gas

        # Firm spending patterns - both firms spend significantly on energy bundle
        input_usage = np.array(
            [
                [50.0, 40.0, 10.0, 5.0],  # Firm 0: heavy energy user (90 of 105 total)
                [30.0, 20.0, 15.0, 10.0],  # Firm 1: moderate energy user (50 of 75 total)
            ]
        )

        # Current tech multipliers (all equal for simplicity)
        current_tech_multipliers = np.ones((n_firms, n_industries))

        # Test investment allocation
        total_investment = np.array([1000.0, 800.0])  # Different investment budgets

        tfp_investment, technical_investment = planner.allocate_productivity_investment(
            total_investment=total_investment,
            current_prices=current_prices,
            input_usage=input_usage,
            current_tech_multipliers=current_tech_multipliers,
            substitution_bundle_matrix=bundle_matrix,
            bundle_significance_threshold=0.1,
            arbitrage_intensity=2.0,
        )

        # Verify TFP investment is zero (as configured)
        assert np.allclose(tfp_investment, 0.0)

        # Check that technical investment sums to total budget for each firm
        technical_sums = technical_investment.sum(axis=1)
        assert np.allclose(technical_sums, total_investment)

        # Key test: Within energy bundle, electricity should get more investment than oil
        # due to bundle arbitrage (electricity is 5x cheaper than oil)
        for firm_idx in range(n_firms):
            oil_investment = technical_investment[firm_idx, 0]
            electricity_investment = technical_investment[firm_idx, 1]

            # Electricity should get significantly more investment than oil
            assert electricity_investment > oil_investment, (
                f"Firm {firm_idx}: Electricity investment ({electricity_investment:.2f}) "
                f"should exceed oil investment ({oil_investment:.2f}) due to bundle arbitrage"
            )

            # With arbitrage_intensity=2.0 and 5x price difference,
            # electricity should get roughly (100/20)^2 = 25x more priority
            arbitrage_ratio = electricity_investment / (oil_investment + 1e-10)
            assert arbitrage_ratio > 5.0, (
                f"Firm {firm_idx}: Arbitrage ratio {arbitrage_ratio:.2f} should be significant "
                f"given 5x price difference and arbitrage_intensity=2.0"
            )

    def test_bundle_arbitrage_significance_threshold(self):
        """Test that bundle arbitrage only applies to firms with significant spending."""

        planner = SimpleProductivityInvestmentPlanner(
            tfp_investment_share=0.0,
            n_firms=2,
        )

        n_firms, n_industries = 2, 4
        bundles = [[0, 1]]  # Energy bundle
        good_bundle = create_good_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(good_bundle))

        current_prices = np.array([100.0, 20.0, 50.0, 60.0])

        # Firm 0: High energy spending (>30%), Firm 1: Low energy spending (<30%)
        input_usage = np.array(
            [
                [40.0, 30.0, 10.0, 10.0],  # Firm 0: 70/90 = 77% energy spending
                [5.0, 5.0, 40.0, 40.0],  # Firm 1: 10/90 = 11% energy spending
            ]
        )

        current_tech_multipliers = np.ones((n_firms, n_industries))
        total_investment = np.array([1000.0, 1000.0])

        tfp_investment, technical_investment = planner.allocate_productivity_investment(
            total_investment=total_investment,
            current_prices=current_prices,
            input_usage=input_usage,
            current_tech_multipliers=current_tech_multipliers,
            substitution_bundle_matrix=bundle_matrix,
            bundle_significance_threshold=0.3,
            arbitrage_intensity=2.0,
        )

        # Firm 0 should show strong arbitrage (high energy spending)
        firm0_oil = technical_investment[0, 0]
        firm0_electricity = technical_investment[0, 1]
        assert firm0_electricity > 2.0 * firm0_oil, "Firm 0 should show strong bundle arbitrage effect"

        # Firm 1 should show weak/no arbitrage (low energy spending)
        firm1_oil = technical_investment[1, 0]
        firm1_electricity = technical_investment[1, 1]
        arbitrage_ratio_firm1 = firm1_electricity / (firm1_oil + 1e-10)
        arbitrage_ratio_firm0 = firm0_electricity / (firm0_oil + 1e-10)

        assert arbitrage_ratio_firm1 < arbitrage_ratio_firm0, (
            f"Firm 1 arbitrage ratio ({arbitrage_ratio_firm1:.2f}) should be less than "
            f"Firm 0 arbitrage ratio ({arbitrage_ratio_firm0:.2f}) due to spending threshold"
        )

    def test_singleton_bundles_no_arbitrage(self):
        """Test that singleton bundles don't apply arbitrage (no substitution opportunity)."""

        planner = SimpleProductivityInvestmentPlanner(tfp_investment_share=0.0, n_firms=1)

        n_firms, n_industries = 1, 3
        # All singleton bundles (no multi-input bundles)
        good_bundle = create_good_bundle(n_industries, [])
        bundle_matrix = create_bundle_matrix(np.array(good_bundle))

        # Moderate price differences
        current_prices = np.array([100.0, 50.0, 75.0])

        # Equal usage for all inputs
        input_usage = np.array([[100.0, 100.0, 100.0]])
        current_tech_multipliers = np.ones((n_firms, n_industries))
        total_investment = np.array([900.0])

        # Test with both low and high arbitrage intensity to show it has no effect
        tfp_low, tech_low = planner.allocate_productivity_investment(
            total_investment=total_investment,
            current_prices=current_prices,
            input_usage=input_usage,
            current_tech_multipliers=current_tech_multipliers,
            substitution_bundle_matrix=bundle_matrix,
            bundle_significance_threshold=0.1,
            arbitrage_intensity=1.0,  # Low arbitrage
        )

        tfp_high, tech_high = planner.allocate_productivity_investment(
            total_investment=total_investment,
            current_prices=current_prices,
            input_usage=input_usage,
            current_tech_multipliers=current_tech_multipliers,
            substitution_bundle_matrix=bundle_matrix,
            bundle_significance_threshold=0.1,
            arbitrage_intensity=10.0,  # High arbitrage
        )

        # With singleton bundles, arbitrage intensity should make no difference
        # since there are no substitution opportunities
        assert np.allclose(tech_low, tech_high, rtol=1e-10), (
            "Singleton bundles should be unaffected by arbitrage intensity"
        )

        # Investment should follow base priority logic (expensive inputs get higher priority)
        investments = tech_low[0]
        assert investments[0] > investments[2] > investments[1], (
            f"Priority should follow price: expensive ($100) > medium ($75) > cheap ($50), "
            f"got {investments[0]:.1f} > {investments[2]:.1f} > {investments[1]:.1f}"
        )

    def test_multiple_bundles_independent_arbitrage(self):
        """Test arbitrage works independently across multiple bundles."""

        planner = SimpleProductivityInvestmentPlanner(
            tfp_investment_share=0.0,
            n_firms=1,
        )

        n_firms, n_industries = 1, 6
        # Two bundles: Energy [0,1] and Transport [2,3], plus singletons [4], [5]
        bundles = [[0, 1], [2, 3]]
        good_bundle = create_good_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(good_bundle))

        # Prices: Within each bundle, one cheap and one expensive option
        current_prices = np.array([100.0, 25.0, 80.0, 20.0, 50.0, 60.0])
        #                         [oil,  elec, car,  bike, food, rent]

        # Firm spends significantly on both energy and transport bundles
        input_usage = np.array([[40.0, 30.0, 25.0, 15.0, 5.0, 5.0]])  # 120 total
        current_tech_multipliers = np.ones((n_firms, n_industries))
        total_investment = np.array([1200.0])

        tfp_investment, technical_investment = planner.allocate_productivity_investment(
            total_investment=total_investment,
            current_prices=current_prices,
            input_usage=input_usage,
            current_tech_multipliers=current_tech_multipliers,
            substitution_bundle_matrix=bundle_matrix,
            bundle_significance_threshold=0.1,
            arbitrage_intensity=2.0,
        )

        investments = technical_investment[0]

        # Within energy bundle: electricity (1) should beat oil (0)
        assert investments[1] > investments[0], (
            f"Energy bundle: Electricity ({investments[1]:.2f}) should beat oil ({investments[0]:.2f})"
        )

        # Within transport bundle: bike (3) should beat car (2)
        assert investments[3] > investments[2], (
            f"Transport bundle: Bike ({investments[3]:.2f}) should beat car ({investments[2]:.2f})"
        )

        # Verify arbitrage strength based on price ratios
        energy_arbitrage = investments[1] / (investments[0] + 1e-10)
        transport_arbitrage = investments[3] / (investments[2] + 1e-10)

        # Both should show arbitrage, transport even stronger (4:1 vs 4:1 price ratio)
        assert energy_arbitrage > 2.0, f"Energy arbitrage ratio {energy_arbitrage:.2f} should be significant"
        assert transport_arbitrage > 2.0, f"Transport arbitrage ratio {transport_arbitrage:.2f} should be significant"

    def test_integration_with_base_priority_factors(self):
        """Test that bundle arbitrage integrates correctly with price/usage/potential factors."""

        planner = SimpleProductivityInvestmentPlanner(
            tfp_investment_share=0.0,
            price_weight=0.3,
            usage_weight=0.4,
            potential_weight=0.3,
            n_firms=1,
        )

        n_firms, n_industries = 1, 4
        bundles = [[0, 1]]  # Bundle with industries 0 and 1
        good_bundle = create_good_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(good_bundle))

        # Prices favor industry 1 within bundle
        current_prices = np.array([80.0, 40.0, 60.0, 70.0])

        # Usage heavily favors industry 0 (should compete with price factor)
        input_usage = np.array([[100.0, 10.0, 20.0, 15.0]])

        # Potential strongly favors industry 1 (low current multiplier)
        current_tech_multipliers = np.array([[1.0, 0.6, 1.0, 1.0]])  # Industry 1 has room for improvement

        total_investment = np.array([1000.0])

        tfp_investment, technical_investment = planner.allocate_productivity_investment(
            total_investment=total_investment,
            current_prices=current_prices,
            input_usage=input_usage,
            current_tech_multipliers=current_tech_multipliers,
            substitution_bundle_matrix=bundle_matrix,
            bundle_significance_threshold=0.1,
            arbitrage_intensity=1.5,
        )

        investments = technical_investment[0]

        # Industry 1 should win due to combination of:
        # - Bundle arbitrage (cheaper price within bundle)
        # - Price priority (cheaper overall)
        # - Improvement potential (lower multiplier)
        # Despite industry 0 having much higher usage
        assert investments[1] > investments[0], (
            f"Industry 1 ({investments[1]:.2f}) should beat industry 0 ({investments[0]:.2f}) "
            f"due to bundle arbitrage + price + potential factors overcoming usage disadvantage"
        )

        # Verify that base factors are still working outside the bundle
        # Industries 2,3 should follow standard priority logic
        assert investments[2] > 0 and investments[3] > 0, (
            "Non-bundle industries should still receive investment based on base factors"
        )

    def test_zero_investment_budget(self):
        """Test handling of zero investment budgets."""

        planner = SimpleProductivityInvestmentPlanner(n_firms=2)

        n_firms, n_industries = 2, 3
        bundles = [[0, 1]]
        good_bundle = create_good_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(good_bundle))

        current_prices = np.array([100.0, 20.0, 50.0])
        input_usage = np.ones((n_firms, n_industries))
        current_tech_multipliers = np.ones((n_firms, n_industries))

        # Zero investment budget
        total_investment = np.zeros(n_firms)

        tfp_investment, technical_investment = planner.allocate_productivity_investment(
            total_investment=total_investment,
            current_prices=current_prices,
            input_usage=input_usage,
            current_tech_multipliers=current_tech_multipliers,
            substitution_bundle_matrix=bundle_matrix,
        )

        # Should handle gracefully with all zeros
        assert np.allclose(tfp_investment, 0.0)
        assert np.allclose(technical_investment, 0.0)
        assert technical_investment.shape == (n_firms, n_industries)

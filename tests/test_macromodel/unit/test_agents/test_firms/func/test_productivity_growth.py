import numpy as np

from macromodel.agents.firms.func.productivity_growth import (
    SectoralTFPGrowth,
    SimpleTFPGrowth,
    StochasticTFPGrowth,
)


class TestSimpleTFPGrowth:
    """Test the SimpleTFPGrowth implementation."""

    def test_base_growth_only(self):
        """Test TFP growth with only base growth rate."""
        growth_func = SimpleTFPGrowth()
        n_firms = 5
        current_tfp = np.ones(n_firms)
        production = np.full(n_firms, 100.0)
        productivity_investment = np.zeros(n_firms)  # No investment
        base_growth = 0.0025  # 0.25% quarterly
        elasticity = 0.3

        tfp_growth = growth_func.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        # Should return base growth for all firms
        expected = np.full(n_firms, base_growth)
        assert np.allclose(tfp_growth, expected)

    def test_investment_driven_growth(self):
        """Test TFP growth with investment."""
        effectiveness = 0.1
        growth_func = SimpleTFPGrowth(investment_effectiveness=effectiveness)
        n_firms = 3
        current_tfp = np.ones(n_firms)
        production = np.array([100.0, 200.0, 150.0])
        productivity_investment = np.array([10.0, 20.0, 0.0])
        base_growth = 0.0025
        elasticity = 0.5

        tfp_growth = growth_func.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        # Calculate expected growth
        investment_intensity = productivity_investment / production
        expected = base_growth + effectiveness * np.power(investment_intensity, elasticity)

        assert np.allclose(tfp_growth, expected)

    def test_zero_production_handling(self):
        """Test that zero production doesn't cause division errors."""
        growth_func = SimpleTFPGrowth()
        current_tfp = np.array([1.0, 1.0, 1.0])
        production = np.array([100.0, 0.0, 50.0])  # Middle firm has zero production
        productivity_investment = np.array([10.0, 5.0, 5.0])
        base_growth = 0.0025
        elasticity = 0.3

        tfp_growth = growth_func.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        # Should not have NaN or inf values
        assert not np.isnan(tfp_growth).any()
        assert not np.isinf(tfp_growth).any()
        # Zero production firm should only get base growth
        assert tfp_growth[1] == base_growth

    def test_update_tfp_static_method(self):
        """Test the static update_tfp method."""
        current_tfp = np.array([1.0, 1.1, 0.9])
        tfp_growth = np.array([0.01, 0.02, -0.005])

        new_tfp = SimpleTFPGrowth.update_tfp(current_tfp, tfp_growth)

        expected = current_tfp * (1 + tfp_growth)
        assert np.allclose(new_tfp, expected)


class TestStochasticTFPGrowth:
    """Test the StochasticTFPGrowth implementation."""

    def test_stochastic_shocks(self):
        """Test that stochastic shocks are applied."""
        shock_std = 0.01
        growth_func = StochasticTFPGrowth(shock_std=shock_std)
        n_firms = 100  # Large number for statistical testing
        current_tfp = np.ones(n_firms)
        production = np.full(n_firms, 100.0)
        productivity_investment = np.zeros(n_firms)
        base_growth = 0.0025
        elasticity = 0.3

        # Set random seed for reproducibility
        np.random.seed(42)

        tfp_growth = growth_func.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        # Check that growth varies around base growth
        assert tfp_growth.mean() != base_growth  # Should have variation
        assert np.abs(tfp_growth.mean() - base_growth) < 0.005  # But mean should be close
        assert tfp_growth.std() > 0  # Should have variation

    def test_zero_shock_std(self):
        """Test that zero shock_std gives same result as SimpleTFPGrowth."""
        # Create StochasticTFPGrowth with shock_std=0.0 in constructor
        stochastic = StochasticTFPGrowth(shock_std=0.0)
        simple = SimpleTFPGrowth()

        n_firms = 5
        current_tfp = np.ones(n_firms)
        production = np.full(n_firms, 100.0)
        productivity_investment = np.full(n_firms, 10.0)
        base_growth = 0.0025
        elasticity = 0.3

        stochastic_growth = stochastic.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        simple_growth = simple.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        assert np.allclose(stochastic_growth, simple_growth)


class TestSectoralTFPGrowth:
    """Test the SectoralTFPGrowth implementation."""

    def test_sector_specific_base_growth(self):
        """Test different base growth rates by sector."""
        sector_base_growth = {
            0: 0.001,  # Low growth sector
            1: 0.003,  # High growth sector
            # Sector 2 uses default
        }
        growth_func = SectoralTFPGrowth(sector_base_growth=sector_base_growth)
        n_firms = 6
        current_tfp = np.ones(n_firms)
        production = np.full(n_firms, 100.0)
        productivity_investment = np.zeros(n_firms)
        base_growth = 0.002  # Default
        elasticity = 0.3

        # Assign firms to sectors
        sector_ids = np.array([0, 0, 1, 1, 2, 2])

        tfp_growth = growth_func.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
            sector_ids=sector_ids,
        )

        assert np.allclose(tfp_growth[0:2], 0.001)  # Sector 0
        assert np.allclose(tfp_growth[2:4], 0.003)  # Sector 1
        assert np.allclose(tfp_growth[4:6], base_growth)  # Sector 2 (default)

    def test_sector_specific_effectiveness(self):
        """Test different investment effectiveness by sector."""
        sector_effectiveness = {
            0: 0.2,  # High effectiveness
            1: 0.05,  # Low effectiveness
        }
        growth_func = SectoralTFPGrowth(sector_effectiveness=sector_effectiveness)
        n_firms = 4
        current_tfp = np.ones(n_firms)
        production = np.full(n_firms, 100.0)
        productivity_investment = np.full(n_firms, 10.0)
        base_growth = 0.0025
        elasticity = 0.5

        sector_ids = np.array([0, 0, 1, 1])

        tfp_growth = growth_func.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
            sector_ids=sector_ids,
        )

        # Calculate expected values
        investment_intensity = 0.1  # 10/100
        high_effect = base_growth + 0.2 * np.power(investment_intensity, elasticity)
        low_effect = base_growth + 0.05 * np.power(investment_intensity, elasticity)

        assert np.allclose(tfp_growth[0:2], high_effect)
        assert np.allclose(tfp_growth[2:4], low_effect)

    def test_no_sector_ids_fallback(self):
        """Test that without sector_ids, behaves like SimpleTFPGrowth."""
        sectoral = SectoralTFPGrowth()
        simple = SimpleTFPGrowth()

        n_firms = 5
        current_tfp = np.ones(n_firms)
        production = np.full(n_firms, 100.0)
        productivity_investment = np.full(n_firms, 10.0)
        base_growth = 0.0025
        elasticity = 0.3

        sectoral_growth = sectoral.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
            # No sector_ids provided
        )

        simple_growth = simple.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        assert np.allclose(sectoral_growth, simple_growth)


class TestProductivityGrowthEdgeCases:
    """Test edge cases and error conditions."""

    def test_negative_investment(self):
        """Test that negative investment doesn't break calculations."""
        growth_func = SimpleTFPGrowth()
        current_tfp = np.array([1.0])
        production = np.array([100.0])
        productivity_investment = np.array([-10.0])  # Negative investment
        base_growth = 0.0025
        elasticity = 0.3

        # Should handle gracefully (negative investment treated as 0)
        tfp_growth = growth_func.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        assert not np.isnan(tfp_growth).any()
        assert not np.isinf(tfp_growth).any()
        # Negative investment means no investment contribution, just base growth
        assert np.allclose(tfp_growth, base_growth)

    def test_very_high_investment_intensity(self):
        """Test behavior with very high investment relative to production."""
        effectiveness = 0.1
        growth_func = SimpleTFPGrowth(investment_effectiveness=effectiveness)
        current_tfp = np.array([1.0])
        production = np.array([1.0])  # Very small production
        productivity_investment = np.array([100.0])  # Very high investment
        base_growth = 0.0025
        elasticity = 0.3

        tfp_growth = growth_func.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        # Should be bounded and reasonable
        assert not np.isnan(tfp_growth).any()
        assert not np.isinf(tfp_growth).any()
        assert tfp_growth[0] > base_growth  # Should show growth from investment

    def test_empty_arrays(self):
        """Test with empty arrays."""
        growth_func = SimpleTFPGrowth()
        current_tfp = np.array([])
        production = np.array([])
        productivity_investment = np.array([])
        base_growth = 0.0025
        elasticity = 0.3

        tfp_growth = growth_func.compute_tfp_growth(
            current_tfp=current_tfp,
            production=production,
            productivity_investment=productivity_investment,
            base_growth_rate=base_growth,
            investment_elasticity=elasticity,
        )

        assert len(tfp_growth) == 0
        assert isinstance(tfp_growth, np.ndarray)

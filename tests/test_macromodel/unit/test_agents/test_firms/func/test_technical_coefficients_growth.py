import numpy as np
import pytest

from macromodel.agents.firms.func.technical_coefficients_growth import (
    NoOpTechnicalGrowth,
    SimpleTechnicalGrowth,
    TechnicalCoefficientsGrowth,
)


class TestNoOpTechnicalGrowth:
    """Test the no-operation technical growth implementation."""

    def test_initialization(self):
        """Test that NoOpTechnicalGrowth initializes correctly."""
        growth_func = NoOpTechnicalGrowth()
        assert growth_func.investment_effectiveness == 0.0
        assert growth_func.diminishing_returns_factor == 0.0

    def test_zero_growth_intermediate(self):
        """Test that NoOp returns zero growth for intermediate inputs."""
        growth_func = NoOpTechnicalGrowth()

        n_firms = 5
        n_industries = 3

        # Create test data
        current_multipliers = np.ones((n_firms, n_industries))
        cumulative_improvements = np.zeros((n_firms, n_industries))
        base_coefficients = np.random.rand(n_industries, n_industries)
        firm_industries = np.array([0, 1, 2, 0, 1])
        technical_investment = np.random.rand(n_firms, n_industries) * 100
        production = np.random.rand(n_firms) * 1000
        prices = np.random.rand(n_industries) * 10

        growth_rates = growth_func.compute_intermediate_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_improvements,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        # Should return all zeros
        assert np.allclose(growth_rates, 0.0)
        assert growth_rates.shape == (n_firms, n_industries)

    def test_zero_growth_capital(self):
        """Test that NoOp returns zero growth for capital inputs."""
        growth_func = NoOpTechnicalGrowth()

        n_firms = 5
        n_industries = 3

        # Create test data
        current_multipliers = np.ones((n_firms, n_industries))
        cumulative_improvements = np.zeros((n_firms, n_industries))
        base_coefficients = np.random.rand(n_industries, n_industries)
        firm_industries = np.array([0, 1, 2, 0, 1])
        technical_investment = np.random.rand(n_firms, n_industries) * 100
        production = np.random.rand(n_firms) * 1000
        prices = np.random.rand(n_industries) * 10

        growth_rates = growth_func.compute_capital_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_improvements,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        # Should return all zeros
        assert np.allclose(growth_rates, 0.0)
        assert growth_rates.shape == (n_firms, n_industries)

    def test_update_multipliers_static(self):
        """Test that update with zero growth maintains multipliers."""
        n_firms = 3
        n_industries = 4
        current_multipliers = np.random.rand(n_firms, n_industries) + 0.5
        growth_rates = np.zeros((n_firms, n_industries))

        new_multipliers = TechnicalCoefficientsGrowth.update_multipliers(current_multipliers, growth_rates)

        assert np.allclose(new_multipliers, current_multipliers)


class TestSimpleTechnicalGrowth:
    """Test the simple technical growth implementation."""

    def test_initialization(self):
        """Test initialization with custom parameters."""
        effectiveness = 0.2
        diminishing = 0.05
        growth_func = SimpleTechnicalGrowth(
            investment_effectiveness=effectiveness, diminishing_returns_factor=diminishing
        )
        assert growth_func.investment_effectiveness == effectiveness
        assert growth_func.diminishing_returns_factor == diminishing

    def test_intermediate_growth_calculation(self):
        """Test intermediate input growth calculation."""
        growth_func = SimpleTechnicalGrowth(
            investment_effectiveness=0.1,
            diminishing_returns_factor=0.0,  # No diminishing returns for simplicity
        )

        n_firms = 2
        n_industries = 3

        # Setup test data
        current_multipliers = np.ones((n_firms, n_industries))
        cumulative_improvements = np.zeros((n_firms, n_industries))

        # Base coefficients: how much of input j is needed to produce 1 unit in industry i
        base_coefficients = np.array(
            [
                [0.1, 0.2, 0.15],  # Industry 0 needs these amounts of inputs 0,1,2
                [0.05, 0.3, 0.1],  # Industry 1 needs these amounts
                [0.2, 0.1, 0.25],  # Industry 2 needs these amounts
            ]
        )

        firm_industries = np.array([0, 1])  # Firm 0 is in industry 0, firm 1 is in industry 1
        production = np.array([100.0, 200.0])
        prices = np.array([10.0, 15.0, 20.0])

        # Investment in improving each input type
        technical_investment = np.array(
            [
                [10.0, 20.0, 0.0],  # Firm 0 invests in inputs 0 and 1
                [0.0, 15.0, 30.0],  # Firm 1 invests in inputs 1 and 2
            ]
        )

        growth_rates = growth_func.compute_intermediate_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_improvements,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        # Check that positive investment leads to positive growth
        assert growth_rates[0, 0] > 0  # Firm 0, input 0 (has investment)
        assert growth_rates[0, 1] > 0  # Firm 0, input 1 (has investment)
        assert growth_rates[0, 2] == 0  # Firm 0, input 2 (no investment)
        assert growth_rates[1, 0] == 0  # Firm 1, input 0 (no investment)
        assert growth_rates[1, 1] > 0  # Firm 1, input 1 (has investment)
        assert growth_rates[1, 2] > 0  # Firm 1, input 2 (has investment)

    def test_capital_growth_calculation(self):
        """Test capital input growth calculation."""
        growth_func = SimpleTechnicalGrowth(investment_effectiveness=0.1, diminishing_returns_factor=0.0)

        n_firms = 2
        n_industries = 3

        # Setup test data
        current_multipliers = np.ones((n_firms, n_industries))
        cumulative_improvements = np.zeros((n_firms, n_industries))
        base_coefficients = np.array([[0.1, 0.2, 0.15], [0.05, 0.3, 0.1], [0.2, 0.1, 0.25]])
        firm_industries = np.array([0, 1])
        production = np.array([100.0, 200.0])
        prices = np.array([10.0, 15.0, 20.0])
        technical_investment = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 30.0]])

        growth_rates = growth_func.compute_capital_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_improvements,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        # Check results
        assert growth_rates[0, 0] > 0  # Firm 0, input 0 (has investment)
        assert growth_rates[0, 1] == 0  # Firm 0, input 1 (no investment)
        assert growth_rates[1, 2] > 0  # Firm 1, input 2 (has investment)

    def test_diminishing_returns(self):
        """Test that diminishing returns reduce growth rates."""
        growth_func = SimpleTechnicalGrowth(investment_effectiveness=0.1, diminishing_returns_factor=0.5)

        n_firms = 1
        n_industries = 2

        current_multipliers = np.ones((n_firms, n_industries))
        base_coefficients = np.array([[0.2, 0.3], [0.1, 0.4]])
        firm_industries = np.array([0])
        technical_investment = np.array([[100.0, 100.0]])
        production = np.array([1000.0])
        prices = np.array([10.0, 20.0])

        # First with no cumulative improvement
        cumulative_zero = np.zeros((n_firms, n_industries))
        growth_no_diminishing = growth_func.compute_intermediate_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_zero,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        # Then with significant cumulative improvement
        cumulative_high = np.array([[1.0, 1.0]])  # Already improved by 100%
        growth_with_diminishing = growth_func.compute_intermediate_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_high,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        # Growth should be lower with cumulative improvement
        assert np.all(growth_with_diminishing < growth_no_diminishing)
        assert np.all(growth_with_diminishing > 0)  # But still positive

    def test_zero_production_handling(self):
        """Test that zero production doesn't cause division errors."""
        growth_func = SimpleTechnicalGrowth()

        n_firms = 3
        n_industries = 2

        current_multipliers = np.ones((n_firms, n_industries))
        cumulative_improvements = np.zeros((n_firms, n_industries))
        base_coefficients = np.array([[0.2, 0.3], [0.1, 0.4]])
        technical_investment = np.ones((n_firms, n_industries)) * 10
        production = np.array([100.0, 0.0, 50.0])  # Firm 1 has zero production
        firm_industries = np.array([0, 0, 1])
        prices = np.array([10.0, 15.0])

        growth_rates = growth_func.compute_intermediate_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_improvements,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        # Should not have NaN or inf values
        assert not np.isnan(growth_rates).any()
        assert not np.isinf(growth_rates).any()

        # Zero production firm should have zero growth
        assert np.allclose(growth_rates[1], 0.0)

    def test_update_multipliers(self):
        """Test multiplier update with growth rates."""
        n_firms = 2
        n_industries = 3

        current_multipliers = np.array([[1.0, 1.2, 0.8], [0.9, 1.1, 1.0]])

        growth_rates = np.array(
            [
                [0.1, 0.0, 0.05],  # 10%, 0%, 5% growth
                [0.0, 0.2, -0.1],  # 0%, 20%, -10% growth (improvement in efficiency)
            ]
        )

        new_multipliers = TechnicalCoefficientsGrowth.update_multipliers(current_multipliers, growth_rates)

        expected = current_multipliers * (1 + growth_rates)
        assert np.allclose(new_multipliers, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_negative_investment(self):
        """Test that negative investment is handled gracefully."""
        growth_func = SimpleTechnicalGrowth()

        n_firms = 1
        n_industries = 2

        current_multipliers = np.ones((n_firms, n_industries))
        cumulative_improvements = np.zeros((n_firms, n_industries))
        base_coefficients = np.array([[0.2, 0.3], [0.1, 0.4]])
        technical_investment = np.array([[-10.0, 20.0]])  # Negative investment
        production = np.array([100.0])
        firm_industries = np.array([0])
        prices = np.array([10.0, 15.0])

        growth_rates = growth_func.compute_intermediate_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_improvements,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        # Negative investment should result in zero growth
        assert growth_rates[0, 0] == 0
        assert growth_rates[0, 1] > 0  # Positive investment still works

    def test_empty_arrays(self):
        """Test with empty arrays."""
        growth_func = SimpleTechnicalGrowth()

        current_multipliers = np.array([]).reshape(0, 0)
        cumulative_improvements = np.array([]).reshape(0, 0)
        base_coefficients = np.array([]).reshape(0, 0)
        technical_investment = np.array([]).reshape(0, 0)
        production = np.array([])
        firm_industries = np.array([]).astype(int)
        prices = np.array([])

        growth_rates = growth_func.compute_intermediate_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_improvements,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        assert len(growth_rates) == 0
        assert isinstance(growth_rates, np.ndarray)

    def test_very_high_cumulative_improvement(self):
        """Test behavior with very high cumulative improvements."""
        growth_func = SimpleTechnicalGrowth(investment_effectiveness=0.1, diminishing_returns_factor=0.5)

        n_firms = 1
        n_industries = 1

        current_multipliers = np.ones((n_firms, n_industries))
        cumulative_improvements = np.array([[10.0]])  # 1000% improvement already
        base_coefficients = np.array([[0.3]])
        technical_investment = np.array([[100.0]])
        production = np.array([1000.0])
        firm_industries = np.array([0])
        prices = np.array([15.0])

        growth_rates = growth_func.compute_intermediate_multiplier_growth(
            current_multipliers=current_multipliers,
            cumulative_improvements=cumulative_improvements,
            base_coefficients=base_coefficients,
            firm_industries=firm_industries,
            technical_investment=technical_investment,
            production=production,
            prices=prices,
        )

        # Should still be positive but very small due to diminishing returns
        assert growth_rates[0, 0] > 0
        assert growth_rates[0, 0] < 0.01  # Very small growth

    def test_mismatched_firm_industries(self):
        """Test handling of firm industries that don't match coefficient dimensions."""
        growth_func = SimpleTechnicalGrowth()

        n_firms = 2
        n_industries = 3

        current_multipliers = np.ones((n_firms, n_industries))
        cumulative_improvements = np.zeros((n_firms, n_industries))
        base_coefficients = np.random.rand(n_industries, n_industries)
        technical_investment = np.ones((n_firms, n_industries))
        production = np.array([100.0, 200.0])
        prices = np.random.rand(n_industries)

        # Invalid industry indices
        firm_industries = np.array([0, 5])  # Index 5 is out of bounds for 3 industries

        with pytest.raises(IndexError):
            growth_func.compute_intermediate_multiplier_growth(
                current_multipliers=current_multipliers,
                cumulative_improvements=cumulative_improvements,
                base_coefficients=base_coefficients,
                firm_industries=firm_industries,
                technical_investment=technical_investment,
                production=production,
                prices=prices,
            )

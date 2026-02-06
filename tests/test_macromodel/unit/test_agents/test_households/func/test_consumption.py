import numpy as np
import pytest

from macromodel.agents.households.func.consumption import (
    CESHouseholdConsumption,
    DefaultHouseholdConsumption,
)
from macromodel.agents.households.utils.create_bundle_matrix import create_bundle_matrix
from macromodel.configurations.households_configuration import create_household_bundle


class TestDefaultHouseholdConsumption:
    def test_compute_target_consumption_basic(self):
        """Test basic consumption computation with default parameters."""
        consumption_obj = DefaultHouseholdConsumption(
            consumption_smoothing_fraction=0.5,
            consumption_smoothing_window=4,
            minimum_consumption_fraction=0.1,
        )

        n_households = 5
        n_industries = 4

        # Set up test data
        historic_consumption_sum = np.ones((3, n_households))  # 3 time periods
        saving_rates = np.full(n_households, 0.2)  # 20% saving rate
        income = np.full(n_households, 100.0)  # Income of 100
        household_benefits = np.full(n_households, 80.0)  # Benefits of 80
        consumption_weights = np.full(n_industries, 1.0 / n_industries)  # Equal weights
        consumption_weights_by_income = np.zeros((n_industries, n_households))
        tau_vat = 0.1  # 10% VAT

        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.02,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=historic_consumption_sum,
            saving_rates=saving_rates,
            income=income,
            household_benefits=household_benefits,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            exogenous_total_consumption=np.zeros(10),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=tau_vat,
        )

        # Check output shape
        assert result.shape == (n_households, n_industries)

        # Check all values are non-negative
        assert np.all(result >= 0)

        # Check consumption is proportional to after-tax income
        expected_consumption_per_household = (1 - 0.2) * 100.0 / (1 + tau_vat)  # 80 / 1.1 ≈ 72.73
        expected_consumption_per_industry = expected_consumption_per_household / n_industries

        assert np.allclose(result, expected_consumption_per_industry, rtol=0.1)

    def test_minimum_consumption_threshold(self):
        """Test that minimum consumption threshold is respected."""
        consumption_obj = DefaultHouseholdConsumption(
            consumption_smoothing_fraction=0.0,  # No smoothing
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.5,  # High minimum threshold
        )

        n_households = 3
        n_industries = 2

        # Set up data where benefits would provide higher consumption than income
        historic_consumption_sum = np.ones((2, n_households))
        saving_rates = np.full(n_households, 0.8)  # High saving rate
        income = np.full(n_households, 50.0)  # Low income
        household_benefits = np.full(n_households, 200.0)  # High benefits
        consumption_weights = np.full(n_industries, 0.5)
        consumption_weights_by_income = np.zeros((n_industries, n_households))
        tau_vat = 0.0

        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=historic_consumption_sum,
            saving_rates=saving_rates,
            income=income,
            household_benefits=household_benefits,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            exogenous_total_consumption=np.zeros(5),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=tau_vat,
        )

        # Expected: minimum_consumption_fraction * (1 - saving_rates) * household_benefits
        # = 0.5 * (1 - 0.8) * 200 = 0.5 * 0.2 * 200 = 20 per household
        expected_per_industry = 20.0 / n_industries

        assert np.allclose(result, expected_per_industry, rtol=0.01)


class TestCESHouseholdConsumption:
    def test_compute_target_consumption_no_substitution_data(self):
        """Test CES consumption falls back to default when substitution data is missing."""
        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.5,
            consumption_smoothing_window=4,
            minimum_consumption_fraction=0.1,
            elasticity_of_substitution=2.0,
        )

        default_consumption = DefaultHouseholdConsumption(
            consumption_smoothing_fraction=0.5,
            consumption_smoothing_window=4,
            minimum_consumption_fraction=0.1,
        )

        n_households = 3
        n_industries = 4

        # Set up identical test data
        test_args = {
            "expected_inflation": 0.02,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((3, n_households)),
            "saving_rates": np.full(n_households, 0.2),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.full(n_households, 80.0),
            "consumption_weights": np.full(n_industries, 0.25),
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(10),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.1,
        }

        # CES without substitution data (should fall back to default)
        ces_result = ces_consumption.compute_target_consumption(**test_args)

        # Default consumption
        default_result = default_consumption.compute_target_consumption(**test_args)

        # Should be identical
        assert np.allclose(ces_result, default_result)

    def test_ces_substitution_within_bundles(self):
        """Test CES substitution behavior within bundles."""
        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.0,  # No smoothing for cleaner test
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            elasticity_of_substitution=2.0,
        )

        n_households = 2
        n_industries = 4

        # Create two bundles: [0, 1] and [2, 3]
        bundles = [[0, 1], [2, 3]]
        bundles_grouped = create_household_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(bundles_grouped))

        # Initial setup
        consumption_weights = np.array([0.2, 0.3, 0.1, 0.4])  # Different initial weights
        initial_prices = np.array([1.0, 1.0, 1.0, 1.0])
        current_prices = np.array([2.0, 1.0, 1.0, 2.0])  # Industry 0 and 3 doubled in price
        initial_taxes = np.array([0.0, 0.0, 0.0, 0.0])
        current_taxes = np.array([0.0, 0.0, 0.0, 0.0])

        test_args = {
            "expected_inflation": 0.0,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((2, n_households)),
            "saving_rates": np.zeros(n_households),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.zeros(n_households),
            "consumption_weights": consumption_weights,
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(5),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.0,
            "prices": current_prices,
            "initial_prices": initial_prices,
            "taxes": current_taxes,
            "initial_taxes": initial_taxes,
            "bundle_matrix": bundle_matrix,
        }

        result = ces_consumption.compute_target_consumption(**test_args)

        # Check shape
        assert result.shape == (n_households, n_industries)

        # With elasticity = 2.0 and price doubling:
        # - Industry 0 price doubled: substitution factor = (2.0)^(-2) = 0.25
        # - Industry 1 price unchanged: substitution factor = 1.0
        # - Industry 2 price unchanged: substitution factor = 1.0
        # - Industry 3 price doubled: substitution factor = (2.0)^(-2) = 0.25

        # Within bundle [0,1]: original weights [0.2, 0.3], after substitution should favor industry 1
        # Within bundle [2,3]: original weights [0.1, 0.4], after substitution should favor industry 2

        # Check that expensive goods (0, 3) have lower consumption than cheaper alternatives (1, 2)
        avg_consumption = np.mean(result, axis=0)
        assert avg_consumption[1] > avg_consumption[0]  # Industry 1 > Industry 0 in bundle [0,1]
        # Industries 2 and 3 converge to similar consumption due to CES substitution balancing
        # industry 2's lower initial weight with industry 3's higher price
        assert np.allclose(avg_consumption[2], avg_consumption[3], rtol=1e-10)  # Should be nearly equal

    def test_ces_bundle_normalization(self):
        """Test that CES substitution preserves bundle totals."""
        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            elasticity_of_substitution=1.5,
        )

        n_households = 1
        n_industries = 3

        # Single bundle with all industries
        bundles = [[0, 1, 2]]
        bundles_grouped = create_household_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(bundles_grouped))

        consumption_weights = np.array([0.5, 0.3, 0.2])
        initial_prices = np.array([1.0, 1.0, 1.0])
        current_prices = np.array([1.5, 1.0, 0.8])  # Varied prices
        initial_taxes = np.zeros(n_industries)
        current_taxes = np.zeros(n_industries)

        # Test with default weights (no substitution)
        test_args_default = {
            "expected_inflation": 0.0,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((2, n_households)),
            "saving_rates": np.zeros(n_households),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.zeros(n_households),
            "consumption_weights": consumption_weights,
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(5),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.0,
        }

        # Test with CES substitution
        test_args_ces = test_args_default.copy()
        test_args_ces.update(
            {
                "prices": current_prices,
                "initial_prices": initial_prices,
                "taxes": current_taxes,
                "initial_taxes": initial_taxes,
                "bundle_matrix": bundle_matrix,
            }
        )

        result_default = ces_consumption.compute_target_consumption(**test_args_default)
        result_ces = ces_consumption.compute_target_consumption(**test_args_ces)

        # Total consumption should be preserved
        assert np.allclose(np.sum(result_default), np.sum(result_ces), rtol=1e-10)

        # But individual industry allocations should differ due to substitution
        assert not np.allclose(result_default, result_ces, rtol=0.1)

    def test_empty_bundle_handling(self):
        """Test CES consumption handles empty bundles gracefully."""
        n_industries = 5

        # Create bundles with some empty ones
        bundles = create_household_bundle(n_industries, [[0, 1], [], [3, 4]])  # Bundle 1 is empty
        bundle_matrix = create_bundle_matrix(np.array(bundles))

        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            elasticity_of_substitution=1.0,
        )

        n_households = 1
        consumption_weights = np.full(n_industries, 0.2)
        prices = np.full(n_industries, 1.0)
        taxes = np.zeros(n_industries)

        test_args = {
            "expected_inflation": 0.0,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((2, n_households)),
            "saving_rates": np.zeros(n_households),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.zeros(n_households),
            "consumption_weights": consumption_weights,
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(5),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.0,
            "prices": prices,
            "initial_prices": prices,
            "taxes": taxes,
            "initial_taxes": taxes,
            "bundle_matrix": bundle_matrix,
        }

        # Should not raise an error
        result = ces_consumption.compute_target_consumption(**test_args)
        assert result.shape == (n_households, n_industries)
        assert np.all(result >= 0)

    def test_zero_elasticity_edge_case(self):
        """Test CES consumption with zero elasticity (no substitution)."""
        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            elasticity_of_substitution=0.0,  # No substitution
        )

        n_households = 1
        n_industries = 3
        bundles = [[0, 1, 2]]
        bundles_grouped = create_household_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(bundles_grouped))

        consumption_weights = np.array([0.5, 0.3, 0.2])
        initial_prices = np.array([1.0, 1.0, 1.0])
        current_prices = np.array([10.0, 1.0, 0.1])  # Extreme price changes
        taxes = np.zeros(n_industries)

        test_args = {
            "expected_inflation": 0.0,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((2, n_households)),
            "saving_rates": np.zeros(n_households),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.zeros(n_households),
            "consumption_weights": consumption_weights,
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(5),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.0,
            "prices": current_prices,
            "initial_prices": initial_prices,
            "taxes": taxes,
            "initial_taxes": taxes,
            "bundle_matrix": bundle_matrix,
        }

        result = ces_consumption.compute_target_consumption(**test_args)

        # With zero elasticity, consumption shares should remain unchanged despite price changes
        expected_shares = consumption_weights / np.sum(consumption_weights)
        result_shares = result[0, :] / np.sum(result[0, :])

        assert np.allclose(result_shares, expected_shares, rtol=1e-10)

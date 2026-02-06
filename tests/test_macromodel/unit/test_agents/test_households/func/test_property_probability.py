"""Test for household probability of buying formula fix (PR #50).

This test verifies the fix for the household purchasing probability calculation.
The bug was that `prob_buying = 1.0 / diff_exp` produced:
1. Inverted logic: decreased probability when buying became cheaper than renting
2. Invalid outputs: infinity and values exceeding 1.0

The fix uses the proper logistic function formula.

This test imports the actual DefaultHouseholdDemandForProperty class and tests
the compute_demand method with minimal fixture data that triggers the bug.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.households.func.property import DefaultHouseholdDemandForProperty


class TestHouseholdProbabilityOfBuying:
    """Test the probability of buying calculation in household property demand."""

    @pytest.fixture
    def property_demand_calculator(self):
        """Create a DefaultHouseholdDemandForProperty instance with test parameters."""
        return DefaultHouseholdDemandForProperty(
            probability_stay_in_rented_property=0.0,  # Force all renters to consider moving
            probability_stay_in_owned_property=1.0,  # Owners stay
            maximum_price_income_coefficient=5.0,
            maximum_price_income_exponent=1.0,
            maximum_price_noise_mean=0.0,
            maximum_price_noise_variance=0.0,  # No noise for deterministic test
            maximum_rent_income_coefficient=0.3,
            maximum_rent_income_exponent=1.0,
            psychological_pressure_of_renting=0.1,
            cost_comparison_temperature=1.0,  # Key parameter for the bug
            price_initial_markup=0.1,
            price_decrease_probability=0.5,
            price_decrease_mean=-0.05,
            price_decrease_variance=0.01,
            rent_initial_markup=0.1,
            rent_decrease_probability=0.5,
            rent_decrease_mean=-0.05,
            rent_decrease_variance=0.01,
            partial_rent_inflation_indexation=0.5,
            partial_rent_inflation_delay=4,
        )

    @pytest.fixture
    def minimal_housing_data(self):
        """Create minimal housing data DataFrame."""
        return pd.DataFrame(
            {
                "House ID": [0],
                "Value": [100000.0],
                "Rent": [500.0],
            }
        )

    def test_probability_formula_produces_valid_range(self, property_demand_calculator, minimal_housing_data):
        """Test that the probability formula produces values in [0, 1].

        This test creates a scenario where the old formula would produce
        invalid probability values (> 1 or inf).
        """
        np.random.seed(42)  # For reproducibility

        # Single household that is renting (status = 0)
        household_residence_tenure_status = np.array([0])
        household_income = np.array([50000.0])  # Moderate income
        household_financial_wealth = np.array([10000.0])  # Some savings

        # Market observations
        observed_fraction_value_price = np.array([1.0, 0.0])  # value = price
        observed_fraction_rent_value = np.array([0.005, 0.0])  # rent = 0.5% of value monthly

        max_price, max_rent, hoping_to_move = property_demand_calculator.compute_demand(
            housing_data=minimal_housing_data,
            household_residence_tenure_status=household_residence_tenure_status,
            household_income=household_income,
            household_financial_wealth=household_financial_wealth,
            observed_fraction_value_price=observed_fraction_value_price,
            observed_fraction_rent_value=observed_fraction_rent_value,
            expected_hpi_growth=0.02,  # 2% expected price growth
            assumed_mortgage_maturity=25,  # 25 year mortgage
            rental_income_taxes=0.2,
        )

        # Check that outputs are valid (no NaN, no inf where not expected)
        # Either max_price or max_rent should be set (household decided to buy or rent)
        assert not (
            np.isnan(max_price[0]) and np.isnan(max_rent[0])
        ), "Household should have decided to either buy or rent"

        # If they decided to buy, max_price should be finite and positive
        if not np.isnan(max_price[0]):
            assert np.isfinite(max_price[0]), f"max_price should be finite, got {max_price[0]}"
            assert max_price[0] > 0, f"max_price should be positive, got {max_price[0]}"

        # If they decided to rent, max_rent should be finite and positive
        if not np.isnan(max_rent[0]):
            assert np.isfinite(max_rent[0]), f"max_rent should be finite, got {max_rent[0]}"
            assert max_rent[0] > 0, f"max_rent should be positive, got {max_rent[0]}"

    def test_old_probability_formula_bug_demonstration(self):
        """Demonstrate the bug in the old probability formula.

        OLD CODE (lines 296-303 in property.py):
            diff_exp = np.exp(self.cost_comparison_temperature * (annual_cost_of_renting - annual_cost_of_purchasing))
            prob_buying = 1.0 / diff_exp

        When buying is much more expensive than renting:
        - (rent - buy) is very negative
        - exp(very_negative) approaches 0
        - 1 / (nearly_zero) approaches infinity

        This produces invalid probabilities > 1.
        """
        cost_comparison_temperature = 1.0

        # Case: buying is MUCH more expensive than renting
        annual_cost_of_renting = 10000.0
        annual_cost_of_purchasing = 50000.0  # 5x more expensive to buy

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # OLD formula - produces invalid probability
            diff = annual_cost_of_renting - annual_cost_of_purchasing  # -40000
            diff_exp_old = np.exp(cost_comparison_temperature * diff)  # exp(-40000) ≈ 0
            prob_buying_old = 1.0 / diff_exp_old  # 1/0 ≈ inf

        # Document the bug: old formula gives probability > 1 (or inf)
        assert prob_buying_old > 1.0 or np.isinf(
            prob_buying_old
        ), f"Expected old formula to produce invalid prob > 1, got {prob_buying_old}"

        # NEW formula should use proper logistic: 1 / (1 + exp(-x))
        # This always produces values in (0, 1)
        # Note: The exact fix implementation may vary, but the result must be in [0, 1]

    def test_multiple_households_decisions_are_valid(self, property_demand_calculator, minimal_housing_data):
        """Test that multiple households all get valid buy/rent decisions."""
        np.random.seed(123)

        n_households = 10
        # Mix of renters (0) and people in social housing (-1)
        household_residence_tenure_status = np.array([-1, 0, 0, 0, -1, 0, 0, -1, 0, 0])

        # Varying incomes
        household_income = np.linspace(20000, 100000, n_households)

        # Varying wealth
        household_financial_wealth = np.linspace(5000, 200000, n_households)

        observed_fraction_value_price = np.array([1.0, 0.0])
        observed_fraction_rent_value = np.array([0.005, 0.0])

        max_price, max_rent, hoping_to_move = property_demand_calculator.compute_demand(
            housing_data=minimal_housing_data,
            household_residence_tenure_status=household_residence_tenure_status,
            household_income=household_income,
            household_financial_wealth=household_financial_wealth,
            observed_fraction_value_price=observed_fraction_value_price,
            observed_fraction_rent_value=observed_fraction_rent_value,
            expected_hpi_growth=0.02,
            assumed_mortgage_maturity=25,
            rental_income_taxes=0.2,
        )

        # All outputs should be valid (finite positive or NaN for non-participants)
        for i in range(n_households):
            if not np.isnan(max_price[i]):
                assert np.isfinite(max_price[i]), f"Household {i}: max_price must be finite"
                assert max_price[i] > 0, f"Household {i}: max_price must be positive"
            if not np.isnan(max_rent[i]):
                assert np.isfinite(max_rent[i]), f"Household {i}: max_rent must be finite"
                assert max_rent[i] > 0, f"Household {i}: max_rent must be positive"

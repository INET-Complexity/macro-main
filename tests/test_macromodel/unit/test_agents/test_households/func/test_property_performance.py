"""Performance test for household property market operations.

This module tests performance-critical operations in the household
property market, particularly the housing listing logic.
"""

import time

import numpy as np
import pandas as pd

from macromodel.agents.households.households import Households


class _TimeseriesList:
    def __init__(self):
        self.values = []

    def append(self, value):
        self.values.append(value)


class _HouseholdTimeseries:
    def __init__(self):
        self.max_price_willing_to_pay = _TimeseriesList()
        self.max_rent_willing_to_pay = _TimeseriesList()

    def current(self, key):
        if key == "expected_income":
            return np.full(5, 50_000.0)
        if key == "wealth_financial_assets":
            return np.full(5, 25_000.0)
        raise KeyError(key)


class _PropertyDemand:
    def compute_demand(self, **kwargs):
        max_price = np.full(5, np.nan)
        max_rent = np.full(5, np.nan)
        households_hoping_to_move = np.array([False, False, False, True, False])
        return max_price, max_rent, households_hoping_to_move

    def compute_initial_sale_price(self, property_values):
        return 1.1 * property_values

    def compute_updated_sale_price(self, sale_prices):
        return sale_prices

    def compute_offered_rent_for_new_properties(self, property_value, observed_fraction_rent_value):
        return observed_fraction_rent_value[0] * property_value + observed_fraction_rent_value[1]

    def compute_offered_rent_for_existing_properties(self, current_offered_rent):
        return current_offered_rent


class TestHouseholdPropertyPerformance:
    """Test performance of household property market operations."""

    def test_household_hoping_to_move_indexing(self):
        """Test that household ID indexing for housing listings is efficient.

        This is a unit test for the specific .isin() operation that was causing
        the performance bottleneck.
        """
        # Create sample data similar to what households.py processes
        n_households = 10000
        n_houses = 5000

        # Boolean array of households hoping to move
        households_hoping_to_move = np.random.random(n_households) < 0.1  # 10% moving

        # House owner IDs
        owner_ids = np.random.randint(0, n_households, size=n_houses)

        # Test the optimized approach (should be fast)
        start_time = time.time()
        household_ids_hoping_to_move = np.flatnonzero(households_hoping_to_move)
        ind_mhr_temp_sale = np.isin(owner_ids, household_ids_hoping_to_move)
        elapsed_time = time.time() - start_time

        # Should complete in < 0.1 seconds even with 10k households and 5k houses
        assert elapsed_time < 0.1, (
            f"NumPy isin took {elapsed_time:.4f}s, expected < 0.1s. "
            "Performance optimization may not be working correctly."
        )

        # Verify correctness - result should be boolean array
        assert isinstance(ind_mhr_temp_sale, np.ndarray)
        assert ind_mhr_temp_sale.dtype == bool
        assert len(ind_mhr_temp_sale) == n_houses

    def test_household_hoping_to_move_mask_lists_matching_owner_id(self):
        """Regression: list only homes that a moving owner can make available.

        A moving owner may own several properties. The previous listing logic
        put all of them on the sale market, including tenant-occupied homes.
        This test verifies that owner-occupied and vacant homes can be listed,
        while tenant-occupied homes are left off the owner-occupier market.
        """
        households = object.__new__(Households)
        households.functions = {"property": _PropertyDemand()}
        households.states = {"Tenure Status of the Main Residence": np.ones(5)}
        households.ts = _HouseholdTimeseries()

        housing_data = pd.DataFrame(
            {
                "House ID": [0, 1, 2, 3, 4],
                "Value": [100.0, 200.0, 300.0, 400.0, 500.0],
                "Rent": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Corresponding Inhabitant Household ID": [0.0, 1.0, 3.0, 4.0, -1.0],
                "Corresponding Owner Household ID": [0, 1, 3, 3, 3],
                "Is Owner-Occupied": [1, 1, 1, 0, 0],
                "Sale Price": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "Temporarily for Sale": [False, False, False, False, False],
                "Up for Rent": [False, False, False, False, False],
                "Newly on the Rental Market": [False, False, False, False, False],
            }
        )

        households.prepare_housing_market_clearing(
            housing_data=housing_data,
            observed_fraction_value_price=np.array([1.0, 0.0]),
            observed_fraction_rent_value=np.array([0.01, 0.0]),
            expected_hpi_growth=0.0,
            assumed_mortgage_maturity=120,
            rental_income_taxes=0.0,
        )

        assert housing_data["Temporarily for Sale"].tolist() == [False, False, True, False, True]
        assert np.isnan(housing_data.loc[0, "Sale Price"])
        assert np.isnan(housing_data.loc[1, "Sale Price"])
        assert housing_data.loc[2, "Sale Price"] == 330.0
        assert np.isnan(housing_data.loc[3, "Sale Price"])
        assert housing_data.loc[4, "Sale Price"] == 550.0

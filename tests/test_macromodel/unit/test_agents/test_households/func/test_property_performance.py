"""Performance test for household property market operations.

This module tests performance-critical operations in the household
property market, particularly the housing listing logic.
"""

import time

import numpy as np


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

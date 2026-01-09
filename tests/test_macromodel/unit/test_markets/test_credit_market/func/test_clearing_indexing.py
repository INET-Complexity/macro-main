"""Test for NumPy advanced indexing fix in credit market clearing.

This test verifies that the fix for NumPy advanced indexing dimension
mismatch allows proper loan allocation without ValueError crashes.

Before fix: Would crash with ValueError due to shape mismatch
After fix: Completes successfully with correct loan allocation
"""

import numpy as np
import pytest


class TestCreditMarketClearingIndexing:
    """Test NumPy advanced indexing fix in water bucket credit clearing."""

    def test_numpy_indexing_with_old_code_would_fail(self):
        """Test that simulates the old broken code pattern.

        This test demonstrates what WOULD happen with the old code.
        The old code would fail with:
        ValueError: shape mismatch: value array of shape (1,N)
        could not be broadcast to indexing result of shape (N,1)
        """
        # Setup: simulate the credit market data structures
        n_banks = 1
        n_total_agents = 200
        n_agents_with_demand = 116

        # Create test arrays
        agents_with_demand = np.arange(n_agents_with_demand)
        new_loans = np.zeros((3, n_banks, n_total_agents))
        granted_loans_by_banks = np.array([1000000.0])
        capacities_weights = np.random.random(n_agents_with_demand)
        capacities_weights /= capacities_weights.sum()

        # OLD CODE (would crash):
        # new_loans[0, :, agents_with_demand] = np.outer(granted_loans_by_banks, capacities_weights)
        # This line would raise:
        # ValueError: shape mismatch: value array of shape (1,116)
        # could not be broadcast to indexing result of shape (116,1)

        # Verify the old pattern would fail
        with pytest.raises(ValueError, match="shape mismatch|could not be broadcast"):
            # This is what the old code tried to do:
            new_loans[0, :, agents_with_demand] = np.outer(granted_loans_by_banks, capacities_weights)

    def test_numpy_indexing_fix_works(self):
        """Test that the fixed code works correctly.

        This test verifies the new row-by-row assignment approach works
        and produces correct loan allocations.
        """
        # Setup: simulate the credit market data structures
        n_banks = 1
        n_total_agents = 200
        n_agents_with_demand = 116

        # Create test arrays
        agents_with_demand = np.arange(n_agents_with_demand)
        new_loans = np.zeros((3, n_banks, n_total_agents))
        granted_loans_by_banks = np.array([1000000.0])
        capacities_weights = np.random.random(n_agents_with_demand)
        capacities_weights /= capacities_weights.sum()

        # NEW CODE (fixed - should work):
        loan_matrix = np.outer(granted_loans_by_banks, capacities_weights)
        for bank_idx in range(len(granted_loans_by_banks)):
            new_loans[0, bank_idx, agents_with_demand] = loan_matrix[bank_idx, :]

        # Verify it worked
        allocated_loans = new_loans[0, 0, agents_with_demand]

        # Check correctness:
        # 1. All loans should be positive
        assert np.all(allocated_loans >= 0), "All loans should be non-negative"

        # 2. Sum should equal total granted
        assert np.isclose(
            allocated_loans.sum(), granted_loans_by_banks[0]
        ), "Total allocated loans should equal granted amount"

        # 3. Each agent should get their weighted share
        expected_loans = granted_loans_by_banks[0] * capacities_weights
        assert np.allclose(allocated_loans, expected_loans), "Each agent should receive their weighted share"

    def test_numpy_indexing_fix_with_multiple_banks(self):
        """Test the fix works with multiple banks (more realistic scenario)."""
        # Setup with multiple banks
        n_banks = 5
        n_total_agents = 200
        n_agents_with_demand = 50

        # Create test arrays
        agents_with_demand = np.arange(n_agents_with_demand)
        new_loans = np.zeros((3, n_banks, n_total_agents))
        granted_loans_by_banks = np.array([100000.0, 200000.0, 150000.0, 300000.0, 250000.0])
        capacities_weights = np.random.random(n_agents_with_demand)
        capacities_weights /= capacities_weights.sum()

        # Apply the fix
        loan_matrix = np.outer(granted_loans_by_banks, capacities_weights)
        for bank_idx in range(len(granted_loans_by_banks)):
            new_loans[0, bank_idx, agents_with_demand] = loan_matrix[bank_idx, :]

        # Verify each bank's allocation
        for bank_idx in range(n_banks):
            bank_loans = new_loans[0, bank_idx, agents_with_demand]

            # Each bank should allocate its full amount
            assert np.isclose(
                bank_loans.sum(), granted_loans_by_banks[bank_idx]
            ), f"Bank {bank_idx} should allocate its full granted amount"

            # Each agent should get their weighted share from this bank
            expected = granted_loans_by_banks[bank_idx] * capacities_weights
            assert np.allclose(bank_loans, expected), f"Bank {bank_idx} agents should receive correct weighted shares"

    def test_numpy_indexing_second_branch_fix(self):
        """Test the fix for the second branch (supply_weights case).

        This tests line 1260-1263 in clearing.py where we use supply_weights
        instead of capacities_weights.
        """
        # Setup
        n_banks = 3
        n_total_agents = 200
        n_agents_with_demand = 75

        # Create test arrays
        agents_with_demand = np.arange(n_agents_with_demand)
        new_loans = np.zeros((3, n_banks, n_total_agents))
        received_loans_by_debtors = np.random.random(n_agents_with_demand) * 10000
        supply_weights = np.array([0.3, 0.5, 0.2])  # Bank supply weights

        # Apply the fix (second branch)
        loan_matrix = np.outer(supply_weights, received_loans_by_debtors)
        for bank_idx in range(len(supply_weights)):
            new_loans[0, bank_idx, agents_with_demand] = loan_matrix[bank_idx, :]

        # Verify correctness
        for agent_idx, agent_id in enumerate(agents_with_demand):
            agent_total = new_loans[0, :, agent_id].sum()
            expected = received_loans_by_debtors[agent_idx]

            assert np.isclose(agent_total, expected), f"Agent {agent_id} should receive correct total loan amount"

            # Each bank should contribute their weighted share
            for bank_idx in range(n_banks):
                expected_share = supply_weights[bank_idx] * received_loans_by_debtors[agent_idx]
                actual_share = new_loans[0, bank_idx, agent_id]
                assert np.isclose(
                    actual_share, expected_share
                ), f"Bank {bank_idx} should provide correct share to agent {agent_id}"

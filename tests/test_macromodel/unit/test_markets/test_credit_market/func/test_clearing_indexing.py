"""Test for NumPy advanced indexing fix in credit market clearing (PR #48).

This test verifies the fix for NumPy advanced indexing dimension mismatch
that caused ValueError crashes during loan allocation.

The bug was that `new_loans[0, :, agents_with_demand] = np.outer(...)` would fail with:
    ValueError: shape mismatch: value array of shape (1,N)
    could not be broadcast to indexing result of shape (N,1)

The fix assigns loans row-by-row to avoid the dimension reordering issue.

This test imports the actual WaterBucketCreditMarketClearer class and tests
the clear_loans method with fixture data.
"""

import numpy as np
import pytest

from macromodel.markets.credit_market.func.clearing import (
    WaterBucketCreditMarketClearer,
)
from macromodel.markets.credit_market.types_of_loans import LoanTypes


class TestCreditMarketClearingIndexingIntegration:
    """Integration test for NumPy indexing fix using actual codebase."""

    @pytest.fixture
    def credit_market_clearer(self):
        """Create a WaterBucketCreditMarketClearer instance with test parameters."""
        return WaterBucketCreditMarketClearer(
            allow_short_term_firm_loans=True,
            allow_household_loans=False,
            firms_max_number_of_banks_visiting=3,
            households_max_number_of_banks_visiting=3,
            consider_loan_type_fractions=False,
            credit_supply_temperature=1.0,
            interest_rates_selection_temperature=1.0,
            creditor_selection_is_deterministic=True,
            creditor_minimum_fill=False,
            debtor_minimum_fill=False,
        )

    def test_clear_loans_no_value_error(self, credit_market_clearer, test_banks, test_firms):
        """Test that clear_loans completes without ValueError.

        Before the fix, this would crash with:
        ValueError: shape mismatch: value array of shape (1,N)
        could not be broadcast to indexing result of shape (N,1)

        After the fix, it should complete successfully.
        """
        np.random.seed(42)

        n_firms = test_firms.ts.current("n_firms")
        n_banks = test_banks.ts.current("n_banks")

        # Set up credit demand for firms
        target_credit = np.full(n_firms, 10000.0)
        test_firms.ts.override_current("target_long_term_credit", target_credit)

        # Initialize tracking arrays
        new_credit_by_bank = np.zeros(n_banks)
        new_credit_by_firm = np.zeros(n_firms)
        new_credit_by_household = np.zeros(1)
        max_supply = np.full(n_banks, np.inf)

        # This should NOT raise ValueError with the fix
        result = credit_market_clearer.clear_loans(
            banks=test_banks,
            firms=test_firms,
            households=None,
            loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=max_supply,
        )

        # Verify result is valid
        assert result is not None, "Result should not be None"
        assert result.shape[0] == 3, "Result should have 3 components (principal, rate, payment)"
        assert result.shape[1] == n_banks, "Result should have correct number of banks"

        # Verify no NaN or unexpected values in loan amounts
        loan_amounts = result[0]
        assert np.all(np.isfinite(loan_amounts) | (loan_amounts == 0)), "Loan amounts should be finite or zero"
        assert np.all(loan_amounts >= 0), "Loan amounts should be non-negative"


class TestCreditMarketClearingIndexingLogic:
    """Test NumPy advanced indexing fix formula in isolation."""

    def test_old_code_pattern_fails(self):
        """Demonstrate the bug in the old code pattern.

        OLD CODE (line 1249 in clearing.py):
            new_loans[0, :, agents_with_demand] = np.outer(granted_loans_by_banks, capacities_weights)

        This fails because NumPy advanced indexing reorders dimensions.
        """
        n_banks = 1
        n_total_agents = 200
        n_agents_with_demand = 116

        agents_with_demand = np.arange(n_agents_with_demand)
        new_loans = np.zeros((3, n_banks, n_total_agents))
        granted_loans_by_banks = np.array([1000000.0])
        capacities_weights = np.random.random(n_agents_with_demand)
        capacities_weights /= capacities_weights.sum()

        # OLD CODE would crash with ValueError
        with pytest.raises(ValueError, match="shape mismatch|could not be broadcast"):
            new_loans[0, :, agents_with_demand] = np.outer(granted_loans_by_banks, capacities_weights)

    def test_new_code_pattern_works(self):
        """Test that the fixed row-by-row assignment works correctly.

        NEW CODE (lines 1249-1252 in clearing.py):
            loan_matrix = np.outer(granted_loans_by_banks, capacities_weights)
            for bank_idx in range(len(granted_loans_by_banks)):
                new_loans[0, bank_idx, agents_with_demand] = loan_matrix[bank_idx, :]
        """
        n_banks = 1
        n_total_agents = 200
        n_agents_with_demand = 116

        agents_with_demand = np.arange(n_agents_with_demand)
        new_loans = np.zeros((3, n_banks, n_total_agents))
        granted_loans_by_banks = np.array([1000000.0])
        capacities_weights = np.random.random(n_agents_with_demand)
        capacities_weights /= capacities_weights.sum()

        # NEW CODE - row-by-row assignment
        loan_matrix = np.outer(granted_loans_by_banks, capacities_weights)
        for bank_idx in range(len(granted_loans_by_banks)):
            new_loans[0, bank_idx, agents_with_demand] = loan_matrix[bank_idx, :]

        # Verify correctness
        allocated_loans = new_loans[0, 0, agents_with_demand]

        assert np.all(allocated_loans >= 0), "All loans should be non-negative"
        assert np.isclose(allocated_loans.sum(), granted_loans_by_banks[0]), (
            "Total allocated should equal granted amount"
        )

        expected_loans = granted_loans_by_banks[0] * capacities_weights
        assert np.allclose(allocated_loans, expected_loans), "Each agent should receive weighted share"

    def test_multiple_banks_scenario(self):
        """Test the fix works with multiple banks."""
        n_banks = 5
        n_total_agents = 200
        n_agents_with_demand = 50

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
            assert np.isclose(bank_loans.sum(), granted_loans_by_banks[bank_idx]), (
                f"Bank {bank_idx} should allocate its full granted amount"
            )

    def test_second_branch_supply_weights(self):
        """Test the fix for the second branch using supply_weights.

        NEW CODE (lines 1257-1260 in clearing.py):
            loan_matrix = np.outer(supply_weights, received_loans_by_debtors)
            for bank_idx in range(len(supply_weights)):
                new_loans[0, bank_idx, agents_with_demand] = loan_matrix[bank_idx, :]
        """
        n_banks = 3
        n_total_agents = 200
        n_agents_with_demand = 75

        agents_with_demand = np.arange(n_agents_with_demand)
        new_loans = np.zeros((3, n_banks, n_total_agents))
        received_loans_by_debtors = np.random.random(n_agents_with_demand) * 10000
        supply_weights = np.array([0.3, 0.5, 0.2])

        # Apply the fix (second branch)
        loan_matrix = np.outer(supply_weights, received_loans_by_debtors)
        for bank_idx in range(len(supply_weights)):
            new_loans[0, bank_idx, agents_with_demand] = loan_matrix[bank_idx, :]

        # Verify each agent receives correct total
        for agent_idx, agent_id in enumerate(agents_with_demand):
            agent_total = new_loans[0, :, agent_id].sum()
            expected = received_loans_by_debtors[agent_idx]
            assert np.isclose(agent_total, expected), f"Agent {agent_id} should receive correct total loan amount"

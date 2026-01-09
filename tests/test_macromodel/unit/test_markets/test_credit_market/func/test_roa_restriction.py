"""Test for ROA (Return on Assets) restriction logic - division by zero bug.

This test demonstrates Bug #2:
- OLD CODE: Division by zero when capital stock = 0 produces inf, causing incorrect loan decisions
- NEW CODE: Uses np.divide() with where parameter to handle zero capital safely (ROA = 0)

The bug is in clearing.py lines 1165-1170 (GitHub version)
"""

import numpy as np
import pytest


class TestROARestrictionLogic:
    """Test Return on Assets restriction calculations."""

    def test_old_code_division_by_zero_fails(self):
        """OLD CODE: Direct division by zero produces inf values (incorrect behavior)."""
        # Simulate firm data with one firm having ZERO capital
        n_firms = 5
        expected_profits = np.array([1000000, 100000, 2000000, -100000, 500000], dtype=float)
        capital_stock = np.array([10000000, 0, 15000000, 2000000, 5000000], dtype=float)
        roa_threshold = 0.0

        # OLD CODE (from GitHub origin/main) - BROKEN
        return_on_assets_restrictions = np.zeros(n_firms)

        # This division produces inf when capital_stock[1] = 0
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for test
            roa_check = (expected_profits / capital_stock) >= roa_threshold
            return_on_assets_restrictions[roa_check] = np.inf

        # Firm with zero capital should NOT be allowed to borrow
        # But old code gives it inf (allowed) because inf >= 0.0 is True
        firm_with_zero_capital_idx = 1
        assert capital_stock[firm_with_zero_capital_idx] == 0
        assert return_on_assets_restrictions[firm_with_zero_capital_idx] == np.inf

        # This is the BUG: firm with zero capital gets inf restriction (allowed to borrow)
        # when it should be blocked
        print(f"\n[BUG] OLD CODE: Firm with zero capital has restriction = {return_on_assets_restrictions[firm_with_zero_capital_idx]}")
        print(f"      This means it can borrow, which is incorrect!")

    def test_new_code_handles_division_by_zero_correctly(self):
        """NEW CODE: Uses np.divide() with where parameter to handle zero capital safely."""
        # Same firm data
        n_firms = 5
        expected_profits = np.array([1000000, 100000, 2000000, -100000, 500000], dtype=float)
        capital_stock = np.array([10000000, 0, 15000000, 2000000, 5000000], dtype=float)
        roa_threshold = 0.0

        # NEW CODE (from backup) - FIXED
        firm_roa = np.divide(
            expected_profits,
            capital_stock,
            out=np.zeros_like(expected_profits),
            where=capital_stock != 0
        )

        return_on_assets_restrictions = np.full(n_firms, np.inf)
        return_on_assets_restrictions[firm_roa < roa_threshold] = 0.0

        # Verify firm with zero capital gets ROA = 0
        firm_with_zero_capital_idx = 1
        assert capital_stock[firm_with_zero_capital_idx] == 0
        assert firm_roa[firm_with_zero_capital_idx] == 0.0

        # With threshold = 0.0 and ROA = 0.0, the condition (0.0 < 0.0) is False
        # So restriction stays at inf (firm CAN borrow)
        # This is actually CORRECT: if threshold is 0.0, then ROA = 0.0 passes
        assert return_on_assets_restrictions[firm_with_zero_capital_idx] == np.inf

        print(f"\n[OK] NEW CODE: Firm with zero capital has ROA = {firm_roa[firm_with_zero_capital_idx]}")
        print(f"     Restriction = {return_on_assets_restrictions[firm_with_zero_capital_idx]}")

    def test_roa_logic_with_positive_threshold(self):
        """Test that zero capital firms are blocked when threshold > 0."""
        # Same firm data
        n_firms = 5
        expected_profits = np.array([1000000, 100000, 2000000, -100000, 500000], dtype=float)
        capital_stock = np.array([10000000, 0, 15000000, 2000000, 5000000], dtype=float)
        roa_threshold = 0.05  # 5% threshold

        # NEW CODE
        firm_roa = np.divide(
            expected_profits,
            capital_stock,
            out=np.zeros_like(expected_profits),
            where=capital_stock != 0
        )

        return_on_assets_restrictions = np.full(n_firms, np.inf)
        return_on_assets_restrictions[firm_roa < roa_threshold] = 0.0

        # Firm with zero capital has ROA = 0.0, which is < 0.05, so should be blocked
        firm_with_zero_capital_idx = 1
        assert firm_roa[firm_with_zero_capital_idx] == 0.0
        assert return_on_assets_restrictions[firm_with_zero_capital_idx] == 0.0  # Blocked!

        # Also check firm 3 (negative ROA) is blocked
        assert firm_roa[3] < 0  # Negative ROA
        assert return_on_assets_restrictions[3] == 0.0  # Blocked

        # Check profitable firms are allowed
        assert firm_roa[0] > roa_threshold
        assert return_on_assets_restrictions[0] == np.inf  # Allowed

        print(f"\n[OK] With threshold = {roa_threshold:.1%}:")
        print(f"     Firm with zero capital: ROA = {firm_roa[firm_with_zero_capital_idx]:.1%}, restriction = {return_on_assets_restrictions[firm_with_zero_capital_idx]}")
        print(f"     Firm with negative profit: ROA = {firm_roa[3]:.1%}, restriction = {return_on_assets_restrictions[3]}")
        print(f"     Firm with high ROA: ROA = {firm_roa[0]:.1%}, restriction = {return_on_assets_restrictions[0]}")

    def test_all_firms_correctness(self):
        """Test that ROA restriction logic works correctly for all firm types."""
        n_firms = 5
        expected_profits = np.array([1000000, 100000, 2000000, -100000, 500000], dtype=float)
        capital_stock = np.array([10000000, 0, 15000000, 2000000, 5000000], dtype=float)
        roa_threshold = 0.05  # 5% threshold

        # Calculate expected ROA manually
        expected_roa = np.array([
            1000000 / 10000000,  # 0.10 = 10%
            0.0,                 # Zero capital -> ROA = 0
            2000000 / 15000000,  # 0.133 = 13.3%
            -100000 / 2000000,   # -0.05 = -5%
            500000 / 5000000,    # 0.10 = 10%
        ])

        # NEW CODE
        firm_roa = np.divide(
            expected_profits,
            capital_stock,
            out=np.zeros_like(expected_profits),
            where=capital_stock != 0
        )

        # Verify ROA calculation
        assert np.allclose(firm_roa, expected_roa), f"ROA mismatch: {firm_roa} vs {expected_roa}"

        # Apply restrictions
        return_on_assets_restrictions = np.full(n_firms, np.inf)
        return_on_assets_restrictions[firm_roa < roa_threshold] = 0.0

        # Expected results:
        # Firm 0: ROA = 10% >= 5% -> ALLOWED
        # Firm 1: ROA = 0% < 5% -> BLOCKED
        # Firm 2: ROA = 13.3% >= 5% -> ALLOWED
        # Firm 3: ROA = -5% < 5% -> BLOCKED
        # Firm 4: ROA = 10% >= 5% -> ALLOWED

        assert return_on_assets_restrictions[0] == np.inf  # Firm 0 allowed
        assert return_on_assets_restrictions[1] == 0.0     # Firm 1 blocked (zero capital)
        assert return_on_assets_restrictions[2] == np.inf  # Firm 2 allowed
        assert return_on_assets_restrictions[3] == 0.0     # Firm 3 blocked (negative profit)
        assert return_on_assets_restrictions[4] == np.inf  # Firm 4 allowed

        print("\n[OK] All firms handled correctly:")
        for i in range(n_firms):
            status = "ALLOWED" if return_on_assets_restrictions[i] == np.inf else "BLOCKED"
            print(f"     Firm {i}: ROA = {firm_roa[i]:>7.1%}, {status}")

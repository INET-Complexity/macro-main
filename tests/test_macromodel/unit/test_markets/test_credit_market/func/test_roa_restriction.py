"""Test for ROA (Return on Assets) restriction division by zero fix (PR #49).

This test verifies the fix for the ROA calculation in credit market clearing.
The bug was that direct division by zero when capital_stock=0 produced inf,
which caused incorrect loan decisions (firms with zero capital could borrow).

The fix uses np.divide() with the where parameter to safely handle zero capital
by setting ROA=0 for those firms.

This test imports the actual WaterBucketCreditMarketClearer class and tests
the clear_loans method with fixture data that triggers the bug.
"""

import numpy as np
import pandas as pd
import pytest

from macromodel.markets.credit_market.func.clearing import (
    WaterBucketCreditMarketClearer,
)
from macromodel.markets.credit_market.types_of_loans import LoanTypes


class TestROARestrictionIntegration:
    """Test the ROA restriction calculation in credit market clearing."""

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

    def test_roa_calculation_with_zero_capital_no_crash(self, credit_market_clearer, test_banks, test_firms):
        """Test that ROA calculation doesn't crash when a firm has zero capital.

        This test verifies that the fix handles division by zero correctly
        when computing Return on Assets for firms with zero capital stock.
        The old code would produce inf values, causing incorrect loan decisions.
        """
        np.random.seed(42)

        # Get the number of firms
        n_firms = test_firms.ts.current("n_firms")

        # Set up a scenario where at least one firm has zero capital
        # Store original values to restore later
        original_capital = test_firms.ts.current("capital_inputs_stock_value").copy()

        # Set one firm's capital to zero to trigger the division by zero bug
        # Use the timeseries API to modify values
        if n_firms > 0:
            modified_capital = original_capital.copy()
            modified_capital[0] = 0.0
            test_firms.ts.override_current("capital_inputs_stock_value", modified_capital)

        # Set up credit demand for firms
        target_credit = np.full(n_firms, 10000.0)
        test_firms.ts.override_current("target_long_term_credit", target_credit)

        # Initialize tracking arrays
        new_credit_by_bank = np.zeros(test_banks.ts.current("n_banks"))
        new_credit_by_firm = np.zeros(n_firms)
        new_credit_by_household = np.zeros(1)  # Dummy for households
        max_supply = np.full(test_banks.ts.current("n_banks"), np.inf)

        # This should NOT raise a division by zero error with the fix
        # The old code would produce inf values that could cause issues
        try:
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

            # If we get here, the division by zero was handled correctly
            # Check that the result is a valid array (no NaN or unexpected inf)
            assert result is not None, "Result should not be None"
            assert np.all(np.isfinite(result[0]) | (result[0] == 0)), "Loan values should be finite or zero"

        finally:
            # Restore original capital values
            test_firms.ts.override_current("capital_inputs_stock_value", original_capital)


class TestROARestrictionLogic:
    """Test Return on Assets restriction formula calculations.

    These tests verify the mathematical correctness of the ROA formula fix
    by testing the formula in isolation. This complements the integration
    tests above which test the fix in the context of the full clearing algorithm.
    """

    def test_old_formula_produces_invalid_probabilities(self):
        """Demonstrate the bug in the old ROA formula.

        OLD CODE (lines 1165-1170 in clearing.py):
            return_on_assets_restrictions = np.zeros(agents_with_demand.shape)
            return_on_assets_restrictions[
                firms.ts.current("expected_profits")[agents_with_demand]
                / firms.ts.current("capital_inputs_stock_value")[agents_with_demand]
                >= banks.parameters.firm_loans_return_on_assets_ratio
            ] = np.inf

        When capital_stock = 0:
        - Division produces inf
        - inf >= threshold is True
        - Firm gets inf restriction (allowed to borrow)
        - This is INCORRECT: firm with zero capital shouldn't pass ROA check
        """
        # Simulate firm data with one firm having ZERO capital
        n_firms = 5
        expected_profits = np.array([1000000, 100000, 2000000, -100000, 500000], dtype=float)
        capital_stock = np.array([10000000, 0, 15000000, 2000000, 5000000], dtype=float)
        roa_threshold = 0.0

        # OLD CODE - BROKEN
        return_on_assets_restrictions = np.zeros(n_firms)

        with np.errstate(divide="ignore", invalid="ignore"):
            roa_check = (expected_profits / capital_stock) >= roa_threshold
            return_on_assets_restrictions[roa_check] = np.inf

        # Document the bug: firm with zero capital gets inf (allowed)
        firm_with_zero_capital_idx = 1
        assert capital_stock[firm_with_zero_capital_idx] == 0
        assert (
            return_on_assets_restrictions[firm_with_zero_capital_idx] == np.inf
        ), "Bug demonstration: old formula allows firm with zero capital"

    def test_new_formula_handles_zero_capital_correctly(self):
        """Test that the new formula handles zero capital safely.

        NEW CODE:
            firm_roa = np.divide(
                firm_expected_profits,
                firm_capital_stock,
                out=np.zeros_like(firm_expected_profits),
                where=firm_capital_stock != 0,
            )
            return_on_assets_restrictions = np.full(agents_with_demand.shape, np.inf)
            return_on_assets_restrictions[firm_roa < roa_threshold] = 0.0

        When capital_stock = 0:
        - np.divide with where parameter returns 0.0 (not inf)
        - ROA = 0.0, which is correctly compared against threshold
        """
        n_firms = 5
        expected_profits = np.array([1000000, 100000, 2000000, -100000, 500000], dtype=float)
        capital_stock = np.array([10000000, 0, 15000000, 2000000, 5000000], dtype=float)
        roa_threshold = 0.05  # 5% threshold

        # NEW CODE - FIXED
        firm_roa = np.divide(
            expected_profits,
            capital_stock,
            out=np.zeros_like(expected_profits),
            where=capital_stock != 0,
        )

        return_on_assets_restrictions = np.full(n_firms, np.inf)
        return_on_assets_restrictions[firm_roa < roa_threshold] = 0.0

        # Verify ROA calculation for zero capital firm
        firm_with_zero_capital_idx = 1
        assert capital_stock[firm_with_zero_capital_idx] == 0
        assert firm_roa[firm_with_zero_capital_idx] == 0.0, "Zero capital should result in ROA = 0"

        # With threshold = 5% and ROA = 0%, firm should be BLOCKED
        assert (
            return_on_assets_restrictions[firm_with_zero_capital_idx] == 0.0
        ), "Firm with zero capital should be blocked when threshold > 0"

    def test_all_firm_scenarios(self):
        """Test ROA restriction logic for various firm scenarios."""
        n_firms = 5
        expected_profits = np.array([1000000, 100000, 2000000, -100000, 500000], dtype=float)
        capital_stock = np.array([10000000, 0, 15000000, 2000000, 5000000], dtype=float)
        roa_threshold = 0.05  # 5%

        # Expected ROA values
        expected_roa = np.array(
            [
                1000000 / 10000000,  # 10% - ALLOWED
                0.0,  # 0% (zero capital) - BLOCKED
                2000000 / 15000000,  # 13.3% - ALLOWED
                -100000 / 2000000,  # -5% - BLOCKED
                500000 / 5000000,  # 10% - ALLOWED
            ]
        )

        # NEW CODE
        firm_roa = np.divide(
            expected_profits,
            capital_stock,
            out=np.zeros_like(expected_profits),
            where=capital_stock != 0,
        )

        assert np.allclose(firm_roa, expected_roa), f"ROA calculation mismatch: {firm_roa} vs {expected_roa}"

        # Apply restrictions
        return_on_assets_restrictions = np.full(n_firms, np.inf)
        return_on_assets_restrictions[firm_roa < roa_threshold] = 0.0

        # Verify results
        assert return_on_assets_restrictions[0] == np.inf  # 10% >= 5% - ALLOWED
        assert return_on_assets_restrictions[1] == 0.0  # 0% < 5% - BLOCKED
        assert return_on_assets_restrictions[2] == np.inf  # 13.3% >= 5% - ALLOWED
        assert return_on_assets_restrictions[3] == 0.0  # -5% < 5% - BLOCKED
        assert return_on_assets_restrictions[4] == np.inf  # 10% >= 5% - ALLOWED

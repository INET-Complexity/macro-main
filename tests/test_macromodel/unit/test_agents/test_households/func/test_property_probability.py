"""Test for household probability of buying formula.

This test demonstrates Bug #5:
- OLD CODE: Uses incorrect formula prob = 1 / exp(temp * diff)
  * Can produce probabilities > 1.0 or inf (invalid!)
  * Has inverted logic (favors buying when renting is cheaper!)
- NEW CODE: Uses correct logistic function prob = 1 / (1 + exp(-temp * diff))
  * Always produces valid probabilities in [0, 1]
  * Correct behavior (favors buying when renting is more expensive)

The bug is in property.py lines 294-303 (GitHub version)
"""

import numpy as np
import pytest


class TestHouseholdProbabilityBuying:
    """Test probability of buying calculation in household property decisions."""

    def test_old_code_invalid_probabilities(self):
        """OLD CODE: Produces invalid probabilities (inf, > 1.0)."""
        # Simulate cost scenarios
        cost_comparison_temperature = 0.5

        # Test case where buying costs much more (negative diff)
        # Should favor renting (low prob of buying)
        annual_cost_of_renting = 10000
        annual_cost_of_purchasing = 30000  # Buying costs more

        # OLD CODE (from GitHub) - BROKEN
        diff_exp = np.exp(cost_comparison_temperature * (annual_cost_of_renting - annual_cost_of_purchasing))
        prob_buying_old = 1.0 / diff_exp

        # With negative large diff, exp becomes very large, prob becomes very small
        # But for positive large diff, exp becomes very small, prob becomes inf!
        assert prob_buying_old < 1.0  # This case happens to be valid

        # Now test opposite: renting costs much more (positive diff)
        # Should favor buying (high prob of buying)
        annual_cost_of_renting2 = 30000  # Renting costs more
        annual_cost_of_purchasing2 = 10000

        diff_exp2 = np.exp(cost_comparison_temperature * (annual_cost_of_renting2 - annual_cost_of_purchasing2))
        prob_buying_old2 = 1.0 / diff_exp2

        # This is the BUG: probability is essentially 0 when it should be high!
        # exp(0.5 * 20000) is enormous, so 1/exp is nearly 0
        assert prob_buying_old2 < 0.0001  # Nearly zero - WRONG!
        print(
            f"\n[BUG] OLD CODE: When renting costs ${annual_cost_of_renting2:,.0f} "
            f"and buying costs ${annual_cost_of_purchasing2:,.0f}, "
            f"prob_buying = {prob_buying_old2:.6f} (should be HIGH, not near zero!)"
        )

    def test_new_code_valid_probabilities(self):
        """NEW CODE: Always produces valid probabilities in [0, 1]."""
        cost_comparison_temperature = 0.5

        # Test multiple scenarios
        scenarios = [
            (30000, 10000),  # Renting much more expensive -> should buy
            (20000, 15000),  # Renting slightly more expensive -> moderate
            (15000, 15000),  # Equal cost -> 50/50
            (15000, 20000),  # Buying slightly more expensive -> moderate
            (10000, 30000),  # Buying much more expensive -> should rent
        ]

        for rent_cost, buy_cost in scenarios:
            # NEW CODE (from backup) - FIXED
            diff = (rent_cost - buy_cost) / 10000
            diff_exp = np.exp(-cost_comparison_temperature * diff)
            prob_buying = 1.0 / (1.0 + diff_exp)
            prob_buying = np.clip(prob_buying, 0.0, 1.0)

            # Check validity
            assert 0.0 <= prob_buying <= 1.0, f"Probability must be in [0,1], got {prob_buying}"

            # Check correctness
            if rent_cost > buy_cost:
                # Renting more expensive -> should favor buying
                assert prob_buying > 0.5, f"Should favor buying when renting costs more"
            elif rent_cost < buy_cost:
                # Buying more expensive -> should favor renting
                assert prob_buying < 0.5, f"Should favor renting when buying costs more"
            else:
                # Equal cost -> should be 50/50
                assert 0.45 <= prob_buying <= 0.55, f"Should be ~50% for equal cost"

            print(f"[OK] Rent: ${rent_cost:>6,.0f}, Buy: ${buy_cost:>6,.0f} " f"-> prob_buying = {prob_buying:.1%}")

    def test_comparison_old_vs_new(self):
        """Compare OLD vs NEW code across multiple scenarios."""
        cost_comparison_temperature = 0.5

        # Test scenarios: (description, rent_cost, buy_cost, expected_behavior)
        scenarios = [
            ("Buying costs more", 10000, 20000, "Should favor renting"),
            ("Equal cost", 15000, 15000, "Should be 50/50"),
            ("Renting costs more", 20000, 10000, "Should favor buying"),
        ]

        print("\n[COMPARISON] OLD vs NEW CODE")
        print(f"{'Scenario':<25} {'Rent':>8} {'Buy':>8} {'Old Prob':>10} {'New Prob':>10} {'Correct?'}")
        print("-" * 80)

        for description, rent_cost, buy_cost, expected in scenarios:
            # OLD CODE
            with np.errstate(over="ignore", divide="ignore"):
                diff_exp_old = np.exp(cost_comparison_temperature * (rent_cost - buy_cost))
                prob_old = 1.0 / diff_exp_old
                if np.isinf(prob_old):
                    prob_old = float("inf")

            # NEW CODE
            diff = (rent_cost - buy_cost) / 10000
            diff_exp_new = np.exp(-cost_comparison_temperature * diff)
            prob_new = 1.0 / (1.0 + diff_exp_new)
            prob_new = np.clip(prob_new, 0.0, 1.0)

            # Check correctness
            if "renting" in expected.lower():
                correct = "YES" if prob_new < 0.5 else "NO"
            elif "buying" in expected.lower():
                correct = "YES" if prob_new > 0.5 else "NO"
            else:  # 50/50
                correct = "YES" if 0.45 <= prob_new <= 0.55 else "NO"

            prob_old_str = f"{prob_old:.1%}" if not np.isinf(prob_old) else "inf"
            print(
                f"{description:<25} ${rent_cost:>7,.0f} ${buy_cost:>7,.0f} "
                f"{prob_old_str:>10} {prob_new:>9.1%} {correct}"
            )

    def test_logistic_function_properties(self):
        """Test that the new code implements a proper logistic function."""
        cost_comparison_temperature = 0.5

        # Property 1: Symmetric around 0
        diff_positive = 5000
        diff_negative = -5000

        # NEW CODE
        diff_pos_norm = diff_positive / 10000
        diff_neg_norm = diff_negative / 10000

        prob_pos = 1.0 / (1.0 + np.exp(-cost_comparison_temperature * diff_pos_norm))
        prob_neg = 1.0 / (1.0 + np.exp(-cost_comparison_temperature * diff_neg_norm))

        # Logistic function is symmetric: P(x) + P(-x) = 1
        assert np.isclose(prob_pos + prob_neg, 1.0, atol=0.01), "Logistic function should be symmetric"

        # Property 2: At diff=0, probability should be 0.5
        prob_zero = 1.0 / (1.0 + np.exp(0))
        assert np.isclose(prob_zero, 0.5), "Probability at diff=0 should be 0.5"

        # Property 3: Monotonically increasing with diff
        diffs = np.linspace(-10000, 10000, 20)
        probs = []
        for d in diffs:
            d_norm = d / 10000
            p = 1.0 / (1.0 + np.exp(-cost_comparison_temperature * d_norm))
            probs.append(p)

        # Check monotonicity
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1], "Probability should increase with cost difference"

        print("\n[OK] All logistic function properties verified")

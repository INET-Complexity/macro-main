import pytest


class TestPolicyRates:
    def test__firm_debt_ratios(self, readers):
        assert readers.policy_rates.get_policy_rates("IND").loc["1964", "Policy Rate"].values[0] == pytest.approx(
            0.045, abs=0.001
        )

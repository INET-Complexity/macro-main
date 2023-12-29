import pytest


class TestPolicyRates:
    def test__firm_debt_ratios(self, readers):
        assert readers.policy_rates.cb_policy_rate("IND", 1964) == pytest.approx(0.00380, abs=0.001)

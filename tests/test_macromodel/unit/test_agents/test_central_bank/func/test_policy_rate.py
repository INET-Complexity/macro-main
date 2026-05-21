import pytest

from macromodel.agents.central_bank.func.policy_rate import ConstantPolicyRate, SmoothTaylorRule


class TestPolicyRate:
    def test__compute_rate(self):
        assert (
            ConstantPolicyRate().compute_rate(
                prev_rate=0.01,
                central_bank_states={},
                growth=0.01,
                inflation=0.01,
            )
            == 0.01
        )

    def test__smooth_taylor_rule_converts_annual_rate_to_period_rate(self):
        rate = SmoothTaylorRule().compute_rate(
            prev_rate=0.01,
            inflation=0.03,
            growth=0.0,
            central_bank_states={
                "rho": 0.5,
                "r_star": 0.02,
                "targeted_inflation_rate": 0.02,
                "phi_pi": 1.5,
                "phi_q": 0.5,
            },
            cpi_yoy_inflation=0.03,
            output_gap=0.02,
            time_unit=3,
        )

        # Annual rate = 0.5 * 0.04 + 0.5 * (0.02 + 0.02 + 1.5 * 0.01 + 0.5 * 0.02) = 0.0525
        assert rate == pytest.approx(0.0525 / 4.0)

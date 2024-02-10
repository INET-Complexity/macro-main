from macromodel.central_bank.func.policy_rate import ConstantPolicyRate


class TestPolicyRate:
    def test__compute_rate(self):
        assert (
            ConstantPolicyRate().compute_rate(
                prev_rate=0.01,
            )
            == 0.01
        )

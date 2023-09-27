class TestCentralBank:
    def test__central_bank_states(self, test_central_bank):
        assert test_central_bank is not None

    def test__central_bank_ts(self, test_central_bank):
        for ts_key in ["policy_rate"]:
            assert ts_key in test_central_bank.ts.get_keys()

    def test__compute_rate(self, test_central_bank):
        assert test_central_bank.compute_rate() == 0.02

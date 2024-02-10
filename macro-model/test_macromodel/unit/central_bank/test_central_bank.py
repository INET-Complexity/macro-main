class TestCentralBank:
    def test__init(self, test_central_bank, datawrapper):
        synthetic_central_bank = datawrapper.synthetic_countries["FRA"].central_bank

        # check that set is empty
        assert not test_central_bank.states

        ts_keys = [
            "policy_rate",
        ]

        assert set(ts_keys).issubset(set(test_central_bank.ts.get_keys()))

        assert test_central_bank.compute_rate() == synthetic_central_bank.central_bank_data["Policy Rate"].iloc[0]

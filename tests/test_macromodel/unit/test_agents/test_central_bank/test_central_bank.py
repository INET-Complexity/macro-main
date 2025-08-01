class TestCentralBank:
    def test__init(self, test_central_bank, datawrapper):
        synthetic_central_bank = datawrapper.synthetic_countries["FRA"].central_bank

        ts_keys = [
            "policy_rate",
        ]

        assert set(ts_keys).issubset(set(test_central_bank.ts.get_keys()))

    # def test__compute_rate(self, test_central_bank):
    #     assert (
    #         test_central_bank.compute_rate(
    #             inflation=0.01,
    #             growth=0.01,
    #         )
    #         == 0.0295
    #     )

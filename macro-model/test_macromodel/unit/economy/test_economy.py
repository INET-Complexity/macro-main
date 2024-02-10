class TestEconomy:
    def test__economy_states(self, test_economy):
        assert test_economy is not None

    def test__economy_ts(self, test_economy):
        for ts_key in [
            "ppi",
            "cpi",
            "cfpi",
            "good_prices",
            "unemployment_rate",
            "participation_rate",
            "vacancy_rate",
            "firm_insolvency_rate",
            "bank_insolvency_rate",
            "household_insolvency_rate",
            "total_growth",
            "sectoral_sentiment",
        ]:
            assert ts_key in test_economy.ts.get_keys()

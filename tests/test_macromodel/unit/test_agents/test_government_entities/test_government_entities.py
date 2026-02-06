class TestGovernmentEntities:
    def test__create(self, test_government_entities):
        assert test_government_entities.country_name == "FRA"

    def test__government_entities_states(self, test_government_entities):
        assert test_government_entities is not None

    def test__government_entities_ts(self, test_government_entities):
        for ts_key in [
            "n_government_entities",
            "consumption_in_usd",
            "consumption_in_lcu",
            "desired_consumption_in_usd",
            "desired_consumption_in_lcu",
        ]:
            assert ts_key in test_government_entities.ts.get_keys()

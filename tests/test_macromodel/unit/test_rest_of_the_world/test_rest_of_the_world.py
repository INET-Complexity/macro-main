from macromodel.configurations import RestOfTheWorldConfiguration
from macromodel.rest_of_the_world import RestOfTheWorld


class TestRestOfTheWorld:
    def test__init(self, test_row):
        assert test_row is not None

    def test__rest_of_the_world_states(self, test_row):
        assert test_row is not None

    def test__rest_of_the_world_ts(self, test_row):
        for ts_key in [
            "exports_real",
            "desired_exports_real",
            "imports_in_usd",
            "imports_in_lcu",
            "desired_imports_in_usd",
            "desired_imports_in_lcu",
            "price_in_usd",
            "price_in_lcu",
        ]:
            assert ts_key in test_row.ts.get_keys()

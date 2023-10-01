import pytest


class TestWorldBankRatesReader:
    def test__rate_dict(self, readers):
        assert readers["exchange_rates"].exchange_rates_dict(2014)["GBR"] == pytest.approx(0.608, abs=0.1)

    def test__to_usd(self, readers):
        assert readers["exchange_rates"].to_usd("GBR", 2014) == pytest.approx(1 / 0.608)

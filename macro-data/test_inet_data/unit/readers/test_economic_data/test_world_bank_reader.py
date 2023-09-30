import pytest


class TestWorldBankReader:
    def test__unemployment_rates(self, readers):
        assert readers["world_bank"].get_unemployment_rate("GBR", 2014) == pytest.approx(6.11e-2, abs=0.01)

    def test__participation_rates(self, readers):
        assert readers["world_bank"].get_participation_rate("GBR", 2014) == pytest.approx(62.7e-2, abs=0.01)

    def test__get_tau_vat(self, readers):
        assert readers["world_bank"].get_tau_vat("GBR", 2014) == pytest.approx(13.2e-2, abs=0.01)

    def test__get_tau_exp(self, readers):
        assert readers["world_bank"].get_tau_exp("GBR", 2014) == pytest.approx(0, abs=0.01)

    def test__get_historic_gdp(self, readers):
        assert readers["world_bank"].get_historic_gdp("GBR", 2014) == pytest.approx(1.863e12, abs=1e10)

    def test_get_current_quarterly_gdp(self, readers):
        assert readers["world_bank"].get_current_monthly_gdp("GBR", 2014) == pytest.approx(155235583333.0, abs=1e10)

    def test__get_gini_coef(self, readers):
        assert readers["world_bank"].get_gini_coef("GBR", 2014) == pytest.approx(34.0e-2, abs=0.1)

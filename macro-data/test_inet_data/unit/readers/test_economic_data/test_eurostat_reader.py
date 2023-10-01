import pytest


class TestEuroStatReader:
    def test__nonfin_firm_debt_ratios(self, readers):
        assert readers["eurostat"].nonfin_firm_debt_ratios("GBR", 2005) == pytest.approx(0.615, abs=0.01)
        assert readers["eurostat"].nonfin_firm_debt_ratios("CHE", 2012) == pytest.approx(0.776, abs=0.1)

    def test__nonfin_firm_deposit_ratios(self, readers):
        assert readers["eurostat"].nonfin_firm_deposit_ratios("GBR", 2005) == pytest.approx(0.231, abs=0.01)
        assert readers["eurostat"].nonfin_firm_deposit_ratios("CHE", 2012) == pytest.approx(0.486, abs=0.01)

    def test__total_nonfin_firm_debt(self, readers):
        assert readers["eurostat"].get_total_nonfin_firm_debt("FRA", 2014) == pytest.approx(2240317000000.0, abs=1e5)

    def test__total_fin_firm_debt(self, readers):
        assert readers["eurostat"].get_total_fin_firm_debt("FRA", 2014) == pytest.approx(447871000000.0, abs=1e5)

    def test__total_nonfin_firm_deposits(self, readers):
        assert readers["eurostat"].get_total_nonfin_firm_deposits("FRA", 2014) == pytest.approx(474690000000.0, abs=1e5)

    def test__total_bank_equity(self, readers):
        assert readers["eurostat"].get_total_bank_equity("FRA", 2014) == pytest.approx(588338000000.0, abs=1e5)

    def test__cb_debt_ratios(self, readers):
        assert readers["eurostat"].cb_debt_ratios("EST", 2003) == pytest.approx(0.003, abs=0.01)
        assert readers["eurostat"].cb_debt_ratios("BGR", 2000) == pytest.approx(0.099, abs=0.01)

    def test__cb_equity_ratios(self, readers):
        assert readers["eurostat"].cb_equity_ratios("EST", 2005) == pytest.approx(0.001, abs=0.01)
        assert readers["eurostat"].cb_equity_ratios("BGR", 2000) == pytest.approx(0.010, abs=0.01)

    def test__general_gov_debt_ratios(self, readers):
        assert readers["eurostat"].general_gov_debt_ratios("HUN", 2005) == pytest.approx(0.605, abs=0.01)
        assert readers["eurostat"].general_gov_debt_ratios("LVA", 2012) == pytest.approx(0.424, abs=0.01)

    def test__central_gov_debt_ratios(self, readers):
        assert readers["eurostat"].central_gov_debt_ratios("HUN", 2005) == pytest.approx(0.592, abs=0.01)
        assert readers["eurostat"].central_gov_debt_ratios("LVA", 2018) == pytest.approx(0.376, abs=0.01)

    def test__shortterm_interest_rates(self, readers):
        assert readers["eurostat"].shortterm_interest_rates("HRV", 2014, 0) == pytest.approx(0.0047, abs=0.01)
        assert readers["eurostat"].shortterm_interest_rates("HRV", 2014, 1) == pytest.approx(0.0073, abs=0.01)
        assert readers["eurostat"].shortterm_interest_rates("HRV", 2014, 3) == pytest.approx(0.0096, abs=0.01)
        assert readers["eurostat"].shortterm_interest_rates("HRV", 2014, 6) == pytest.approx(0.0132, abs=0.01)
        assert readers["eurostat"].shortterm_interest_rates("HRV", 2014, 12) == pytest.approx(0.0180, abs=0.01)

        assert readers["eurostat"].shortterm_interest_rates("BGR", 2008, 0) == pytest.approx(0.0516, abs=0.01)
        assert readers["eurostat"].shortterm_interest_rates("LVA", 2008, 1) == pytest.approx(0.0632, abs=0.01)
        assert readers["eurostat"].shortterm_interest_rates("BGR", 2008, 3) == pytest.approx(0.0714, abs=0.01)
        assert readers["eurostat"].shortterm_interest_rates("LVA", 2008, 6) == pytest.approx(0.0891, abs=0.01)
        assert readers["eurostat"].shortterm_interest_rates("LVA", 2008, 12) == pytest.approx(0.1011, abs=0.01)

    def test__longterm_central_gov_bond_rates(self, readers):
        assert readers["eurostat"].longterm_central_gov_bond_rates("HUN", 2005) == pytest.approx(0.0706, abs=0.01)
        assert readers["eurostat"].longterm_central_gov_bond_rates("LVA", 2012) == pytest.approx(0.0457, abs=0.01)

    def test__dividend_payout_ratio(self, readers):
        assert readers["eurostat"].dividend_payout_ratio("AUT", 2016) == pytest.approx(0.714, abs=0.01)
        assert readers["eurostat"].dividend_payout_ratio("LUX", 2012) == pytest.approx(0.283, abs=0.01)

    def test__firm_risk_premium(self, readers):
        assert readers["eurostat"].firm_risk_premium("FRA", 2014) == pytest.approx(0.00570, abs=0.001)

    def test__tax_rate_on_capital_formation(self, readers):
        assert readers["eurostat"].taxrate_on_capital_formation("AUT", 2010) == pytest.approx(0.23176, abs=0.001)

    def test__quarterly_gdp(self, readers):
        assert readers["eurostat"].get_quarterly_gdp("FRA", 2014, 1) == pytest.approx(535467e6, abs=1e6)
        assert readers["eurostat"].get_quarterly_gdp("FRA", 2014, 2) == pytest.approx(535937e6, abs=1e6)
        assert readers["eurostat"].get_quarterly_gdp("FRA", 2014, 3) == pytest.approx(539324e6, abs=1e6)
        assert readers["eurostat"].get_quarterly_gdp("FRA", 2014, 4) == pytest.approx(541314e6, abs=1e6)

    def test__monthly_gdp(self, readers):
        assert readers["eurostat"].get_monthly_gdp("FRA", 2014, 4) == pytest.approx(535937e6, abs=1e6)
        assert readers["eurostat"].get_monthly_gdp("FRA", 2014, 8) == pytest.approx(539987e6, abs=1e6)
        assert readers["eurostat"].get_monthly_gdp("FRA", 2014, 12) == pytest.approx(544749e6, abs=1e6)

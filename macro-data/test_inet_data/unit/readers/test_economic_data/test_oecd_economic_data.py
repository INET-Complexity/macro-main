import numpy as np
import pytest


class TestOECDEconData:
    def test__tau_sif(self, readers):
        assert readers["oecd_econ"].read_tau_sif("FRA", 2014) == pytest.approx(41.86e-2, abs=1e-4)

    def test__tau_siw(self, readers):
        assert readers["oecd_econ"].read_tau_siw("FRA", 2014) == pytest.approx(14.05e-2, abs=1e-4)

    def test__tau_firm(self, readers):
        assert readers["oecd_econ"].read_tau_firm("FRA", 2014) == pytest.approx(37.996e-2, abs=1e-4)

    def test__tau_inc(self, readers):
        assert readers["oecd_econ"].read_tau_income("FRA", 2014) == pytest.approx(0.28619674, abs=1e-4)

    """
    def test__immediate_interest_rates(self, readers):
        # france does not have an immediate interest rate in OECD
        assert readers["oecd_econ"].read_immediate_interest_rates(
            "JPN", 2014
        ) == pytest.approx(0.07e-2, abs=1e-4)

    def test__shortterm_interest_rates(self, readers):
        assert readers["oecd_econ"].read_shortterm_interest_rates(
            "FRA", 2014
        ) == pytest.approx(0.2099333e-2, abs=1e-4)

    def test__longterm_interest_rates(self, readers):
        assert readers["oecd_econ"].read_longterm_interest_rates(
            "FRA", 2014
        ) == pytest.approx(1.666442e-2, abs=1e-4)
    """

    def test__tierone_reserves(self, readers):
        assert readers["oecd_econ"].read_tierone_reserves("BEL", 2004) == pytest.approx(14.349e-2, abs=1e-4)

    def test__number_of_banks(self, readers):
        assert readers["oecd_econ"].read_number_of_banks("BEL", 2004) == pytest.approx(59, abs=1)

    def test__number_of_bank_branches(self, readers):
        assert readers["oecd_econ"].read_number_of_bank_branches("BEL", 2004) == pytest.approx(9525, abs=1)

    def test__number_of_bank_employees(self, readers):
        assert readers["oecd_econ"].read_number_of_bank_employees("BEL", 2004) == pytest.approx(70483, abs=10)

    def test__bank_distributed_profit(self, readers):
        assert readers["oecd_econ"].read_bank_distributed_profit("BEL", 2004) == pytest.approx(3916.57e6, abs=1e5)

    def test__bank_retained_profit(self, readers):
        assert readers["oecd_econ"].read_bank_retained_profit("BEL", 2004) == pytest.approx(-464.09e6, abs=1e5)

    def test__bank_total_assets(self, readers):
        assert readers["oecd_econ"].read_bank_total_assets("BEL", 2004) == pytest.approx(929203.50e6, abs=1e5)

    def test__employees_by_industry(self, readers):
        assert np.all(readers["oecd_econ"].employees_by_industry(2014, "FRA"))

    def test__general_gov_debt(self, readers):
        assert readers["oecd_econ"].general_gov_debt("FRA", 2014) == pytest.approx(2039884e6)

    def test__firm_zetas(self, readers):
        zetas = readers["oecd_econ"].read_firm_size_zetas("FRA", 2014)
        assert zetas[0] == pytest.approx(3.04555, abs=1e-4)
        assert zetas[1] == pytest.approx(1.59509, abs=1e-4)
        assert zetas[2] == pytest.approx(2.16292, abs=1e-4)

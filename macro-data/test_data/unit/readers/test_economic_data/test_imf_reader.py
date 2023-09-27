import pytest


class TestIMFReader:
    # assumes a scale of 100
    def test__number_of_commercial_banks(self, readers):
        assert readers["imf_reader"].number_of_commercial_banks(2013, "AFG") == pytest.approx(0.00016, abs=1e-2)

    def test__number_of_commercial_depositors(self, readers):
        assert readers["imf_reader"].number_of_commercial_depositors(2013, "AFG") == pytest.approx(26.95139, abs=1e-2)

    def test__number_of_commercial_borrowers(self, readers):
        assert readers["imf_reader"].number_of_commercial_borrowers(2013, "AFG") == pytest.approx(0.64544, abs=1e-2)

    def test__total_commercial_deposits(self, readers):
        assert readers["imf_reader"].total_commercial_deposits(2013, "AFG") == pytest.approx(207825.56e6, abs=1e7)

    def test__total_commercial_loans(self, readers):
        assert readers["imf_reader"].total_commercial_loans(2013, "AFG") == pytest.approx(46962.25e6, abs=1e7)

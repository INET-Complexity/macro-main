import pytest


class TestONSReader:
    def test__firm_size_zetas(self, readers):
        assert readers.ons.get_firm_size_zetas()[0] == pytest.approx(1.217, abs=1e-3)
        assert readers.ons.get_firm_size_zetas()[6] == pytest.approx(0.889, abs=1e-3)

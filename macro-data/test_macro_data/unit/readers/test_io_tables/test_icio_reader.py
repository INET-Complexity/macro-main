from functools import reduce
from itertools import product

import numpy as np
import pytest


class TestICIOReader:
    def test__countries_agg(self, readers):
        assert set(readers.icio[2014].considered_countries) == {"FRA"}

    def test__industries(self, readers):
        assert len(readers.icio[2014].industries) == 18
        try:
            readers.icio[2014].iot.loc[readers.icio[2014].considered_countries[0]].loc[readers.icio[2014].industries]
        except KeyError:
            assert False, "Raised a keyerror"

    def test__industries_agg(self, readers):
        try:
            readers.icio[2014].iot.loc[readers.icio[2014].considered_countries[0]].loc[readers.icio[2014].industries]
        except KeyError:
            assert False, "Raised a keyerror"

    @pytest.mark.parametrize("country", ["FRA"])
    def test__output(self, readers, country):
        assert np.all(readers.icio[2014].get_total_output(country)) > 0

    @pytest.mark.parametrize(
        "country, symbol",
        product(["FRA"], ["Firm Fixed Capital Formation", "Household Consumption"]),
    )
    def test__column_allc(self, readers, country: str, symbol: str):
        assert len(readers.icio[2014].column_allc(country, symbol)) == len(readers.icio[2014].industries)

    def test__hh_consumption(self, readers):
        assert readers.icio[2014].get_hh_consumption_weights("FRA").sum() == pytest.approx(1)

    def test__govt_cons(self, readers):
        assert readers.icio[2014].govt_consumption_weights("FRA").sum() == pytest.approx(1)

    def test__import_export(self, readers):
        total_exports = reduce(
            lambda a, b: a + b,
            [readers.icio[2014].get_exports(country) for country in readers.icio[2014].considered_countries + ["ROW"]],
        )
        total_imports = reduce(
            lambda a, b: a + b,
            [readers.icio[2014].get_imports(country) for country in readers.icio[2014].considered_countries + ["ROW"]],
        )

        assert total_imports.sum() - total_exports.sum() == pytest.approx(0)

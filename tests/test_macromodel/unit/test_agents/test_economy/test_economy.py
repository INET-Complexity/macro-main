import numpy as np
import pytest


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
            "cpi_yoy_inflation",
            "potential_output",
            "output_gap",
        ]:
            assert ts_key in test_economy.ts.get_keys()

    def test__compute_cpi_yoy_inflation(self, test_economy):
        test_economy.ts.dicts["cpi_inflation"] = [[0.01], [0.02], [0.03], [0.04]]

        test_economy.compute_cpi_yoy_inflation(exogenous_cpi_inflation_before=np.array([]))

        expected = np.prod([1.01, 1.02, 1.03, 1.04]) - 1.0
        assert test_economy.ts.current("cpi_yoy_inflation")[0] == pytest.approx(expected)

    def test__compute_output_gap(self, test_economy):
        test_economy.ts.dicts["ppi"] = [[1.0], [1.1]]
        test_economy.ts.dicts["total_output"] = [[100.0], [132.0]]
        test_economy.ts.dicts["potential_output"] = [[100.0]]

        test_economy.compute_output_gap()

        expected_real_output = 132.0 / 1.1
        expected_potential_output = 0.4 * expected_real_output + 0.6 * 100.0
        expected_output_gap = np.log(expected_real_output) - np.log(expected_potential_output)

        assert test_economy.ts.current("real_gross_output")[0] == pytest.approx(expected_real_output)
        assert test_economy.ts.current("potential_output")[0] == pytest.approx(expected_potential_output)
        assert test_economy.ts.current("output_gap")[0] == pytest.approx(expected_output_gap)

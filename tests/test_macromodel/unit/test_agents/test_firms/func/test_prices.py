import numpy as np
import pandas as pd
import pytest

from macro_data.readers.exo_prices.exo_prices_reader import SectorExoPrices, SectorExoPricesReader
from macromodel.agents.firms.func.prices import DefaultPriceSetter, SectorExogenousPriceSetter

N_FIRMS = 4
# Two firms per industry: indices 0,1 → B05a; indices 2,3 → C19
INDUSTRIES = ["B05a", "B05a", "C19", "C19"]

DEFAULT_PARAMS = dict(
    price_setting_noise_std=0.0,
    price_setting_speed_gf=0.0,
    price_setting_speed_dp=0.0,
    price_setting_speed_cp=0.0,
)

# With all speeds=0 and noise=0, default produces prev_prices unchanged.
PRICE_KWARGS = dict(
    prev_prices=np.ones(N_FIRMS),
    current_estimated_ppi_inflation=0.0,
    excess_demand=np.zeros(N_FIRMS),
    inventories=np.zeros(N_FIRMS),
    production=np.ones(N_FIRMS),
    prev_average_good_prices=np.ones(2),
    prev_firm_prices=np.ones(N_FIRMS),
    prev_supply=np.ones(N_FIRMS),
    prev_demand=np.ones(N_FIRMS),
    current_firm_sectors=np.array([0, 0, 1, 1]),
    curr_unit_costs=np.ones(N_FIRMS),
    prev_unit_costs=np.ones(N_FIRMS),
    ppi_during=np.ones(N_FIRMS),
    current_time=4,  # year ≈ 2014.75
)


def _make_exo_prices(industries: list[str]) -> SectorExoPrices:
    """Price path that rises from 1.0 (2013) to 2.0 (2030) for each industry."""
    df = pd.DataFrame({ind: [1.0, 2.0] for ind in industries}, index=[2013, 2030])
    reader = SectorExoPricesReader(prices=df)
    exo = SectorExoPrices.from_reader(reader, initial_year=2014)
    # per-firm base prices (one value per firm, indexed by firm index)
    exo.initial_model_prices = np.ones(N_FIRMS)
    return exo


class TestSectorExoPricesReader:
    def test_read_from_csv(self, tmp_path):
        csv = "year,B05a,C19\n2013,100.0,80.0\n2014,110.0,90.0\n2030,130.0,110.0\n"
        p = tmp_path / "firm_prices.csv"
        p.write_text(csv)
        reader = SectorExoPricesReader.read_from_raw_data(p)
        assert reader.prices is not None
        assert list(reader.prices.columns) == ["B05a", "C19"]
        assert reader.prices.loc[2014, "B05a"] == pytest.approx(110.0)

    def test_missing_file_returns_none(self, tmp_path):
        reader = SectorExoPricesReader.read_from_raw_data(tmp_path / "missing.csv")
        assert reader.prices is None

    def test_from_reader_copies_dataframe(self):
        df = pd.DataFrame({"B05a": [1.0, 2.0]}, index=[2013, 2030])
        exo = SectorExoPrices.from_reader(SectorExoPricesReader(prices=df), initial_year=2015)
        assert exo.prices is df
        assert exo.initial_year == 2015
        assert exo.initial_model_prices is None


class TestSectorExogenousPriceSetter:
    def _make_setter(self) -> SectorExogenousPriceSetter:
        setter = SectorExogenousPriceSetter(**DEFAULT_PARAMS)
        setter.overriden_industries = INDUSTRIES
        return setter

    def test_no_exo_prices_matches_default(self):
        """Without exo data the setter is identical to DefaultPriceSetter."""
        default = DefaultPriceSetter(**DEFAULT_PARAMS)
        setter = self._make_setter()  # firm_exo_prices is None
        assert np.allclose(
            default.compute_price(**PRICE_KWARGS),
            setter.compute_price(**PRICE_KWARGS),
        )

    def test_overrides_only_target_industry(self):
        """B05a prices are overridden; C19 prices follow the default."""
        setter = self._make_setter()
        setter.firm_exo_prices = _make_exo_prices(["B05a"])

        default_prices = DefaultPriceSetter(**DEFAULT_PARAMS).compute_price(**PRICE_KWARGS)
        result = setter.compute_price(**PRICE_KWARGS)

        # B05a firms (0, 1) should differ from default (rising price path > 2014 base)
        assert result[0] != pytest.approx(default_prices[0])
        assert result[1] != pytest.approx(default_prices[1])
        # C19 firms (2, 3) should be unchanged
        assert result[2] == pytest.approx(default_prices[2])
        assert result[3] == pytest.approx(default_prices[3])

    def test_overrides_all_industries(self):
        """All four firms are overridden when both industries are in the price file."""
        setter = self._make_setter()
        setter.firm_exo_prices = _make_exo_prices(["B05a", "C19"])

        default_prices = DefaultPriceSetter(**DEFAULT_PARAMS).compute_price(**PRICE_KWARGS)
        result = setter.compute_price(**PRICE_KWARGS)

        # All firms should differ from default
        assert not np.allclose(result, default_prices)
        # All should be overridden to the same normalised ratio (same price path)
        assert pytest.approx(result[0]) == result[1] == result[2] == result[3]

    def test_unknown_industry_in_file_is_ignored(self):
        """Industry codes in the CSV that don't appear in the model are silently skipped."""
        setter = self._make_setter()
        setter.firm_exo_prices = _make_exo_prices(["UNKNOWN_SECTOR"])

        default_prices = DefaultPriceSetter(**DEFAULT_PARAMS).compute_price(**PRICE_KWARGS)
        result = setter.compute_price(**PRICE_KWARGS)

        assert np.allclose(result, default_prices)

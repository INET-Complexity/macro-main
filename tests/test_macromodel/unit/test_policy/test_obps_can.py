import numpy as np
import pandas as pd
import pytest

from macro_data.readers.policy_data.obps_can_reader import OBPSCANData
from macromodel.policy.output_based_price_system_can import OutputBasedPriceSystemCAN

# Two industries: C24a is in the regulated list, A01 is not.
INDUSTRIES = ["C24a", "A01"]
CARBON_PRICE = 50.0
REDUCTION_FACTOR = 0.8
TIGHTENING_RATE = 0.02
# reference intensity set directly on each fixture: 0.5 tCO2/unit
REFERENCE_INTENSITY = 0.5


def _make_data(
    carbon_price: float = CARBON_PRICE,
    reduction_factor: float = REDUCTION_FACTOR,
    tightening_rate: float = TIGHTENING_RATE,
) -> OBPSCANData:
    df_rates = pd.DataFrame({"Date": list(range(2014, 2052)), "CAN": [carbon_price] * 38})
    df_policy = pd.DataFrame(
        {
            "Industry": ["C24a"],
            "reduction_factor": [reduction_factor],
            "tightening_rate": [tightening_rate],
        }
    )
    return OBPSCANData(df_rates=df_rates, df_policy=df_policy)


def _make_obps(year: int = 2020) -> OutputBasedPriceSystemCAN:
    obps = OutputBasedPriceSystemCAN(country_name="CAN", industries=INDUSTRIES, obps_data=_make_data())
    obps.current_year = year
    obps.current_t = year - 2014
    # pre-set reference intensity so tests are independent of the accumulation logic
    obps.baseline_emission_intensity[0] = REFERENCE_INTENSITY
    return obps


def _call(obps, input_em, capital_em=None, production=None, record_reference=False):
    if production is None:
        production = np.array([100.0, 50.0])
    if capital_em is None:
        capital_em = np.zeros(len(INDUSTRIES))
    return obps.compute_obps(
        use_obps_reg=True,
        record_obps_reference=record_reference,
        production=production,
        input_em=np.array(input_em),
        capital_em=capital_em,
    )


class TestComputeOBPSDisabledAndEarlyYears:
    def test_disabled_returns_all_zeros(self):
        obps = _make_obps(year=2020)
        cost = obps.compute_obps(
            use_obps_reg=False,
            record_obps_reference=False,
            production=np.array([100.0, 50.0]),
            input_em=np.array([999.0, 999.0]),
            capital_em=np.zeros(2),
        )
        assert np.all(cost == 0.0)

    def test_before_2019_returns_all_zeros(self):
        obps = _make_obps(year=2017)
        cost = _call(obps, input_em=[99.0, 99.0])
        assert np.all(cost == 0.0)

    def test_2018_returns_all_zeros(self):
        obps = _make_obps(year=2018)
        cost = _call(obps, input_em=[99.0, 99.0])
        assert np.all(cost == 0.0)


class TestComputeOBPSCosts:
    # limit = production * reduction_factor * reference_intensity
    #       = 100 * 0.8 * 0.5 = 40.0

    def test_positive_cost_when_above_limit(self):
        # emissions = 60 > limit 40 → cost = (60-40)*50 = 1000
        obps = _make_obps(year=2020)
        cost = _call(obps, input_em=[60.0, 0.0])
        assert cost[0] == pytest.approx(1000.0)

    def test_negative_cost_rebate_when_below_limit(self):
        # emissions = 30 < limit 40 → cost = (30-40)*50 = -500
        obps = _make_obps(year=2020)
        cost = _call(obps, input_em=[30.0, 0.0])
        assert cost[0] == pytest.approx(-500.0)

    def test_zero_cost_at_exact_limit(self):
        # emissions == limit → cost = 0
        obps = _make_obps(year=2020)
        cost = _call(obps, input_em=[40.0, 0.0])
        assert cost[0] == pytest.approx(0.0)

    def test_unregulated_industry_always_zero(self):
        # A01 (index 1) is not in regulated_industries
        obps = _make_obps(year=2020)
        cost = _call(obps, input_em=[0.0, 999.0])
        assert cost[1] == pytest.approx(0.0)

    def test_zero_production_skipped(self):
        obps = _make_obps(year=2020)
        cost = _call(obps, input_em=[999.0, 0.0], production=np.array([0.0, 50.0]))
        assert cost[0] == pytest.approx(0.0)

    def test_input_and_capital_emissions_summed(self):
        # input_em=25, capital_em=20 → total=45, limit=40 → cost=(45-40)*50=250
        obps = _make_obps(year=2020)
        cost = _call(obps, input_em=[25.0, 0.0], capital_em=np.array([20.0, 0.0]))
        assert cost[0] == pytest.approx(250.0)

    def test_emission_limit_attribute_updated(self):
        obps = _make_obps(year=2020)
        _call(obps, input_em=[60.0, 0.0])
        assert obps.emission_limit[0] == pytest.approx(40.0)


class TestGetLimit:
    def test_pre_2023_limit(self):
        # limit = production * reduction_factor * reference_intensity = 100*0.8*0.5 = 40
        obps = _make_obps(year=2021)
        limit = obps.get_limit(0, production=100.0)
        assert limit == pytest.approx(40.0)

    def test_post_2022_tightening(self):
        # 2024: limit = 100 * (0.4 - 0.4*0.02*(2024-2022)) = 100 * (0.4 - 0.016) = 38.4
        obps = _make_obps(year=2024)
        limit = obps.get_limit(0, production=100.0)
        assert limit == pytest.approx(38.4)

    def test_unknown_industry_returns_zero(self):
        obps = _make_obps(year=2020)
        # index 1 is A01, which has no row in df_policy
        limit = obps.get_limit(1, production=100.0)
        assert limit == pytest.approx(0.0)

    def test_limit_floored_at_zero_when_tightening_exceeds_baseline(self):
        # 2022 + 50 years of tightening at 2%/yr → multiplier goes negative; floor clamps to 0
        obps = _make_obps(year=2072)
        limit = obps.get_limit(0, production=100.0)
        assert limit == pytest.approx(0.0)


class TestReferenceAccumulation:
    def test_reference_intensity_computed_from_2017_to_2018(self):
        obps = OutputBasedPriceSystemCAN(country_name="CAN", industries=INDUSTRIES, obps_data=_make_data())
        # Simulate 2017, 2018, 2019 accumulation (reference recorded in 2017-2018 only)
        for year, em, prod in [
            (2017, [30.0, 0.0], [100.0, 50.0]),
            (2018, [30.0, 0.0], [100.0, 50.0]),
            (2019, [30.0, 0.0], [100.0, 50.0]),
        ]:
            obps.current_year = year
            obps.current_t = year - 2014
            obps.compute_obps(
                use_obps_reg=True,
                record_obps_reference=True,
                production=np.array(prod),
                input_em=np.array(em),
                capital_em=np.zeros(2),
            )
        # intensity = total_em / total_prod = 60 / 200 = 0.3 (accumulated during 2017-2018)
        assert obps.baseline_emission_intensity[0] == pytest.approx(0.3)
        assert obps.baseline_emission_intensity[1] == pytest.approx(0.0)

    def test_reference_not_accumulated_when_flag_false(self):
        obps = OutputBasedPriceSystemCAN(country_name="CAN", industries=INDUSTRIES, obps_data=_make_data())
        obps.current_year = 2017
        obps.current_t = 3
        obps.compute_obps(
            use_obps_reg=True,
            record_obps_reference=False,
            production=np.array([100.0, 50.0]),
            input_em=np.array([99.0, 0.0]),
            capital_em=np.zeros(2),
        )
        assert np.all(obps.reference_emission == 0.0)


class TestSparseYearSchedule:
    """Test OBPS with sparse (non-contiguous) year schedules matching real Canada data."""

    def _make_sparse_data(self) -> OBPSCANData:
        """Create a realistic sparse carbon price schedule: 2014, 2019, 2030."""
        df_rates = pd.DataFrame(
            {
                "Date": [2014, 2019, 2030],
                "CAN": [15.0, 50.0, 170.0],
            }
        )
        df_policy = pd.DataFrame(
            {
                "Industry": ["C24a"],
                "reduction_factor": [0.8],
                "tightening_rate": [0.02],
            }
        )
        return OBPSCANData(df_rates=df_rates, df_policy=df_policy)

    def test_sparse_schedule_initializes_without_error(self):
        """Sparse schedule should not crash during initialization."""
        obps = OutputBasedPriceSystemCAN(country_name="CAN", industries=INDUSTRIES, obps_data=self._make_sparse_data())
        obps.baseline_emission_intensity[0] = 0.5
        assert obps.get_price() == pytest.approx(15.0)

    def test_sparse_schedule_forward_fills_intermediate_years(self):
        """Years between milestones should use the last milestone's price (forward-fill)."""
        obps = OutputBasedPriceSystemCAN(country_name="CAN", industries=INDUSTRIES, obps_data=self._make_sparse_data())
        obps.baseline_emission_intensity[0] = 0.5

        obps.current_year = 2014
        assert obps.get_price() == pytest.approx(15.0)

        obps.current_year = 2016
        assert obps.get_price() == pytest.approx(15.0)

        obps.current_year = 2019
        assert obps.get_price() == pytest.approx(50.0)

        obps.current_year = 2025
        assert obps.get_price() == pytest.approx(50.0)

        obps.current_year = 2030
        assert obps.get_price() == pytest.approx(170.0)

        obps.current_year = 2035
        assert obps.get_price() == pytest.approx(170.0)

    def test_sparse_schedule_obps_cost_with_milestone_prices(self):
        """OBPS cost should use correct price at each milestone year."""
        obps = OutputBasedPriceSystemCAN(country_name="CAN", industries=INDUSTRIES, obps_data=self._make_sparse_data())
        obps.baseline_emission_intensity[0] = 0.5

        # At 2019: price = 50, limit = 100*0.8*0.5 = 40, emissions = 60 → cost = 20 * 50 = 1000
        obps.current_year = 2019
        obps.current_t = 5
        cost = obps.compute_obps(
            use_obps_reg=True,
            record_obps_reference=False,
            production=np.array([100.0, 50.0]),
            input_em=np.array([60.0, 0.0]),
            capital_em=np.zeros(2),
        )
        assert cost[0] == pytest.approx(1000.0)

        # At 2030: price = 170, B = 0.4, limit = 100*(0.4 - 0.4*0.02*8) = 33.6, emissions = 60
        # → cost = 26.4 * 170 = 4488
        obps.current_year = 2030
        obps.current_t = 16
        cost = obps.compute_obps(
            use_obps_reg=True,
            record_obps_reference=False,
            production=np.array([100.0, 50.0]),
            input_em=np.array([60.0, 0.0]),
            capital_em=np.zeros(2),
        )
        assert cost[0] == pytest.approx(4488.0)

    def test_sparse_schedule_with_reset(self):
        """Reset should clear accumulated state on sparse schedule."""
        obps = OutputBasedPriceSystemCAN(country_name="CAN", industries=INDUSTRIES, obps_data=self._make_sparse_data())

        obps.current_year = 2030
        obps.baseline_emission_intensity[0] = 0.5
        obps.reference_emission[0] = 100.0

        obps.reset()

        assert obps.current_year == 2014
        assert obps.baseline_emission_intensity[0] == 0.0
        assert obps.reference_emission[0] == 0.0

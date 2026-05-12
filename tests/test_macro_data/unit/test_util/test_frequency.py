import pandas as pd
import pytest

from macro_data.util.frequency import annual_to_period, periods_per_year


class TestPeriodsPerYear:
    @pytest.mark.parametrize(
        ("time_unit", "expected"),
        [(1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)],
    )
    def test_valid_divisors(self, time_unit, expected):
        assert periods_per_year(time_unit) == expected

    @pytest.mark.parametrize("time_unit", [0, 5, 7, 10, 13])
    def test_invalid_time_unit(self, time_unit):
        with pytest.raises(ValueError):
            periods_per_year(time_unit)


class TestAnnualToPeriod:
    def test_scalar_conversion(self):
        assert annual_to_period(0.12, 3) == pytest.approx(0.03)

    def test_series_conversion(self):
        series = pd.Series([0.12, 0.24], index=["a", "b"])
        expected = pd.Series([0.03, 0.06], index=["a", "b"])
        pd.testing.assert_series_equal(annual_to_period(series, 3), expected)

    def test_dataframe_conversion_all_numeric_columns(self):
        frame = pd.DataFrame({"Policy Rate": [0.12, 0.24], "Other Rate": [0.06, 0.18]})
        expected = pd.DataFrame({"Policy Rate": [0.03, 0.06], "Other Rate": [0.015, 0.045]})
        pd.testing.assert_frame_equal(annual_to_period(frame, 3), expected)

    def test_dataframe_conversion_target_column(self):
        frame = pd.DataFrame(
            {
                "Policy Rate": [0.12, 0.24],
                "Other Rate": [0.06, 0.18],
            }
        )
        expected = pd.DataFrame(
            {
                "Policy Rate": [0.03, 0.06],
                "Other Rate": [0.06, 0.18],
            }
        )
        pd.testing.assert_frame_equal(annual_to_period(frame, 3, "Policy Rate"), expected)

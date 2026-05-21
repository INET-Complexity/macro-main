import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.exogenous_data import (
    ExogenousCountryData,
    convert_growth_rates_to_model_period,
    convert_levels_to_model_period,
)


def test_convert_growth_rates_to_model_period_quarterly_noop():
    data = pd.DataFrame({"GDP": [0.03, 0.06]}, index=pd.to_datetime(["2020-01-01", "2020-04-01"]))

    converted = convert_growth_rates_to_model_period(data, time_unit=3)

    pd.testing.assert_frame_equal(converted, data)


def test_convert_growth_rates_to_model_period_monthly_compounds_to_quarterly_rate():
    data = pd.DataFrame({"GDP": [0.331]}, index=pd.to_datetime(["2020-01-01"]))

    converted = convert_growth_rates_to_model_period(data, time_unit=1)

    expected_monthly_rate = (1.331 ** (1.0 / 3.0)) - 1.0
    assert list(converted.index) == list(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    np.testing.assert_allclose(converted["GDP"].to_numpy(), expected_monthly_rate)
    np.testing.assert_allclose((1.0 + converted["GDP"]).prod() - 1.0, 0.331)


def test_convert_growth_rates_to_model_period_annual_compounds_quarters():
    data = pd.DataFrame(
        {"GDP": [0.1, 0.1, 0.1, 0.1]},
        index=pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]),
    )

    converted = convert_growth_rates_to_model_period(data, time_unit=12)

    assert list(converted.index) == list(pd.to_datetime(["2020-01-01"]))
    np.testing.assert_allclose(converted["GDP"].iloc[0], (1.1**4) - 1.0)


def test_convert_levels_to_model_period_interpolates_monthly_levels():
    data = pd.DataFrame({"Unemployment Rate (Value)": [0.03, 0.06]}, index=pd.to_datetime(["2020-01-01", "2020-04-01"]))

    converted = convert_levels_to_model_period(data, time_unit=1)

    expected = np.array([0.03, 0.04, 0.05, 0.06, 0.06, 0.06])
    assert list(converted.index) == list(
        pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"])
    )
    np.testing.assert_allclose(converted["Unemployment Rate (Value)"].to_numpy(), expected)


class TestExogenous:
    def test__exogenous(self, readers, industry_data):
        country = Country("FRA")
        data = ExogenousCountryData.from_data_readers(
            country_name=country,
            readers=readers,
            year=2014,
            quarter=1,
            industry_vectors=industry_data[country]["industry_vectors"],
        )

        assert data.inflation.shape[0] > 0

        calibration_data = data.get_calibration_data(2014, 1)

        assert (calibration_data[("FRA", "HPI (Value)")].dropna() > 0).all()

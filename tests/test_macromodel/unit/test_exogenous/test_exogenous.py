import numpy as np
import pandas as pd

from macromodel.exogenous.exogenous_ts import create_exogenous_timeseries


def _minimal_exogenous_inputs():
    return {
        "inflation_during": pd.DataFrame({"CPI Inflation": [0.01], "PPI Inflation": [0.02]}),
        "national_accounts_during": pd.DataFrame(),
        "unemployment_rate_during": pd.DataFrame({"Unemployment Rate (Value)": [0.05]}),
        "vacancy_rate_during": pd.DataFrame({"Vacancy Rate (Value)": [0.01]}),
        "house_price_index_during": pd.DataFrame(
            {
                "Real House Price Index Growth": [0.0],
                "Nominal House Price Index Growth": [0.0],
            }
        ),
        "exchange_rates_data_during": pd.DataFrame(
            {"Exchange Rate": [1.2, 1.3]},
            index=pd.to_datetime(["2014-01-01", "2015-01-01"]),
        ),
    }


def _scalar_exchange_rates(exchange_rate_ts):
    return [float(np.asarray(value).squeeze()) for value in exchange_rate_ts]


def test_create_exogenous_timeseries_repeats_annual_exchange_rates_quarterly():
    ts = create_exogenous_timeseries(**_minimal_exogenous_inputs(), time_unit=3)

    assert _scalar_exchange_rates(ts.exchange_rate) == [1.2] * 4 + [1.3] * 4


def test_create_exogenous_timeseries_repeats_annual_exchange_rates_monthly():
    ts = create_exogenous_timeseries(**_minimal_exogenous_inputs(), time_unit=1)

    assert _scalar_exchange_rates(ts.exchange_rate) == [1.2] * 12 + [1.3] * 12

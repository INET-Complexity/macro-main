import pandas as pd

from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram


def create_individuals_timeseries(data: pd.DataFrame, scale: int) -> TimeSeries:
    return TimeSeries(
        n_individuals=len(data),
        #
        employee_income=data["Employee Income"].values,
        employee_income_histogram=get_histogram(data["Employee Income"].values, scale),
        income_from_unemployment_benefits=data["Income from Unemployment Benefits"].values,
        income=data["Income"].values,
        expected_income=data["Income"].values,
        income_histogram=get_histogram(data["Income"].values, scale),
        #
        labour_inputs=data["Labour Inputs"].values,
        reservation_wages=data["Employee Income"].values + data["Income from Unemployment Benefits"].values,
    )

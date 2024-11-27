import pandas as pd

from macromodel.timeseries import TimeSeries


def create_central_bank_timeseries(data: pd.DataFrame) -> TimeSeries:
    return TimeSeries(policy_rate=[data["policy_rate"].values[0]])

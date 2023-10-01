import pandas as pd

from inet_macromodel.timeseries import TimeSeries


def create_central_bank_timeseries(data: pd.DataFrame) -> TimeSeries:
    return TimeSeries(policy_rate=[data["Policy Rate"].values[0]])

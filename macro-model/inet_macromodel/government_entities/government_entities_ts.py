import pandas as pd

from inet_macromodel.timeseries import TimeSeries


def create_government_entities_timeseries(data: pd.DataFrame, n_government_entities: int) -> TimeSeries:
    return TimeSeries(
        n_government_entities=n_government_entities,
        consumption_in_usd=data["Consumption in USD"].values,
        consumption_in_lcu=data["Consumption in LCU"].values,
        total_consumption=[data["Consumption in LCU"].values.sum()],
        desired_consumption_in_usd=data["Consumption in USD"].values,
        desired_consumption_in_lcu=data["Consumption in LCU"].values,
    )

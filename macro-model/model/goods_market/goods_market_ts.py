import numpy as np

from model.timeseries import TimeSeries


def create_goods_market_timeseries(n_industries: int) -> TimeSeries:
    return TimeSeries(
        total_industry_supply=np.zeros(n_industries),
        total_industry_demand=np.zeros(n_industries),
    )

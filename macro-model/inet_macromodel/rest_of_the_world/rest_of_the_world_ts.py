import numpy as np
import pandas as pd

from inet_macromodel.timeseries import TimeSeries


def create_rest_of_the_world_timeseries(
    data: pd.DataFrame,
    initial_inflation: float,
    n_industries: int,
) -> TimeSeries:
    return TimeSeries(
        #
        inflation=[initial_inflation],
        #
        exports_real=data["Exports"].values,
        desired_exports_real=data["Exports"].values,
        #
        imports_in_usd=data["Imports in USD"].values,
        imports_in_lcu=data["Imports in LCU"].values,
        desired_imports_in_usd=data["Imports in USD"].values,
        desired_imports_in_lcu=data["Imports in LCU"].values,
        #
        price_in_usd=data["Price in USD"].values,
        price_in_lcu=data["Price in LCU"].values,
        price_offered=np.full(n_industries, 1.0),
    )

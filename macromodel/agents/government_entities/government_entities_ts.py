from typing import Optional

import numpy as np
import pandas as pd

from macromodel.timeseries import TimeSeries


def create_government_entities_timeseries(
    data: pd.DataFrame,
    n_government_entities: int,
    add_emissions: bool = False,
    emission_factors_lcu: Optional[np.ndarray] = None,
    emitting_indices: Optional[np.ndarray] = None,
) -> TimeSeries:
    if add_emissions:
        emissions = np.sum(data["Consumption in LCU"].values[emitting_indices] * emission_factors_lcu)
    else:
        emissions = None

    return TimeSeries(
        n_government_entities=n_government_entities,
        consumption_in_usd=data["Consumption in USD"].values,
        consumption_in_lcu=data["Consumption in LCU"].values,
        total_consumption=[data["Consumption in LCU"].values.sum()],
        desired_consumption_in_usd=data["Consumption in USD"].values,
        desired_consumption_in_lcu=data["Consumption in LCU"].values,
        emissions=emissions,
    )

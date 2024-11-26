from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_linear(
    household_data: pd.DataFrame,
    independents: list[str],
    dependent: str,
) -> tuple[np.ndarray, Any]:
    dependent_data = household_data[dependent].values
    x = household_data[independents].values
    if len(independents) == 0:
        return dependent_data.mean(), None
    # x = (x - x.min()) / (x.max() - x.min())
    x /= x.sum(axis=0)

    # Fit
    reg = LinearRegression().fit(x, dependent_data)
    pred = reg.predict(x)  # noqa

    return pred, reg

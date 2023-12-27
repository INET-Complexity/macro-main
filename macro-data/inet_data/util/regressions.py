from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


# TODO does the normalisation make sense?
def fit_linear(
    household_data: pd.DataFrame,
    independents: list[str],
    dependent: str,
    model: LinearRegression,
) -> np.ndarray:
    """
    Fits a linear regression model to the given data.

    First, the missing values are imputed using the mean of the column and mapped to 0;1.

    Then, the model is fitted.

    Args:
        household_data (pd.DataFrame): The household data.
        independents (list[str]): The list of independent variables.
        dependent (str): The dependent variable.
        model (LinearRegression): The linear regression model.

    Returns:
        np.ndarray: The predicted values.

    Raises:
        None
    """
    dependent_data = household_data[dependent].values
    x = household_data[independents].values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    x = imp_mean.fit_transform(x)
    if len(independents) == 0:
        return dependent_data.mean()

    # Fit
    # x = (x - x.min()) / (x.max() - x.min())
    x /= x.sum(axis=0)
    model.fit(x, dependent_data)
    pred = model.predict(x)  # noqa

    return pred

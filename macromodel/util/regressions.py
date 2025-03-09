"""Linear regression utilities for economic analysis.

This module provides utilities for performing linear regression analysis
on economic data, particularly for household-level analysis. It handles
data preprocessing, model fitting, and prediction.

The module supports:
- Linear regression modeling
- Feature normalization
- Automatic handling of edge cases
- Prediction generation
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_linear(
    household_data: pd.DataFrame,
    independents: list[str],
    dependent: str,
) -> tuple[np.ndarray, Any]:
    """Fit a linear regression model to household data.

    This function fits a linear regression model using specified
    independent variables to predict a dependent variable. It handles
    data preprocessing and edge cases.

    Args:
        household_data: DataFrame containing household-level data
        independents: List of column names for independent variables
        dependent: Column name of dependent variable

    Returns:
        tuple:
        - np.ndarray: Predicted values from the model
        - Any: Fitted regression model (or None if no independents)

    Note:
        - If no independent variables are provided, returns mean of
          dependent variable and None for the model
        - Features are normalized by their sum before fitting

    Example:
        predictions, model = fit_linear(
            data,
            independents=["income", "wealth"],
            dependent="consumption"
        )
    """
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

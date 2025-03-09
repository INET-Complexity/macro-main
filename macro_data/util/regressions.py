"""
This module provides utilities for performing regression analysis on economic data,
with a focus on handling missing values and appropriate data normalization. It
implements linear regression with preprocessing steps suitable for economic time
series and cross-sectional data.

The module combines scikit-learn's regression capabilities with custom preprocessing
steps that are particularly relevant for economic data:
1. Missing value imputation using column means
2. Feature normalization through sum-to-one scaling
3. Linear regression with flexible model specification

Key features:
- Automatic handling of missing values
- Feature normalization appropriate for economic variables
- Support for multiple independent variables
- Flexible model specification through scikit-learn

Example:
    ```python
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from macro_data.util.regressions import fit_linear

    # Create sample economic data
    data = pd.DataFrame({
        'gdp': [100, 102, 98, 103, 101],
        'consumption': [80, 82, 79, 83, 81],
        'investment': [20, np.nan, 19, 21, 20]  # Missing value
    })

    # Fit model predicting GDP from consumption and investment
    model = LinearRegression()
    predictions = fit_linear(
        data=data,
        independents=['consumption', 'investment'],
        dependent='gdp',
        model=model
    )
    ```
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


# TODO does the normalisation make sense?
def fit_linear(
    data: pd.DataFrame,
    independents: list[str],
    dependent: str,
    model: LinearRegression,
) -> np.ndarray:
    """
    Fit a linear regression model with preprocessing for economic data.

    This function performs several steps to prepare and fit economic data:
    1. Extracts dependent and independent variables
    2. Imputes missing values using column means
    3. Normalizes features by dividing by column sums
    4. Fits the specified linear regression model
    5. Generates predictions

    The preprocessing steps are designed for economic data where:
    - Missing values are common and can be reasonably imputed with means
    - Variables often need to be normalized but should maintain proportions
    - Relationships are expected to be approximately linear

    Args:
        data (pd.DataFrame): Input DataFrame containing both dependent and
            independent variables. All variables should be numeric or
            coercible to numeric.
        independents (list[str]): Names of columns to use as independent
            variables (features). If empty, returns mean of dependent variable.
        dependent (str): Name of the column to predict (target variable).
        model (LinearRegression): Initialized but unfitted scikit-learn
            linear regression model. Can be configured with custom parameters.

    Returns:
        np.ndarray: Array of predicted values for the dependent variable,
            same length as input data.

    Notes:
        - Missing values are imputed before model fitting
        - Features are normalized by dividing by column sums
        - If no independent variables are provided, returns mean of dependent
        - Underflow warnings are ignored during computation
        - The function modifies the model in-place by fitting it

    Example:
        ```python
        # Predict GDP growth using multiple indicators
        model = LinearRegression(fit_intercept=True)
        predictions = fit_linear(
            data=economic_data,
            independents=['consumption_growth', 'investment_growth'],
            dependent='gdp_growth',
            model=model
        )

        # Access model coefficients
        coefficients = dict(zip(independents, model.coef_))
        ```
    """
    dependent_data = data[dependent].values
    x = data[independents].values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    np.seterr(under="ignore")
    x = imp_mean.fit_transform(x)
    if len(independents) == 0:
        return dependent_data.mean()

    # Fit
    # x = (x - x.min()) / (x.max() - x.min())
    x /= x.sum(axis=0)
    model.fit(x, dependent_data)
    pred = model.predict(x)  # noqa

    return pred

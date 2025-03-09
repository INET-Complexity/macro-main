"""
This module provides utilities for cleaning and preprocessing economic data, with a focus
on outlier detection and removal using statistical methods. It implements robust data
cleaning techniques suitable for economic time series and cross-sectional data.

The module uses multivariate normal distributions to identify outliers in multiple
dimensions simultaneously, making it particularly useful for cleaning related economic
variables that may have complex relationships.

Key features:
- Multivariate outlier detection
- Support for log-transformed probability densities
- Configurable outlier thresholds
- Preservation of data structure with NaN replacement

Example:
    ```python
    import pandas as pd
    from macro_data.util.clean_data import remove_outliers

    # Create sample data
    data = pd.DataFrame({
        'gdp': [100, 102, 98, 500, 101],  # 500 is an outlier
        'consumption': [80, 82, 79, 400, 81]  # 400 is an outlier
    })

    # Remove outliers from both columns
    cleaned_data = remove_outliers(
        data=data,
        cols=['gdp', 'consumption'],
        quantile=0.05
    )
    ```
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def remove_outliers(
    data: pd.DataFrame,
    cols: list[str],
    quantile: float = 0.05,
    use_logpdf=True,
) -> pd.DataFrame:
    """
    Remove outliers from specified columns using multivariate normal distribution.

    This function identifies and removes outliers by:
    1. Computing the multivariate normal distribution of the selected columns
    2. Calculating probability densities for each observation
    3. Identifying observations below the specified quantile threshold
    4. Replacing outlier values with NaN

    The function can work with either regular or log-transformed probability
    densities, making it suitable for both normally distributed and log-normally
    distributed data.

    Args:
        data (pd.DataFrame): Input DataFrame containing the data to clean.
        cols (list[str]): Column names to check for outliers. These columns
            should contain numeric data or data that can be coerced to numeric.
        quantile (float, optional): Threshold for outlier detection. Observations
            with probability density below this quantile are considered outliers.
            Defaults to 0.05 (5th percentile).
        use_logpdf (bool): Whether to use log probability density function.
            Set to True for better numerical stability with widely varying values.
            Defaults to True.

    Returns:
        pd.DataFrame: Copy of input DataFrame with outliers replaced by NaN in
            the specified columns. Original DataFrame remains unchanged.

    Notes:
        - The function handles missing values by dropping them before computing
          the distribution parameters, then reintroducing them in the output.
        - Non-numeric values in specified columns are coerced to numeric,
          with non-convertible values becoming NaN.
        - The covariance matrix may become singular with highly correlated
          variables, but this is handled by allowing singular matrices.

    Example:
        ```python
        # Remove outliers from GDP and consumption data
        clean_df = remove_outliers(
            data=economic_data,
            cols=['gdp', 'consumption'],
            quantile=0.01,  # More conservative threshold
            use_logpdf=True
        )
        ```
    """
    # Load
    data_r = data[cols]
    for col in cols:
        data_r.loc[:, col] = pd.to_numeric(data_r[col], errors="coerce")  #
    data_r = data_r.dropna().astype(float)

    # Find outliers
    covariance_matrix = np.cov(data_r.values.T)
    mean = data_r.values.mean(axis=0)
    model = multivariate_normal(cov=covariance_matrix, mean=mean, allow_singular=True)
    if use_logpdf:
        p = model.logpdf(data[cols].astype(float)).reshape(-1)
        outliers = p <= np.quantile(p, quantile)
    else:
        p = model.pdf(data[cols].astype(float)).reshape(-1)
        outliers = p <= np.quantile(p, quantile)
    # Remove them
    data.loc[outliers, cols] = np.nan

    return data

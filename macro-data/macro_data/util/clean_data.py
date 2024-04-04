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
    Remove outliers from the specified columns of a DataFrame. Outliers are detected using a multivariate normal distribution,
    and outliers are considered to be observations with a probability density below the specified quantile.

    Args:
        data (pd.DataFrame): The input DataFrame.
        cols (list[str]): The list of column names to remove outliers from.
        quantile (float, optional): The quantile threshold for outlier detection. Defaults to 0.05.
        use_logpdf (bool): Whether to use the logpdf when computing the probability density

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.

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

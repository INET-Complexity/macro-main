import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def remove_outliers(
    data: pd.DataFrame,
    cols: list[str],
    quantile: float = 0.05,
) -> pd.DataFrame:
    # Load
    data_r = data[cols]
    for col in cols:
        data_r.loc[:, col] = pd.to_numeric(data_r[col], errors="coerce")  #
    data_r = data_r.dropna().astype(float)

    # Find outliers
    covariance_matrix = np.cov(data_r.values.T)
    mean = data_r.values.mean(axis=0)
    model = multivariate_normal(cov=covariance_matrix, mean=mean, allow_singular=True)
    p = model.pdf(data[cols].astype(float)).reshape(-1)
    outliers = p <= np.quantile(p, quantile)

    # Remove them
    data.loc[outliers, cols] = np.nan

    return data

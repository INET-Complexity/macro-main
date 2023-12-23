import numpy as np
import pandas as pd


def get_quantiles(n: int, data: np.ndarray) -> np.ndarray:
    o = data.argpartition(np.arange(1, n) * len(data) // n)
    quantile_groups = np.empty(len(data), int)
    quantile_groups[o] = np.arange(len(data)) * n // len(data)
    return quantile_groups


def partition_into_quintiles(data: np.ndarray) -> np.ndarray:
    d = pd.Series(data)
    quintiles = pd.qcut(d, 5, labels=False)
    return quintiles.values

import numpy as np


def get_quantiles(n: int, data: np.ndarray) -> np.ndarray:
    o = data.argpartition(np.arange(1, n) * len(data) // n)
    quantile_groups = np.empty(len(data), int)
    quantile_groups[o] = np.arange(len(data)) * n // len(data)
    return quantile_groups


def partition_into_quintiles(data: np.ndarray) -> np.ndarray:
    data_ind = np.argsort(data)
    step_size = int(np.floor(len(data_ind) / 5))
    quintiles = np.zeros_like(data, dtype=int)
    quintiles[data_ind[0:step_size]] = 0
    quintiles[data_ind[step_size : 2 * step_size]] = 1
    quintiles[data_ind[2 * step_size : 3 * step_size]] = 2
    quintiles[data_ind[3 * step_size : 4 * step_size]] = 3
    quintiles[data_ind[4 * step_size :]] = 4

    return quintiles

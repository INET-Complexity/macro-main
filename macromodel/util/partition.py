"""Data partitioning utilities for quantile analysis.

This module provides utilities for partitioning numerical data into
quantiles and quintiles. It's used throughout the model for analyzing
distributions and creating stratified groupings of economic agents.

The module supports:
- Arbitrary quantile partitioning
- Quintile-specific partitioning
- Efficient array operations
- Memory-efficient implementations
"""

import numpy as np


def get_quantiles(n: int, data: np.ndarray) -> np.ndarray:
    """Partition data into n equal-sized quantile groups.

    This function efficiently partitions data into n quantiles using
    numpy's partition algorithm, avoiding a full sort operation.

    Args:
        n: Number of quantile groups to create
        data: Input array to partition

    Returns:
        np.ndarray: Array of same length as input, containing quantile
            group assignments (0 to n-1) for each element

    Example:
        groups = get_quantiles(4, income_data)  # Quartile assignments
    """
    o = data.argpartition(np.arange(1, n) * len(data) // n)
    quantile_groups = np.empty(len(data), int)
    quantile_groups[o] = np.arange(len(data)) * n // len(data)
    return quantile_groups


def partition_into_quintiles(data: np.ndarray) -> np.ndarray:
    """Partition data into five equal-sized quintile groups.

    This function sorts data and assigns elements to quintile groups,
    useful for economic analysis like income distribution studies.

    Args:
        data: Input array to partition into quintiles

    Returns:
        np.ndarray: Array of same length as input, containing quintile
            assignments (0-4) for each element, where:
            - 0: Bottom quintile (0-20%)
            - 1: Second quintile (20-40%)
            - 2: Middle quintile (40-60%)
            - 3: Fourth quintile (60-80%)
            - 4: Top quintile (80-100%)

    Example:
        quintiles = partition_into_quintiles(wealth_data)
        top_quintile_mask = (quintiles == 4)
    """
    data_ind = np.argsort(data)
    step_size = int(np.floor(len(data_ind) / 5))
    quintiles = np.zeros_like(data, dtype=int)
    quintiles[data_ind[0:step_size]] = 0
    quintiles[data_ind[step_size : 2 * step_size]] = 1
    quintiles[data_ind[2 * step_size : 3 * step_size]] = 2
    quintiles[data_ind[3 * step_size : 4 * step_size]] = 3
    quintiles[data_ind[4 * step_size :]] = 4

    return quintiles

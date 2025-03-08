"""
This module provides utilities for partitioning numerical data into quantiles and
quintiles, which is particularly useful for economic analysis where data often
needs to be grouped into equal-sized bins (e.g., income quintiles, wealth deciles).

The module offers two main functions:
- get_quantiles: Flexible partitioning into n equal-sized groups
- partition_into_quintiles: Specific partitioning into quintiles (5 groups)

Key features:
- Memory-efficient quantile computation
- Support for numpy arrays and pandas Series
- Equal-sized group partitioning
- Handling of ties through pandas qcut

Example:
    ```python
    import numpy as np
    from macro_data.util.partition import get_quantiles, partition_into_quintiles

    # Sample income data
    income_data = np.array([30000, 45000, 60000, 75000, 90000,
                           35000, 50000, 65000, 80000, 95000])

    # Split into deciles (10 groups)
    income_deciles = get_quantiles(10, income_data)

    # Split into quintiles (5 groups)
    income_quintiles = partition_into_quintiles(income_data)
    ```
"""

import numpy as np
import pandas as pd


def get_quantiles(n: int, data: np.ndarray) -> np.ndarray:
    """
    Partition data into n equal-sized groups using efficient numpy operations.

    This function uses numpy's partition algorithm to efficiently split data
    into n quantiles without fully sorting the array. It's particularly useful
    for large datasets where full sorting would be computationally expensive.

    Args:
        n (int): Number of groups to partition the data into. Common values are:
            - 4 for quartiles
            - 5 for quintiles
            - 10 for deciles
            - 100 for percentiles
        data (np.ndarray): 1-D numpy array containing the values to partition.
            Values should be numeric and comparable.

    Returns:
        np.ndarray: Array of same length as input, containing integer labels
            [0, n-1] indicating which quantile group each value belongs to.

    Notes:
        - The function attempts to create equal-sized groups, but with
          indivisible lengths, some groups may differ by one element
        - The implementation is memory-efficient as it avoids full sorting
        - Groups are zero-indexed, e.g., for quintiles: 0-4

    Example:
        ```python
        # Split wealth data into deciles
        wealth_data = np.array([100, 200, 150, 300, 250, 400, 350, 500, 450, 600])
        deciles = get_quantiles(10, wealth_data)
        # Result: array([0, 2, 1, 4, 3, 6, 5, 8, 7, 9])
        ```
    """
    o = data.argpartition(np.arange(1, n) * len(data) // n)
    quantile_groups = np.empty(len(data), int)
    quantile_groups[o] = np.arange(len(data)) * n // len(data)
    return quantile_groups


def partition_into_quintiles(data: np.ndarray) -> np.ndarray:
    """
    Partition data specifically into quintiles (5 equal-sized groups).

    This function uses pandas' qcut to create quintiles, which handles ties
    and edge cases more robustly than the pure numpy approach. It's particularly
    useful for economic analysis where quintile-based analysis is common
    (e.g., income quintiles, consumption quintiles).

    Args:
        data (np.ndarray): 1-D numpy array containing the values to partition.
            Values should be numeric and comparable.

    Returns:
        np.ndarray: Array of same length as input, containing integer labels
            [0-4] indicating which quintile group each value belongs to.

    Notes:
        - Uses pandas qcut which handles ties more robustly than numpy
        - Groups are zero-indexed (0-4)
        - Equal-sized groups are attempted but may not be exactly equal
          due to ties

    Example:
        ```python
        # Split income data into quintiles
        income_data = np.array([30000, 45000, 60000, 75000, 90000,
                               35000, 50000, 65000, 80000, 95000])
        quintiles = partition_into_quintiles(income_data)
        # Result: array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        ```
    """
    d = pd.Series(data)
    quintiles = pd.qcut(d, 5, labels=False)
    return quintiles.values

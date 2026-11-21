"""Histogram computation utilities for data analysis.

This module provides utilities for computing normalized histograms from
numerical data, with support for scaling and normalization. It's used
throughout the model for analyzing distributions of various economic
metrics.

The module supports:
- Normalized histogram computation
- Optional value scaling
- Configurable bin counts
- Empty data handling
- Range normalization
"""

from typing import Optional

import numpy as np


def get_histogram(values: np.ndarray, scale: Optional[int], bins: int = 40, normalise: bool = False) -> np.ndarray:
    """Compute a normalized histogram from numerical data.

    This function creates a histogram from input values, with options
    for scaling, normalization, and bin configuration. It handles
    edge cases like empty arrays and zero-range data.

    Args:
        values: Input data array to histogram
        scale: Optional scaling factor for values (e.g., 1000 for thousands)
        bins: Number of histogram bins (default: 40)
        normalise: Whether to normalize values to [0,1] range (default: False)

    Returns:
        np.ndarray: 2xN array containing:
            - Row 0: Normalized bin counts (sums to 1)
            - Row 1: Bin edges
            For empty input, returns array of NaN values

    Example:
        hist = get_histogram(
            values=data,
            scale=1000,
            bins=50,
            normalise=True
        )
        counts = hist[0, :-1]  # Normalized counts
        edges = hist[1, :]     # Bin edges
    """

    values = fillna(values)
    if len(values) == 0:
        return np.full((2, bins + 1), np.nan)
    if normalise:
        diff = np.max(values) - np.min(values)
        if diff > 0:
            values = (values - np.min(values)) / diff
        else:
            values = values - np.min(values)
    if scale is None:
        hist, bin_edges = np.histogram(values, bins=bins)
    else:
        hist, bin_edges = np.histogram(values / scale, bins=bins)
    hist = hist.astype(float)
    hist /= hist.sum()
    return np.array([np.concatenate((hist, [np.nan])), bin_edges])


def fillna(array: np.ndarray, value: float = 0):
    """Fill NaN values in an array with a specified value.

    Args:
        array (np.ndarray): Input array with potential NaN values.
        value (float, optional): Value to replace NaN. Defaults to 0.

    Returns:
        np.ndarray: Array with NaN values replaced.
    """
    return np.where(np.isnan(array), value, array)

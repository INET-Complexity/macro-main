import numpy as np

from typing import Optional


def get_histogram(values: np.ndarray, scale: Optional[int], bins: int = 40, normalise: bool = False) -> np.ndarray:
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

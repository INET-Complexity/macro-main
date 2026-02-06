import numpy as np


def create_bundle_matrix(good_bundle: np.ndarray) -> np.ndarray:
    """Create a normalized bundle matrix from industry-to-bundle mapping.

    For each bundle, assigns weight 1/n where n is the number of industries in the bundle.
    Industries not in a bundle are treated as singletons (weight 1 on themselves).

    Args:
        good_bundle (np.ndarray): Array mapping each industry to its bundle index.

    Returns:
        np.ndarray: (n_industries, n_bundles) matrix of normalized bundle weights.
    """
    n_industries = good_bundle.shape[0]
    n_bundles = good_bundle.max() + 1

    a = np.zeros((n_bundles, n_industries), dtype=float)

    for industry_idx, bundle_idx in enumerate(good_bundle):
        a[bundle_idx, industry_idx] = 1.0

    # Normalize each row so that entries sum to 1
    row_sums = a.sum(axis=1, keepdims=True)
    aggregation_matrix = a / row_sums

    return aggregation_matrix.T

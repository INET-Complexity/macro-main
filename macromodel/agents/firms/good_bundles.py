import numpy as np
from typing import List


def create_good_bundle(n_industries: int, bundles: List[List[int]]) -> np.ndarray:
    """Assign bundle indices to industries based on substitution groups.

    For a given number of industries, assign each industry to a bundle index.
    Industries listed together in a bundle share the same index. Industries not
    listed in any bundle are assigned unique bundle indices individually.

    After assignment, bundle indices are relabeled to ensure dense, increasing
    numbering based on first appearance.

    Args:
        n_industries (int): Total number of industries.
        bundles (List[List[int]]): List of substitution bundles, where each
            bundle is a list of industry indices.

    Returns:
        np.ndarray: Array of shape (n_industries,) mapping each industry to its bundle index.
    """

    good_bundle = np.full(n_industries, -1, dtype=int)
    bundle_idx = 0

    # Assign bundle indices to industries included in bundles
    for bundle in bundles:
        for industry in bundle:
            good_bundle[industry] = bundle_idx
        bundle_idx += 1

    # Assign remaining industries that are not in any bundle
    for i in range(n_industries):
        if good_bundle[i] == -1:
            good_bundle[i] = bundle_idx
            bundle_idx += 1

    # Relabel to ensure increasing order
    seen = {}
    new_labels = []
    for x in good_bundle:
        if x not in seen:
            seen[x] = len(seen)
        new_labels.append(seen[x])

    good_bundle = np.array(new_labels, dtype=int)

    return good_bundle

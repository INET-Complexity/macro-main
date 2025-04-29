import numpy as np
import pytest

from macromodel.configurations.firms_configuration import create_good_bundle


def test_create_good_bundle_empty():
    n_industries = 5
    bundles = []
    expected = np.array([0, 1, 2, 3, 4])
    result = np.array(create_good_bundle(n_industries, bundles))
    assert np.array_equal(result, expected)


def test_create_good_bundle_nonempty():
    n_industries = 5
    bundles = [[0, 1], [3, 4]]
    expected = np.array([0, 0, 1, 2, 2])
    result = np.array(create_good_bundle(n_industries, bundles))
    assert np.array_equal(result, expected)


def test_create_good_bundle_partial():
    n_industries = 6
    bundles = [[1, 4]]
    expected = np.array([0, 1, 2, 3, 1, 4])
    result = np.array(create_good_bundle(n_industries, bundles))
    assert np.array_equal(result, expected)


def test_create_good_bundle_all_in_bundle():
    n_industries = 4
    bundles = [[0, 1, 2, 3]]
    expected = np.array([0, 0, 0, 0])
    result = np.array(create_good_bundle(n_industries, bundles))
    assert np.array_equal(result, expected)


def test_create_good_bundle_invalid_index():
    n_industries = 3
    bundles = [[0, 3]]  # 3 is invalid
    with pytest.raises(IndexError):
        _ = create_good_bundle(n_industries, bundles)

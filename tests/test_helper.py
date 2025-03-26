import pytest
import numpy as np
from exect.helper import (
    remove_duplicates_accordingly,
    sort_accordingly,
)  # replace with actual module name


def test_remove_duplicates_accordingly():
    a = np.array([1, 2, 2, 3, 4, 4, 4, 5])
    b = np.array([8, 7, 6, 5, 4, 3, 2, 1])
    expected_a = np.array([1, 2, 3, 4, 5])
    expected_b = np.array([8, 7, 5, 4, 1])

    result_a, result_b = remove_duplicates_accordingly(a, b)

    assert np.array_equal(result_a, expected_a)
    assert np.array_equal(result_b, expected_b)


def test_remove_duplicates_accordingly_strings():
    a = np.array([1, 2, 2, 3, 4, 4, 4, 5])
    b = np.array(["eight", "seven", "six", "five", "four", "three", "two", "one"])
    expected_a = np.array([1, 2, 3, 4, 5])
    expected_b = np.array(["eight", "seven", "five", "four", "one"])

    result_a, result_b = remove_duplicates_accordingly(a, b)

    assert np.array_equal(result_a, expected_a)
    assert np.array_equal(result_b, expected_b)


def test_remove_duplicates_accordingly_error():
    a = np.array([1, 2, 3])
    b = np.array([1, 2])
    with pytest.raises(AssertionError):
        remove_duplicates_accordingly(a, b)


def test_sort_accordingly():
    a = np.array([8, 4, 7, 1, 2, 5, 3, 6])
    b = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    expected_a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    expected_b = np.array([4, 5, 7, 2, 6, 8, 3, 1])
    a, b = sort_accordingly(a, b)

    assert np.array_equal(a, expected_a)
    assert np.array_equal(b, expected_b)


def test_sort_accordingly_strings():
    a = np.array([8, 4, 7, 1, 2, 5, 3, 6])
    b = np.array(["one", "two", "three", "four", "five", "six", "seven", "eight"])

    expected_a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    expected_b = np.array(
        ["four", "five", "seven", "two", "six", "eight", "three", "one"]
    )

    result_a, result_b = sort_accordingly(a, b)

    assert np.array_equal(result_a, expected_a)
    assert np.array_equal(result_b, expected_b)

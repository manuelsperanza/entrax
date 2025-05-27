import pytest
import numpy as np
from numpy.testing import assert_allclose
from entrax.utils.numeric import float_gcd, float_gcd_reduce, linespace_extra

# Test float_gcd function
def test_float_gcd_basic():
    # 0.5 and 0.25 have a common factor 0.25
    result = float_gcd(0.5, 0.25)
    assert result == pytest.approx(0.25)

def test_float_gcd_with_tolerance():
    # For tol=1e-3, 1.2 and 0.8 scaled yield a gcd of 0.4
    result = float_gcd(1.2, 0.8, tol=1e-3)
    assert result == pytest.approx(0.4)

def test_float_gcd_negative():
    # Negative numbers should yield the same common factor
    result = float_gcd(-1.2, 0.8)
    assert result == pytest.approx(0.4)

def test_float_gcd_invalid_tolerance():
    with pytest.raises(ValueError):
        float_gcd(1.0, 2.0, tol=0)

# Test float_gcd_reduce function
def test_float_gcd_reduce():
    values = [0.5, 0.75, 1.0]  # expected common factor is 0.25
    result = float_gcd_reduce(values)
    assert result == pytest.approx(0.25)

# Test linespace_extra function for two points (should behave like numpy.linspace)
def test_linespace_extra_two_points():
    points = np.array([0.0, 10.0])
    n = 5
    result = linespace_extra(points.copy(), n)
    expected = np.linspace(0, 10, n)
    assert_allclose(result, expected)

# Test linespace_extra function with multiple points
def test_linespace_extra_multiple_points():
    points = np.array([0.0, 3.0, 7.0])
    n = 10
    result = linespace_extra(points.copy(), n)
    # Ensure the range is covered and values are in ascending order
    assert result[0] == pytest.approx(0.0)
    assert result[-1] >= 7.0
    for x in result:
         assert 0.0 <= x <= 7.0

# Test error when less than two points are provided
def test_linespace_extra_error():
    points = np.array([1.0])
    with pytest.raises(ValueError):
         linespace_extra(points.copy(), 5)

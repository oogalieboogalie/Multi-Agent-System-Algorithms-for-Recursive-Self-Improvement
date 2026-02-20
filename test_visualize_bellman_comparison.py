import sys
from unittest.mock import MagicMock
import math

# Mock numpy before importing the module that uses it
mock_np = MagicMock()
mock_np.sqrt = math.sqrt
mock_np.log = math.log
sys.modules["numpy"] = mock_np

# Mock matplotlib to avoid import errors
mock_plt = MagicMock()
sys.modules["matplotlib"] = mock_plt
sys.modules["matplotlib.pyplot"] = mock_plt

import pytest
from visualize_bellman_comparison import calculate_bellman_gain

def test_calculate_bellman_gain_values():
    """Test calculate_bellman_gain with specific inputs."""
    entropy = 0.5
    depth = 2
    # The implementation uses p = 1.0 / (depth + 3), consistent with documentation
    # p = 1.0 / 5 = 0.2
    # bellman_weight = sqrt(entropy^2 + p * log(1/p))
    # bellman_weight = sqrt(0.25 + 0.2 * log(5))
    # log(5) approx 1.6094
    # bellman_weight = sqrt(0.25 + 0.32188) = sqrt(0.57188) approx 0.7562
    # gain = bellman_weight - entropy = 0.7562 - 0.5 = 0.2562

    expected_gain = math.sqrt(entropy**2 + (1.0/(depth+3)) * math.log(depth+3)) - entropy
    calculated_gain = calculate_bellman_gain(entropy, depth)

    assert math.isclose(calculated_gain, expected_gain, rel_tol=1e-5)

def test_calculate_bellman_gain_decreases_with_depth():
    """Test that gain decreases as depth increases.

    With p = 1/(depth+3), at depth 0, p = 1/3 approx 0.333.
    Since p < 1/e (approx 0.368), the term p*log(1/p) increases as p increases.
    As depth increases, p decreases, so p*log(1/p) decreases, meaning gain decreases.
    """
    entropy = 0.5
    previous_gain = calculate_bellman_gain(entropy, 0)

    for depth in range(1, 10):
        current_gain = calculate_bellman_gain(entropy, depth)
        assert current_gain < previous_gain
        previous_gain = current_gain

def test_calculate_bellman_gain_positive():
    """Test that gain is generally positive."""
    # Since bellman_volatility_weight(p, q) = sqrt(q^2 + p*log(1/p))
    # And p*log(1/p) > 0 for 0 < p < 1
    # sqrt(q^2 + positive) > q (assuming q >= 0)
    # So gain > 0

    entropy = 0.5
    depth = 5
    gain = calculate_bellman_gain(entropy, depth)
    assert gain > 0

def test_calculate_bellman_gain_zero_entropy():
    """Test behavior with zero entropy."""
    entropy = 0.0
    depth = 2
    # p = 0.2
    # gain = sqrt(0 + 0.2*log(5)) - 0
    gain = calculate_bellman_gain(entropy, depth)
    assert gain > 0

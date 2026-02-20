import sys
from unittest.mock import MagicMock
import math

# Mock numpy before importing the module that uses it
mock_np = MagicMock()
mock_np.sqrt = math.sqrt
mock_np.log = math.log
sys.modules["numpy"] = mock_np

import pytest
import numpy as np
from DEMONSTRATION_bellman_algorithms import (
    bellman_volatility_weight,
    calculate_exit_signal,
)

def test_bellman_volatility_weight_base_cases():
    """Test the U(p, q) formula with base cases."""
    # If p <= 0 or p >= 1, should return q
    assert bellman_volatility_weight(0, 0.5) == 0.5
    assert bellman_volatility_weight(-0.1, 0.5) == 0.5
    assert bellman_volatility_weight(1.0, 0.5) == 0.5
    assert bellman_volatility_weight(1.5, 0.5) == 0.5

def test_bellman_volatility_weight_amplification_vs_linear():
    """Test that Bellman weight provides amplification compared to linear p*q for rare events."""
    q = 0.5
    # Rare event
    p = 0.01

    linear_gain = q * p
    bellman_weight = bellman_volatility_weight(p, q)
    bellman_gain = bellman_weight - q

    # Bellman gain should be higher than linear gain for rare events
    # bellman_gain = sqrt(0.25 + 0.01*log(100)) - 0.5 = 0.544 - 0.5 = 0.044
    # linear_gain = 0.5 * 0.01 = 0.005
    assert bellman_gain > linear_gain

def test_calculate_exit_signal_depth_zero():
    """Depth 0 should never exit to allow at least one expansion."""
    assert calculate_exit_signal(entropy=0.5, cost=1.0, depth=0) is False
    assert calculate_exit_signal(entropy=1.0, cost=10.0, depth=0) is False

def test_calculate_exit_signal_cost_influence():
    """Higher cost should lead to earlier exit."""
    entropy = 0.5
    depth = 1

    # Very low cost should not exit
    assert calculate_exit_signal(entropy, cost=0.0001, depth=depth) is False

    # Very high cost should exit
    assert calculate_exit_signal(entropy, cost=1.0, depth=depth) is True

def test_calculate_exit_signal_depth_influence():
    """Increasing depth should eventually trigger an exit for non-zero cost."""
    entropy = 0.5
    cost = 0.01

    # At some depth, it must stop because p*log(1/p) goes to 0 as depth increases (p=1/(depth+1))
    exit_found = False
    for depth in range(1, 1000):
        if calculate_exit_signal(entropy, cost, depth):
            exit_found = True
            break
    assert exit_found is True

def test_calculate_exit_signal_entropy_influence():
    """Higher entropy should decrease marginal benefit, leading to earlier exit."""
    # marginal_benefit = sqrt(entropy^2 + p*log(1/p)) - entropy
    # This decreases as entropy increases.

    cost = 0.05
    depth = 5

    # We check that benefit(high_entropy) < benefit(low_entropy)
    p = 1.0 / (depth + 1)
    benefit = lambda e: math.sqrt(e**2 + p * math.log(1/p)) - e

    assert benefit(2.0) < benefit(0.1)

    # Test actual exit signal behavior
    # If it exits at low entropy, it should also exit at high entropy (for same cost and depth)
    if calculate_exit_signal(0.1, cost, depth):
        assert calculate_exit_signal(2.0, cost, depth) is True

def test_calculate_exit_signal_zero_cost():
    """If cost is 0, it should never exit."""
    # At finite depth, p > 0, so p*log(1/p) > 0, so benefit > 0.
    for depth in range(1, 100):
        assert calculate_exit_signal(entropy=0.5, cost=0, depth=depth) is False

def test_bellman_volatility_weight_standard_values():
    """Test with standard values p=0.5, q=0.5."""
    p = 0.5
    q = 0.5
    # Expected: sqrt(0.5^2 + 0.5 * log(1/0.5))
    expected = math.sqrt(0.25 + 0.5 * math.log(2))
    assert math.isclose(bellman_volatility_weight(p, q), expected, rel_tol=1e-9)

def test_bellman_volatility_weight_edge_cases():
    """Test edge cases for p close to 0 and 1, and q=0."""
    # p close to 0
    p_small = 1e-9
    q = 0.5
    # Should be close to sqrt(q^2) = q because p*log(1/p) -> 0
    val_small = bellman_volatility_weight(p_small, q)
    assert val_small > q
    assert math.isclose(val_small, q, abs_tol=1e-4)

    # p close to 1
    p_large = 1.0 - 1e-9
    # log(1/p) approx log(1) = 0. So result approx q.
    val_large = bellman_volatility_weight(p_large, q)
    # The term p*log(1/p) is approx 1e-9, so result is approx q + 1e-9.
    # We use a slightly larger tolerance to account for this small shift.
    assert math.isclose(val_large, q, rel_tol=1e-8)

    # q = 0
    p = 0.5
    # result = sqrt(0 + p*log(1/p))
    expected_q0 = math.sqrt(p * math.log(1/p))
    assert math.isclose(bellman_volatility_weight(p, 0), expected_q0, rel_tol=1e-9)

def test_bellman_volatility_weight_monotonicity():
    """Test that output increases with q."""
    p = 0.5
    q1 = 0.4
    q2 = 0.6
    assert bellman_volatility_weight(p, q1) < bellman_volatility_weight(p, q2)

def test_bellman_volatility_weight_peak_behavior():
    """Test that p*log(1/p) component peaks around p=1/e."""
    # We test with q=0 to isolate the p term
    p_peak = 1.0 / math.e
    val_peak = bellman_volatility_weight(p_peak, 0)

    # Check neighbors
    val_left = bellman_volatility_weight(p_peak - 0.01, 0)
    val_right = bellman_volatility_weight(p_peak + 0.01, 0)

    assert val_peak > val_left
    assert val_peak > val_right

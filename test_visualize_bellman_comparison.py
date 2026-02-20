import sys
from unittest.mock import MagicMock
import pytest

# Mock numpy and matplotlib before importing the module
mock_np = MagicMock()
# Fix for pytest interaction with numpy mock
mock_np.bool_ = bool
mock_np.float64 = float

mock_plt = MagicMock()
sys.modules["numpy"] = mock_np
sys.modules["matplotlib"] = mock_plt
sys.modules["matplotlib.pyplot"] = mock_plt

from visualize_bellman_comparison import heuristic_weight

def test_heuristic_weight_rare_event():
    """Test heuristic_weight with a rare event (low p)."""
    p = 0.01
    q = 0.5
    # Formula: q * (1 - p) + q
    # 0.5 * (0.99) + 0.5 = 0.495 + 0.5 = 0.995
    expected = 0.995
    result = heuristic_weight(p, q)
    assert result == pytest.approx(expected)

def test_heuristic_weight_common_event():
    """Test heuristic_weight with a common event (high p)."""
    p = 0.9
    q = 0.5
    # Formula: q * (1 - p) + q
    # 0.5 * (0.1) + 0.5 = 0.05 + 0.5 = 0.55
    expected = 0.55
    result = heuristic_weight(p, q)
    assert result == pytest.approx(expected)

def test_heuristic_weight_certainty():
    """Test heuristic_weight when p=1 (certainty/common)."""
    p = 1.0
    q = 0.5
    # Formula: q * (0) + q = q
    expected = 0.5
    result = heuristic_weight(p, q)
    assert result == pytest.approx(expected)

def test_heuristic_weight_impossibility():
    """Test heuristic_weight when p=0 (impossible/extremely rare)."""
    p = 0.0
    q = 0.5
    # Formula: q * (1) + q = 2q
    expected = 1.0
    result = heuristic_weight(p, q)
    assert result == pytest.approx(expected)

def test_heuristic_weight_quality_scaling():
    """Test that heuristic_weight scales linearly with quality q."""
    p = 0.5
    q1 = 0.2
    q2 = 0.4

    # Formula: q * (1 - p) + q = q * (2 - p)
    # For p=0.5, multiplier is 1.5

    w1 = heuristic_weight(p, q1) # 0.2 * 1.5 = 0.3
    w2 = heuristic_weight(p, q2) # 0.4 * 1.5 = 0.6

    assert w1 == pytest.approx(0.3)
    assert w2 == pytest.approx(0.6)
    assert w2 == pytest.approx(2 * w1)

def test_heuristic_weight_linearity_check():
    """Verify the linear nature of the heuristic boost."""
    q = 1.0
    # p goes from 0 to 1
    # Weight goes from 2q to q

    w_0 = heuristic_weight(0.0, q)   # 2.0
    w_05 = heuristic_weight(0.5, q)  # 1.5
    w_1 = heuristic_weight(1.0, q)   # 1.0

    # Check that midpoint is exactly average of endpoints (linear behavior)
    assert w_05 == pytest.approx((w_0 + w_1) / 2)

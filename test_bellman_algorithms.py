import sys
import math
from unittest.mock import MagicMock

# ================================================================================
# MOCK NUMPY IMPLEMENTATION
# ================================================================================

class FakeArray:
    """A fake numpy array class supporting basic operations."""
    def __init__(self, data):
        self.data = [float(x) for x in data]

    def __len__(self):
        return len(self.data)

    def sum(self):
        return sum(self.data)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return FakeArray([x / other for x in self.data])
        raise NotImplementedError(f"Division not supported for {type(other)}")

    def __mul__(self, other):
        if isinstance(other, FakeArray):
            return FakeArray([x * y for x, y in zip(self.data, other.data)])
        if isinstance(other, (int, float)):
            return FakeArray([x * other for x in self.data])
        raise NotImplementedError(f"Multiplication not supported for {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return FakeArray([x + other for x in self.data])
        raise NotImplementedError(f"Addition not supported for {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"FakeArray({self.data})"

class FakeNumpy:
    """A fake numpy module."""
    def array(self, data):
        return FakeArray(data)

    def sum(self, data):
        if hasattr(data, 'sum'):
            return data.sum()
        return sum(data)

    def log(self, data):
        if isinstance(data, FakeArray):
            return FakeArray([math.log(x) for x in data.data])
        return math.log(data)

    def sqrt(self, data):
        if isinstance(data, FakeArray):
            return FakeArray([math.sqrt(x) for x in data.data])
        return math.sqrt(data)

# Inject the fake numpy module
sys.modules["numpy"] = FakeNumpy()

import pytest
# Now we can import the module that uses numpy
from DEMONSTRATION_bellman_algorithms import (
    bellman_volatility_weight,
    calculate_exit_signal,
    RecursiveIntelligenceCascade,
    Criterion
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
    if calculate_exit_signal(0.1, cost, depth):
        assert calculate_exit_signal(2.0, cost, depth) is True

def test_calculate_exit_signal_zero_cost():
    """If cost is 0, it should never exit."""
    # At finite depth, p > 0, so p*log(1/p) > 0, so benefit > 0.
    for depth in range(1, 100):
        assert calculate_exit_signal(entropy=0.5, cost=0, depth=depth) is False

# ================================================================================
# TESTS FOR RecursiveIntelligenceCascade & Criterion
# ================================================================================

def test_criterion_initialization():
    """Test Criterion dataclass initialization."""
    c = Criterion("test", 0.5, lambda d, c: True)
    assert c.name == "test"
    assert c.weight == 0.5
    assert c.matcher("any", {}) is True

def test_ric_evaluate_sharp():
    """Test _evaluate_sharp method scoring."""
    # Define criteria
    criteria = [
        Criterion("rare", 0.9, lambda d, c: "rare" in d),
        Criterion("common", 0.5, lambda d, c: "common" in d)
    ]

    ric = RecursiveIntelligenceCascade(
        processor=lambda x: [],
        criteria=criteria,
        synthesizer=lambda x: None
    )

    # Match rare criterion: p = 1 - 0.9 = 0.1
    # bellman_volatility_weight(0.1, 0.0) > 0
    score_rare = ric._evaluate_sharp("rare event")
    assert score_rare > 0

    # Match common criterion: p = 1 - 0.5 = 0.5
    score_common = ric._evaluate_sharp("common event")
    assert score_common > 0

    # No match
    score_none = ric._evaluate_sharp("nothing")
    assert score_none == 0.0

def test_ric_prune_sharp():
    """Test _prune_sharp selects top thoughts."""
    # Use p=0.37 (approx) which gives max signal boost. weight ~ 0.63.
    ric = RecursiveIntelligenceCascade(
        processor=lambda x: [],
        criteria=[Criterion("high", 0.63, lambda d, c: "high" in d)],
        synthesizer=lambda x: None
    )

    thoughts = ["high1", "low1", "high2", "low2", "low3"]
    # Should select "high1", "high2" first (score > 0)

    pruned = ric._prune_sharp(thoughts)
    assert "high1" in pruned
    assert "high2" in pruned
    # With 5 items, all should be returned as default limit is 10
    assert len(pruned) == 5

    # Test limiting
    thoughts_many = [f"high{i}" for i in range(20)]
    pruned_many = ric._prune_sharp(thoughts_many)
    assert len(pruned_many) == 10

def test_ric_process_flow():
    """Test the process method runs and terminates."""
    # Mock processor to expand
    def processor(thought):
        if len(thought) < 10:
             return [thought + "a", thought + "b"]
        return []

    # Mock synthesizer
    synthesizer = MagicMock(return_value="result")

    # Mock criteria to always return some score > 0.3 to retain thoughts
    criteria = [Criterion("always", 0.63, lambda d, c: True)]

    ric = RecursiveIntelligenceCascade(
        processor=processor,
        criteria=criteria,
        synthesizer=synthesizer,
        thinking_cost=0.01,
        max_depth=5
    )

    result, metadata = ric.process("start")

    assert result == "result"
    assert metadata["exit_reason"] is not None
    assert metadata["thoughts_processed"] > 0
    # Should run for at least depth 1
    assert metadata["depth_reached"] > 0

def test_ric_bellman_exit():
    """Test that process exits due to Bellman Exit (high cost)."""
    # High cost ensuring immediate exit or early exit
    ric = RecursiveIntelligenceCascade(
        processor=lambda x: ["expand"],
        criteria=[Criterion("always", 0.63, lambda d, c: True)],
        synthesizer=lambda x: "result",
        thinking_cost=10.0, # Very high cost
        max_depth=20
    )

    result, metadata = ric.process("start")

    # Should exit very early due to cost > gain
    assert metadata["exit_reason"] == "BELLMAN_STOP"

    # Check that it didn't run till max depth
    assert metadata["depth_reached"] < 20

import sys
import math
import pytest
import importlib
from unittest.mock import MagicMock, patch

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

@pytest.fixture
def bellman_module():
    """Fixture to import the module with FakeNumpy injected."""
    fake_numpy = FakeNumpy()

    # We need to ensure 'numpy' is patched before importing the module
    with patch.dict(sys.modules, {"numpy": fake_numpy}):
        # Since module might be imported elsewhere or cached, reload is safer
        # But import_module works for first time
        if "DEMONSTRATION_bellman_algorithms" in sys.modules:
            module = importlib.reload(sys.modules["DEMONSTRATION_bellman_algorithms"])
        else:
            module = importlib.import_module("DEMONSTRATION_bellman_algorithms")

        yield module

    # Cleanup: remove from sys.modules to prevent leakage
    if "DEMONSTRATION_bellman_algorithms" in sys.modules:
        del sys.modules["DEMONSTRATION_bellman_algorithms"]

# ================================================================================
# TESTS
# ================================================================================

def test_bellman_volatility_weight_base_cases(bellman_module):
    """Test the U(p, q) formula with base cases."""
    bellman = bellman_module.bellman_volatility_weight
    # If p <= 0 or p >= 1, should return q
    assert bellman(0, 0.5) == 0.5
    assert bellman(-0.1, 0.5) == 0.5
    assert bellman(1.0, 0.5) == 0.5
    assert bellman(1.5, 0.5) == 0.5

def test_bellman_volatility_weight_amplification_vs_linear(bellman_module):
    """Test that Bellman weight provides amplification compared to linear p*q for rare events."""
    bellman = bellman_module.bellman_volatility_weight
    q = 0.5
    # Rare event
    p = 0.01

    linear_gain = q * p
    bellman_weight = bellman(p, q)
    bellman_gain = bellman_weight - q

    # Bellman gain should be higher than linear gain for rare events
    assert bellman_gain > linear_gain

def test_calculate_exit_signal_depth_zero(bellman_module):
    """Depth 0 should never exit to allow at least one expansion."""
    exit_signal = bellman_module.calculate_exit_signal
    assert exit_signal(entropy=0.5, cost=1.0, depth=0) is False
    assert exit_signal(entropy=1.0, cost=10.0, depth=0) is False

def test_calculate_exit_signal_cost_influence(bellman_module):
    """Higher cost should lead to earlier exit."""
    exit_signal = bellman_module.calculate_exit_signal
    entropy = 0.5
    depth = 1

    # Very low cost should not exit
    assert exit_signal(entropy, cost=0.0001, depth=depth) is False

    # Very high cost should exit
    assert exit_signal(entropy, cost=1.0, depth=depth) is True

def test_calculate_exit_signal_depth_influence(bellman_module):
    """Increasing depth should eventually trigger an exit for non-zero cost."""
    exit_signal = bellman_module.calculate_exit_signal
    entropy = 0.5
    cost = 0.01

    # At some depth, it must stop because p*log(1/p) goes to 0 as depth increases (p=1/(depth+1))
    exit_found = False
    for depth in range(1, 1000):
        if exit_signal(entropy, cost, depth):
            exit_found = True
            break
    assert exit_found is True

def test_calculate_exit_signal_entropy_influence(bellman_module):
    """Higher entropy should decrease marginal benefit, leading to earlier exit."""
    exit_signal = bellman_module.calculate_exit_signal
    cost = 0.05
    depth = 5

    # Test actual exit signal behavior
    if exit_signal(0.1, cost, depth):
        assert exit_signal(2.0, cost, depth) is True

def test_calculate_exit_signal_zero_cost(bellman_module):
    """If cost is 0, it should never exit."""
    exit_signal = bellman_module.calculate_exit_signal
    # At finite depth, p > 0, so p*log(1/p) > 0, so benefit > 0.
    for depth in range(1, 100):
        assert exit_signal(entropy=0.5, cost=0, depth=depth) is False

# ================================================================================
# TESTS FOR RecursiveIntelligenceCascade & Criterion
# ================================================================================

def test_criterion_initialization(bellman_module):
    """Test Criterion dataclass initialization."""
    Criterion = bellman_module.Criterion
    c = Criterion("test", 0.5, lambda d, c: True)
    assert c.name == "test"
    assert c.weight == 0.5
    assert c.matcher("any", {}) is True

def test_ric_evaluate_sharp(bellman_module):
    """Test _evaluate_sharp method scoring."""
    Criterion = bellman_module.Criterion
    RecursiveIntelligenceCascade = bellman_module.RecursiveIntelligenceCascade

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

def test_ric_prune_sharp(bellman_module):
    """Test _prune_sharp selects top thoughts."""
    Criterion = bellman_module.Criterion
    RecursiveIntelligenceCascade = bellman_module.RecursiveIntelligenceCascade

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

def test_ric_process_flow(bellman_module):
    """Test the process method runs and terminates."""
    Criterion = bellman_module.Criterion
    RecursiveIntelligenceCascade = bellman_module.RecursiveIntelligenceCascade

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

def test_ric_bellman_exit(bellman_module):
    """Test that process exits due to Bellman Exit (high cost)."""
    Criterion = bellman_module.Criterion
    RecursiveIntelligenceCascade = bellman_module.RecursiveIntelligenceCascade

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

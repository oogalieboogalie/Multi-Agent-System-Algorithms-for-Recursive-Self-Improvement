import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. ALGORITHMS
# ==============================================================================

def bellman_volatility_weight(p, q):
    """Sharp Bellman: U(p, q) = sqrt(q^2 + p*log(1/p))"""
    if p <= 0 or p >= 1: return q
    return np.sqrt(q**2 + p * np.log(1/p))

def heuristic_weight(p, q):
    """Standard Heuristic: Linear combination (q * (1-p))"""
    # Common approach: Rare events (low p) are just "weighted" by their rarity linearly
    # or often just ignored if p is too low.
    return q * (1 - p) + q # Simple boost, but linear

def calculate_bellman_gain(entropy, depth):
    """Expected information gain using Bellman"""
    p = 1.0 / (depth + 1)
    return bellman_volatility_weight(p, entropy) - entropy

def calculate_heuristic_gain(entropy, depth):
    """Standard Heuristic Gain (diminishing returns)"""
    # Usually modeled as simple exponential decay or 1/x
    return entropy / (depth + 1)

def plot_signal_amplification(ax):
    """CHART 1: SIGNAL AMPLIFICATION (The "Needle in Haystack" Effect)"""
    # X-axis: Probability (p) - from Rare (0.01) to Common (1.0)
    # Y-axis: Priority Score

    probabilities = np.linspace(0.001, 1.0, 100)
    fixed_quality = 0.5

    bellman_scores = [bellman_volatility_weight(p, fixed_quality) for p in probabilities]
    heuristic_scores = [heuristic_weight(p, fixed_quality) for p in probabilities]

    ax.plot(probabilities, bellman_scores, color='#00ff9d', linewidth=2, label='Sharp Bellman (Logarithmic)')
    ax.plot(probabilities, heuristic_scores, color='#ff0055', linewidth=2, linestyle='--', label='Standard Linear')

    ax.set_title('Signal Amplification (Rare Events)', fontsize=12)
    ax.set_xlabel('Probability (p) [Left is Rarer]', fontsize=10)
    ax.set_ylabel('Priority Score', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend()
    ax.invert_xaxis() # Rare events on the left

    # Annotation
    annotate_y = bellman_scores[5]
    ax.annotate('Massive Boost for\nRare Signals',
                 xy=(0.05, annotate_y),
                 xytext=(0.2, annotate_y + 0.2),
                 arrowprops=dict(facecolor='white', shrink=0.05))

def plot_stopping_problem(ax):
    """CHART 2: THE STOPPING PROBLEM (Bellman Exit)"""
    # X-axis: Depth of Thought
    # Y-axis: Marginal Information Gain

    depths = np.arange(1, 50)
    fixed_entropy = 0.8
    thinking_cost = 0.05

    bellman_gains = [calculate_bellman_gain(fixed_entropy, d) for d in depths]
    heuristic_gains = [calculate_heuristic_gain(fixed_entropy, d) for d in depths]
    cost_line = [thinking_cost] * len(depths)

    ax.plot(depths, bellman_gains, color='#00ff9d', linewidth=2, label='Bellman Expected Gain')
    ax.plot(depths, heuristic_gains, color='#ff0055', linewidth=2, linestyle='--', label='Heuristic Gain')
    ax.plot(depths, cost_line, color='white', linewidth=1, linestyle=':', label='Thinking Cost')

    # Find intersection (Exit Point)
    bellman_exit = next((d for d, g in enumerate(bellman_gains) if g < thinking_cost), None)

    if bellman_exit:
        ax.scatter([bellman_exit], [thinking_cost], color='yellow', s=100, zorder=5)
        ax.annotate(f'Bellman Exit\n(Depth {bellman_exit})',
                     xy=(bellman_exit, thinking_cost),
                     xytext=(bellman_exit + 5, thinking_cost + 0.1),
                     arrowprops=dict(facecolor='yellow', shrink=0.05))

    ax.set_title('The Stopping Problem (Efficiency)', fontsize=12)
    ax.set_xlabel('Cascade Depth', fontsize=10)
    ax.set_ylabel('Marginal Information Gain', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend()

# ==============================================================================
# 2. SIMULATION & DATA GENERATION
# ==============================================================================

def generate_charts():
    print("Generating Bellman Comparison Charts...")

    # Setup
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Sharp Bellman vs. Standard Heuristic', fontsize=16, color='white')

    # --------------------------------------------------------------------------
    # CHART 1: SIGNAL AMPLIFICATION (The "Needle in Haystack" Effect)
    # --------------------------------------------------------------------------
    plot_signal_amplification(ax1)

    # --------------------------------------------------------------------------
    # CHART 2: THE STOPPING PROBLEM (Bellman Exit)
    # --------------------------------------------------------------------------
    plot_stopping_problem(ax2)

    # Save
    output_path = 'bellman_comparison_chart.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to: {os.path.abspath(output_path)}")
    plt.close()

if __name__ == "__main__":
    try:
        generate_charts()
        print("Success! Open 'bellman_comparison_chart.png' to see the difference.")
    except ImportError:
        print("Error: matplotlib is not installed. Run 'pip install matplotlib' first.")

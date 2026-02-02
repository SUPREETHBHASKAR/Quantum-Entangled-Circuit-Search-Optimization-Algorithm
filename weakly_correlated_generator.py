import numpy as np
import pandas as pd

def generate_weakly_correlated_knapsack(n=500, w_min=10, w_max=100, r=10, capacity_ratio=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Generate weights uniformly
    weights = np.random.randint(w_min, w_max + 1, size=n)

    # Generate profits weakly correlated: p_i ∈ [w_i - r, w_i + r]
    profits = np.array([np.random.randint(max(1, w - r), w + r + 1) for w in weights])

    # Capacity as a fraction of total weight
    capacity = int(capacity_ratio * np.sum(weights))

    # Create DataFrame
    df = pd.DataFrame({'profit': profits, 'weight': weights})

    # Save to text file
    with open('500dv_wc_instance3.txt', 'w', newline='') as f:
        f.write(f"{n} {capacity}\n")
        df.to_csv(f, sep=' ', index=False, header=False)

    return df, capacity

# Example usage
df, capacity = generate_weakly_correlated_knapsack(n=750, w_min=10000, w_max=50000, r=2500, capacity_ratio=0.5, seed=100)
print(f"Knapsack Capacity: {capacity}")

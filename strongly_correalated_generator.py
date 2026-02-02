import numpy as np
import pandas as pd

def generate_strongly_correlated_knapsack(n=500, w_min=10, w_max=100, r=100, capacity_ratio=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    weights = np.random.randint(w_min, w_max + 1, size=n)
    profits = weights + r
    capacity = int(capacity_ratio * np.sum(weights))

    df = pd.DataFrame({'profit': profits, 'weight': weights})

    # Write to file: first row = n and capacity, rest = profit weight
    with open('14dv_sc_pedro.txt', 'w', newline='') as f:
        f.write(f"{n} {capacity}\n")
        df.to_csv(f, sep=' ', index=False, header=False)

    return df, capacity

# Example usage:
df, capacity = generate_strongly_correlated_knapsack(n=14, r=50, seed=20)
print(f"Knapsack Capacity: {capacity}")

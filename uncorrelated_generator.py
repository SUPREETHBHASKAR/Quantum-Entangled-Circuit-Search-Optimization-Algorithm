import numpy as np
import pandas as pd

def generate_uncorrelated_knapsack(n=500, w_min=10, w_max=100, p_min=1, p_max=1000, capacity_ratio=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Generate weights independently
    weights = np.random.randint(w_min, w_max + 1, size=n)

    # Generate profits independently (uncorrelated with weights)
    profits = np.random.randint(p_min, p_max + 1, size=n)

    # Capacity as a fraction of total weight
    capacity = int(capacity_ratio * np.sum(weights))

    # Create DataFrame
    df = pd.DataFrame({'profit': profits, 'weight': weights})

    # Save to text file
    with open('1000dv_uc_instance2.txt', 'w', newline='') as f:
        f.write(f"{n} {capacity}\n")
        df.to_csv(f, sep=' ', index=False, header=False)

    return df, capacity

# Example usage
df, capacity = generate_uncorrelated_knapsack(
    n=1000,
    w_min=100000, w_max=100100,
    p_min=1, p_max=1000,
    capacity_ratio=0.5,
    seed=68900
)

print(f"Knapsack Capacity: {capacity}")
print(df.head())

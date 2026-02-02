import numpy as np

# Example profits, weights, and capacity (replace with your actual values)
profits = np.array([
    114, 135, 133, 150, 115, 80, 109, 82, 69, 116, 98, 148
])

weights = np.array([
    64, 85, 83, 100, 65, 30, 59, 32, 19, 66, 48, 98
])
capacity = 374
# Evaluation function
def evaluate_vector(vec):
    total_weight = np.dot(vec, weights)
    total_profit = np.dot(vec, profits)
    violation = max(0, total_weight - capacity)
    return total_profit, violation


# Transformation + evaluation
def transform_and_evaluate(initial_vectors, R, T):
    results = []
    for vec in initial_vectors:
        # Transformation
        transformed = (vec @ R) % 2
        transformed = (transformed + T) % 2

        # Evaluation
        profit, violation = evaluate_vector(transformed)

        results.append({
            "original": vec.tolist(),
            "transformed": transformed.tolist(),
            "profit": profit,
            "violation": violation
        })
    return results


# ==== Example Usage ====

initial_vectors = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
])



# Example R and T (must be consistent with vector length = 12)
R = np.array([
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
])

 # identity matrix
T = np.array([
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
])

results = transform_and_evaluate(initial_vectors, R, T)

for r in results:
    print(r)

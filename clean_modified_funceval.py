import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product
import csv
import os

profits = np.array([
    114, 135, 133, 150, 115, 80, 109, 82, 69, 116, 98, 148
])

weights = np.array([
    64, 85, 83, 100, 65, 30, 59, 32, 19, 66, 48, 98
])
capacity = 374

# GA parameters
N = 50
generations = 3000
trials = 1
z = 3
p = len(profits)
num_vectors = 2 ** z
e = 1  #the number of extra ones
crossover_rate = 0.9
mutation_rate = 0.2


# Base vectors
# base_vectors = [([int(b) for b in format(i, f'0{p}b')]) for i in range(num_vectors)]


# Deb's fitness function
def evaluate_vector(vec):
    total_weight = np.dot(vec, weights)
    total_profit = np.dot(vec, profits)
    violation = max(0, total_weight - capacity)
    return total_profit, violation
def deb_fitness(vec, min_feas_fit):
    profit, violation = evaluate_vector(vec)
    if violation == 0:
        return profit, True, 0
    else:
        return min_feas_fit - violation, False, violation

def generate_RT():
    R = np.zeros((p, p), dtype=int)
    for i in range(p):
        R[i][i] = 1
        extra_ones = random.sample([j for j in range(p) if j != i], min(e, p - 1))
        for j in extra_ones:
            R[i][j] = 1
    T = np.zeros(p, dtype=int)
    ones_count = random.randint(1, p)
    T[random.sample(range(p), ones_count)] = 1
    return R, T


best_fitness_across_trials = []
best_vectors_across_trials = []
best_R_across_trials = []
best_T_across_trials = []

for trial in range(trials):
    population = [generate_RT() for _ in range(N)] #
    best_feas_fitness_ever = -1
    best_feas_vec_ever = None
    best_R_ever = None
    best_T_ever = None
    min_feasible = float("inf")

    #places the hadamard gates in random z positions and this is fixed for the trial
    hadamard_positions = random.sample(range(p), z)  # Choose 'z' random positions for Hadamard gates
    # hadamard_positions = random.Random(42).sample(range(p), z)
    superposed_bits = list(product([0, 1], repeat=z))  # Generate all 2^z combinations
    # base_vectors = [[random.randint(0, 1) for _ in range(p)] for _ in range(num_vectors)]
    # initial_vectors = np.array(base_vectors)
    base_vectors = []
    for bits in superposed_bits:
        vec = [0] * p
        for idx, pos in enumerate(hadamard_positions):
            vec[pos] = bits[idx]
        base_vectors.append(vec)

    initial_vectors = np.array(base_vectors)
    print("initial_vectors =", initial_vectors)
    output_dir = r"C:\Users\SupreethBhaskar\PycharmProjects\circuit_search\unifrom crossover results\12dv\12dv_sc_pedro"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"trial_{trial + 1}.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)


        generation = 0
        while generation < generations:
            # Sample one vector
            sampled_vec = initial_vectors[random.randint(0, num_vectors - 1)] #a vector is sampled from the initial vectors

            # Step 1: Evaluate all transformed vectors to find global min_feasible and best feasible vector
            min_feasible = float('inf')  # Reset min feasible at each generation if desired
            fitness_results = [] #a list of tuples, where each tuple is deb fitness and the r and t pair.
            all_transformed = []  # store (transformed_vector, R, T) for Deb fitness calculation

            #transformation of the sampled vector using  r,t matrices
            for R, T in population:
                transformed = (sampled_vec @ R) % 2 #binary vector
                transformed = (transformed + T) % 2

                profit, violation = evaluate_vector(transformed)
                if violation == 0 and profit < min_feasible:
                    min_feasible = profit
                if violation == 0 and profit > best_feas_fitness_ever:
                    best_feas_fitness_ever = profit
                    best_feas_vec_ever = transformed.copy()
                    best_R_ever = R.copy()
                    best_T_ever = T.copy()

                all_transformed.append((transformed, R, T))
            #at the end of this process,all r,t pairs have transformed the sampled vector.
            #The resulting vector and the associated r,t pair is a tuple stored in the list all_transformed, min feasible fitness is updated
            #best feasible fitness and its vector is updated

            # Step 2: Calculate Deb fitness using global min_feasible
            fitness_results = [] #a list of tuples, where each tuple is deb fitness and the r and t pair.
            for vec, R, T in all_transformed:
                deb_fit, _, _ = deb_fitness(vec, min_feasible)
                fitness_results.append((deb_fit, R, T))
            #here, deb fitness has been assigned to each r,t pair
            # Binary Tournament Selection
            selected = []
            for _ in range(N):
                a, b = random.sample(fitness_results, 2)
                selected.append(a if a[0] > b[0] else b)

            #uniform crossover
            next_gen = []

            for i in range(0, N, 2):
                fit1, R1, T1 = selected[i]
                fit2, R2, T2 = selected[i + 1]

                # ✅ No copy required since selected is not reused.

                if random.random() < crossover_rate:
                    # --- Vectorized Uniform Row-wise Crossover for R ---
                    row_mask = np.random.rand(p) < 0.5
                    R1_rows = R1[row_mask].copy()  # copy rows to avoid inplace issues in swap
                    R1[row_mask] = R2[row_mask]
                    R2[row_mask] = R1_rows

                    # --- Uniform Bitwise Crossover for T ---
                    mask = np.random.randint(0, 2, size=p, dtype=T1.dtype)
                    T1_tmp = (T1 & ~mask) | (T2 & mask)
                    T2_tmp = (T2 & ~mask) | (T1 & mask)
                    T1, T2 = T1_tmp, T2_tmp

                next_gen.append((R1, T1))
                next_gen.append((R2, T2))


            # Mutation
            for idx in range(N):
                if random.random() < mutation_rate:
                    R, T = next_gen[idx]
                    row = random.randint(0, p - 1)
                    non_diag = [i for i in range(p) if i != row]
                    R[row] = np.eye(1, p, row, dtype=int).flatten()
                    R[row][random.choice(non_diag)] = 1

                    flip_mask = np.random.rand(p) < (1 / p)
                    T = (T ^ flip_mask.astype(int)) % 2
                    next_gen[idx] = (R, T)


            # Step 3: Evaluate new generation on the same sampled vector to update min_feasible & best feasible
            all_transformed_new = [] # store (transformed_vector, R, T) for Deb fitness calculation
            for R, T in next_gen:
                transformed = (sampled_vec @ R) % 2
                transformed = (transformed + T) % 2

                profit, violation = evaluate_vector(transformed)
                if violation == 0 and profit < min_feasible:
                    min_feasible = profit
                if violation == 0 and profit > best_feas_fitness_ever:
                    best_feas_fitness_ever = profit
                    best_feas_vec_ever = transformed.copy()
                    best_R_ever = R.copy()
                    best_T_ever = T.copy()

                all_transformed_new.append((transformed, R, T))

            # Step 4: Calculate Deb fitness for new generation using updated min_feasible
            new_results = []  #a list of tuples, where each tuple is deb fiitness and the r and t pair.
            for vec, R, T in all_transformed_new:
                deb_fit, _, _ = deb_fitness(vec, min_feasible)
                new_results.append((deb_fit, R, T))

            # Elitism: keep top N from old + new
            combined = fitness_results + new_results
            combined.sort(reverse=True, key=lambda x: x[0])
            population = [(R, T) for _, R, T in combined[:N]]

            if best_feas_vec_ever is not None:
                writer.writerow([
                    best_feas_fitness_ever,
                    best_feas_vec_ever.tolist(),
                    best_R_ever.tolist(),
                    best_T_ever.tolist()
                ])
            else:
                writer.writerow([0, [], [], []])

            generation += 1

    # Store across trials
    best_fitness_across_trials.append(best_feas_fitness_ever if best_feas_vec_ever is not None else 0)
    best_vectors_across_trials.append(best_feas_vec_ever)
    best_R_across_trials.append(best_R_ever)
    best_T_across_trials.append(best_T_ever)

    # Print summary
    print(f"Trial {trial + 1}/{trials}: Best Feasible Fitness = {best_fitness_across_trials[-1]}")
    print(f"Vector: {best_feas_vec_ever}")
    print(f"Best R:\n{best_R_ever}")
    print(f"Best T:\n{best_T_ever}")

# Assuming best_fitness_across_trials is a list or array of numbers
best = np.array(best_fitness_across_trials)

mean_fitness = np.mean(best)
median_fitness = np.median(best)
max_fitness = np.max(best)
min_fitness = np.min(best)

print(f"Mean Feasible Fitness: {mean_fitness:.2f}")
print(f"Median Feasible Fitness: {median_fitness:.2f}")
print(f"Maximum Feasible Fitness: {max_fitness}")
print(f"Minimum Feasible Fitness: {min_fitness}")

# Plotting boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(best_fitness_across_trials, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue'))
plt.title('Best Feasible Fitness Across 30 Trials')
plt.ylabel('Feasible Fitness')
plt.ylim(99700, 100000)  # Set y-axis limits
plt.grid(True)
plt.show()

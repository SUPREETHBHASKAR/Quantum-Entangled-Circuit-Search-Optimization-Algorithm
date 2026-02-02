import numpy as np


def convert_to_numpy_string(matrix_str):
    rows = matrix_str.strip().split("\n")
    matrix = []
    for row in rows:
        clean_row = row.replace("[", "").replace("]", "").strip()
        numbers = [int(x) for x in clean_row.split()]
        matrix.append(numbers)

    arr = np.array(matrix)

    formatted = "np.array([\n"
    for row in arr:
        formatted += "    " + str(list(row)) + ",\n"
    formatted = formatted.rstrip(",\n") + "\n])"

    return formatted


# Example: multiple matrices
matrices = [
    """[[0 0 0 0 0 0 0 0 0 0 0 0]
 [1 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 0 0 0]
 [1 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 1 0]
 [1 0 0 0 0 0 0 0 0 0 1 0]
 [0 1 0 0 0 0 0 0 0 0 1 0]
 [1 1 0 0 0 0 0 0 0 0 1 0]]""",

    """[[1 0 1 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 1 0 0]
 [1 0 1 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 1 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 1]
 [0 0 0 0 0 1 0 0 0 0 1 0]
 [0 0 0 0 0 0 1 0 0 0 1 0]
 [0 0 0 0 0 0 0 1 0 0 1 0]
 [0 1 0 0 0 0 0 0 1 0 0 0]
 [0 1 0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 1 0 0 1 0]
 [0 0 0 0 0 0 0 0 0 0 1 1]]""",

"""[1 1 0 0 0 1 1 1 1 1 0 0]""",

]

# Convert and print each
for i, matrix_str in enumerate(matrices, start=1):
    result = convert_to_numpy_string(matrix_str)
    print(f"Matrix {i}:\n{result}\n")

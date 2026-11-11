import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    size = len(b)
    x = np.zeros(size, dtype=float)

    for _ in range(n):
        x_new = np.zeros(size, dtype=float)

        for i in range(size):
            sum_others = sum(A[i][j] * x[j] for j in range(size) if j != i)
            x_new[i] = (b[i] - sum_others) / A[i][i]

        x = x_new

    return np.round(x, 4).tolist()
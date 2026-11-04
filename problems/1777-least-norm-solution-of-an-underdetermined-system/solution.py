import numpy as np

def min_norm_solution(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    return A.T @ np.linalg.solve(A @ A.T, b)


def min_norm_via_kkt(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    m, n = A.shape

    # KKT system
    K = np.block([
        [2 * np.eye(n), A.T],
        [A, np.zeros((m, m))]
    ])

    rhs = np.concatenate([np.zeros(n), b])

    solution = np.linalg.solve(K, rhs)

    x = solution[:n]
    lam = solution[n:]

    return x, lam


def min_norm_from_particular(A, x_p):
    A = np.asarray(A, dtype=float)
    x_p = np.asarray(x_p, dtype=float)

    # Remove the row-space component's orthogonal complement
    return A.T @ np.linalg.solve(A @ A.T, A @ x_p)
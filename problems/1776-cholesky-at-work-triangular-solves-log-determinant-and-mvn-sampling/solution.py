import numpy as np

def forward_sub(L, b):
    n = len(b)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def back_sub(U, y):
    n = len(y)
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x


def chol_solve(A, b):
    L = np.linalg.cholesky(A)

    y = forward_sub(L, b)
    x = back_sub(L.T, y)

    return x


def logdet_chol(A):
    L = np.linalg.cholesky(A)

    return float(2 * np.sum(np.log(np.diag(L))))


def mvn_sample(mu, Sigma, eps):
    L = np.linalg.cholesky(Sigma)

    return eps @ L.T + mu
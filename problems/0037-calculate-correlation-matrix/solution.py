import numpy as np

def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        Y = X

    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    numerator = X.T @ Y

    denominator = np.sqrt(
        np.sum(X**2, axis=0)[:, None] *
        np.sum(Y**2, axis=0)[None, :]
    )

    return numerator / denominator
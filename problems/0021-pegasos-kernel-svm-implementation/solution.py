import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    n = len(data)
    alphas = np.zeros(n, dtype=float)
    bias = 0.0

    # Compute kernel matrix
    if kernel == 'linear':
        K = data @ data.T
    elif kernel == 'rbf':
        sq_dist = np.sum(data**2, axis=1)[:, None] + np.sum(data**2, axis=1)[None, :] - 2 * (data @ data.T)
        K = np.exp(-sq_dist / (2 * sigma**2))
    else:
        raise ValueError("Unsupported kernel")

    for t in range(1, iterations + 1):
        eta = 1 / (lambda_val * t)

        for i in range(n):
            g = np.sum(alphas * labels * K[:, i]) + bias

            if labels[i] * g < 1:
                alphas[i] += eta * (1 - lambda_val * alphas[i])
                bias += eta * labels[i]
            else:
                alphas[i] -= eta * lambda_val * alphas[i]

    return alphas.tolist(), float(bias)
import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    # Standardize the data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    standardized = (data - mean) / std

    # Compute covariance matrix
    cov_matrix = np.cov(standardized, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx[:k]]

    # Fix eigenvector signs
    for i in range(k):
        for j in range(components.shape[0]):
            if abs(components[j, i]) > 1e-10:
                if components[j, i] < 0:
                    components[:, i] *= -1
                break

    return np.round(components, 4)
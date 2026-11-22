import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n = X.shape[0]

    subset_size = n if replacements else n // 2

    subsets = []

    for _ in range(n_subsets):
        idx = np.random.choice(n, subset_size, replace=replacements)

        X_subset = X[idx].tolist()
        y_subset = y[idx].tolist()

        subsets.append((X_subset, y_subset))

    return subsets
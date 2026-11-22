import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = X.shape

    features = [np.ones(n_samples)]

    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n_features), d):
            feature = np.prod(X[:, comb], axis=1)
            features.append(feature)

    result = np.column_stack(features)

    return np.sort(result, axis=1)
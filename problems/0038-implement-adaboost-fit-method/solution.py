import numpy as np
import math

def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, 1 / n_samples)
    clfs = []

    for _ in range(n_clf):
        best_error = float('inf')
        best_stump = {}

        for j in range(n_features):
            thresholds = np.unique(X[:, j])

            for t in thresholds:
                p = 1
                predictions = np.ones(n_samples)
                predictions[X[:, j] < t] = -1

                error = np.sum(w[predictions != y])

                if error > 0.5:
                    p = -1
                    error = 1 - error

                if error < best_error:
                    best_error = error
                    best_stump = {
                        'polarity': p,
                        'threshold': t,
                        'feature_index': j
                    }

        alpha = 0.5 * math.log(
            (1 - best_error) / (best_error + 1e-10)
        )

        predictions = np.ones(n_samples)
        predictions[X[:, best_stump['feature_index']] < best_stump['threshold']] = -1
        predictions *= best_stump['polarity']

        w *= np.exp(-alpha * y * predictions)
        w /= np.sum(w)

        best_stump['alpha'] = alpha
        clfs.append(best_stump)

    return clfs
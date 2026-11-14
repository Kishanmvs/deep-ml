import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1

    folds = []
    start = 0

    for size in fold_sizes:
        folds.append(indices[start:start + size].tolist())
        start += size

    result = []

    for i in range(k):
        test_indices = folds[i]
        train_indices = []

        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])

        result.append((train_indices, test_indices))

    return result
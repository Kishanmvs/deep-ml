import numpy as np

def cholesky_decomposition(A):
    try:
        A = np.array(A, dtype=float)

        # Check if matrix is valid and non-empty
        if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
            return -1

        # Check for invalid values
        if not np.all(np.isfinite(A)):
            return -1

        # Check if symmetric
        if not np.allclose(A, A.T):
            return -1

        n = A.shape[0]
        L = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))

                if i == j:
                    value = A[i][i] - s

                    if value <= 0:
                        return -1

                    L[i][j] = np.sqrt(value)
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]

        return L.tolist()

    except (ValueError, TypeError, OverflowError):
        return -1
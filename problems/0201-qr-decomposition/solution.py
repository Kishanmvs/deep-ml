import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using the Gram-Schmidt process.

    Args:
        A: An m x n matrix represented as a list of lists

    Returns:
        Tuple of (Q, R), or (-1, -1) if decomposition is not possible
    """
    try:
        A = np.array(A, dtype=float)

        if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            return -1, -1

        if not np.all(np.isfinite(A)):
            return -1, -1

        m, n = A.shape

        if m < n:
            return -1, -1

        Q = np.zeros((m, n))
        R = np.zeros((n, n))

        for j in range(n):
            v = A[:, j].copy()

            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v -= R[i, j] * Q[:, i]

            R[j, j] = np.linalg.norm(v)

            if np.isclose(R[j, j], 0):
                return -1, -1

            Q[:, j] = v / R[j, j]

        return Q.tolist(), R.tolist()

    except (ValueError, TypeError, OverflowError):
        return -1, -1
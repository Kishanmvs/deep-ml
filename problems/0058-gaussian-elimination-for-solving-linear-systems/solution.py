def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # Forward elimination with partial pivoting
    for i in range(n):
        # Find the row with the largest absolute pivot
        max_row = i + np.argmax(np.abs(A[i:, i]))

        # Swap rows in A and b
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Check for a singular matrix
        if abs(A[i, i]) < 1e-12:
            raise ValueError("The system has no unique solution.")

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x.tolist()
import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    # Step 1: Calculate A.T @ A
    ATA = A.T @ A

    # Step 2: Calculate eigenvalues analytically
    trace = np.trace(ATA)
    det = np.linalg.det(ATA)

    discriminant = np.sqrt(max(0, trace**2 - 4 * det))

    lambda1 = (trace + discriminant) / 2
    lambda2 = (trace - discriminant) / 2

    # Step 3: Singular values
    s = np.sqrt(np.maximum([lambda1, lambda2], 0))

    # Step 4: Calculate eigenvectors
    def eigenvector(M, eigenvalue):
        a = M[0, 0] - eigenvalue
        b = M[0, 1]
        c = M[1, 1] - eigenvalue

        if abs(b) > 1e-10:
            v = np.array([b, -a])
        elif abs(a) < abs(c):
            v = np.array([1.0, 0.0])
        else:
            v = np.array([0.0, 1.0])

        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else np.array([1.0, 0.0])

    v1 = eigenvector(ATA, lambda1)
    v2 = np.array([-v1[1], v1[0]])

    V = np.column_stack((v1, v2)).T

    # Step 5: Calculate U
    U = np.zeros((2, 2))

    if s[0] > 1e-10:
        U[:, 0] = A @ v1 / s[0]

    if s[1] > 1e-10:
        U[:, 1] = A @ v2 / s[1]
    else:
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])

    return U, s, V
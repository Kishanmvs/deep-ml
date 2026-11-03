import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """

    # Initial guess
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)

    # Initial residual
    r = b - np.dot(A, x)

    # Initial search direction
    p = r.copy()

    # Squared residual norm
    rs_old = np.dot(r, r)

    for _ in range(n):

        # Check convergence
        if np.sqrt(rs_old) < tol:
            break

        # Matrix-vector multiplication
        Ap = np.dot(A, p)

        # Step size
        alpha = rs_old / np.dot(p, Ap)

        # Update solution
        x = x + alpha * p

        # Update residual
        r = r - alpha * Ap

        # New squared residual norm
        rs_new = np.dot(r, r)

        # Update search direction
        beta = rs_new / rs_old
        p = r + beta * p

        rs_old = rs_new

    return x.tolist()
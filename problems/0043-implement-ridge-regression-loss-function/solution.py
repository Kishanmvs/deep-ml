import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # Calculate predictions
    y_pred = X @ w

    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # L2 regularization penalty
    penalty = alpha * np.sum(w ** 2)

    # Total Ridge Loss
    return float(mse + penalty)

import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):

    weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []

    for epoch in range(epochs):

        # Forward propagation
        z = np.dot(features, weights) + bias
        predictions = 1 / (1 + np.exp(-z))

        # Calculate MSE
        errors = predictions - labels
        mse = np.mean(errors ** 2)
        mse_values.append(mse)

        # Backpropagation
        dz = (2 / len(labels)) * errors * predictions * (1 - predictions)

        dw = np.dot(features.T, dz)
        db = np.sum(dz)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return (
        np.round(weights, 4),
        round(float(bias), 4),
        [round(float(m), 4) for m in mse_values]
    )

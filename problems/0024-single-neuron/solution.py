import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):

    probabilities = []

    for x in features:
        z = sum(xi * wi for xi, wi in zip(x, weights)) + bias

        probability = 1 / (1 + math.exp(-z))

        probabilities.append(probability)

    mse = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)

    probabilities = [round(p, 4) for p in probabilities]
    mse = round(mse, 4)

    return probabilities, mse

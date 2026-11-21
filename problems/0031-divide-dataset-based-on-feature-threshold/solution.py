import numpy as np

def divide_on_feature(X, feature_i, threshold):
    condition = X[:, feature_i] >= threshold

    X1 = X[condition]
    X2 = X[~condition]

    return [X1, X2]
import numpy as np

def linear_predict(feature: np.ndarray, weights: np.ndarray, bias: float) -> float:
    return np.dot(feature, weights) + bias

def linear_loss(features: np.ndarray, label: float, weights: np.ndarray, bias: float) -> float:
    return linear_predict(features, weights, bias) - label

def linear_cost(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float) -> float:
    return np.sum(linear_loss(features, labels, weights, bias) ** 2) / (2 * len(features))

def linear_cost_with_reg(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float, lambda_: float) -> float:
    return linear_cost(features, labels, weights, bias) + (lambda_ / 2 * len(features)) * np.sum(weights ** 2)
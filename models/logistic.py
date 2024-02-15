import numpy as np

def sigmoid(x: np.ndarray) -> any:
    return 1 / (1 + np.exp(-x))

def logistic_predict(features: np.ndarray, weights: np.ndarray, bias: float) -> any:
    return sigmoid(np.dot(features, weights) + bias)

def logistic_loss(features: np.ndarray, label: float, weights: np.ndarray, bias: float) -> float:
    return -label * np.log(logistic_predict(features, weights, bias)) - (1 - label) * np.log(1 - logistic_predict(features, weights, bias))

def logistic_cost(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float) -> float:
    return np.mean(logistic_loss(features, labels, weights, bias) ** 2)

def logistic_cost_with_reg(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float, lambda_: float) -> float:
    return logistic_cost(features, labels, weights, bias) + (lambda_ / 2 * len(features)) * np.sum(weights ** 2)
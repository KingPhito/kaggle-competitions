from dataclasses import dataclass
from enum import Enum
import numpy as np
from .linear import *
from .logistic import *


class ModelType(Enum):
    LINEAR_REGRESSION = 1
    LOGISTIC_REGRESSION = 2

@dataclass
class ModelParams:
    weights: np.ndarray
    bias: float
    features: np.ndarray
    labels: np.ndarray
    type: ModelType
    regularize: bool = False
    lambda_: float = 0.0


class ScratchModel:
    def __init__(self, params: ModelParams):
        self.params = params

    def predict(self, features: np.ndarray) -> float:
        match self.params.type:
            case ModelType.LINEAR_REGRESSION:
                return linear_predict(features, self.params.weights, self.params.bias)
            case ModelType.LOGISTIC_REGRESSION:
                return logistic_predict(features, self.params.weights, self.params.bias)
        return 0.

    def loss(self, features: np.ndarray, label: float) -> float:
        match self.params.type:
            case ModelType.LINEAR_REGRESSION:
                return linear_loss(features, label, self.params.weights, self.params.bias)
            case ModelType.LOGISTIC_REGRESSION:
                return logistic_loss(features, label, self.params.weights, self.params.bias)
        return 0.
    
    def cost(self, features: np.ndarray, labels: np.ndarray) -> float:
        match self.params.type:
            case ModelType.LINEAR_REGRESSION:
                if self.params.regularize:
                    return linear_cost_with_reg(features, labels, self.params.weights, self.params.bias, self.params.lambda_)
                return linear_cost(features, labels, self.params.weights, self.params.bias)
            case ModelType.LOGISTIC_REGRESSION:
                if self.params.regularize:
                    return logistic_cost_with_reg(features, labels, self.params.weights, self.params.bias, self.params.lambda_)
                return logistic_cost(features, labels, self.params.weights, self.params.bias)
        return 0.
    
    def update_weights(self):
        rows, col = self.params.features.shape
        d_cost_w = np.zeros(col)
        d_cost_b = 0.
        for i in range(rows):
            err = 0.
            match self.params.type:
                case ModelType.LINEAR_REGRESSION:
                    err = linear_loss(self.params.features[i], self.params.labels[i], self.params.weights, self.params.bias)
                case ModelType.LOGISTIC_REGRESSION:
                    err = logistic_loss(self.params.features[i], self.params.labels[i], self.params.weights, self.params.bias)
            for j in range(col):
                d_cost_w[j] += err * self.params.features[i][j]
            d_cost_b += err
        d_cost_w /= rows
        d_cost_b /= rows
        if self.params.regularize:
            d_cost_w += (self.params.lambda_ / rows) * self.params.weights
        return d_cost_w, d_cost_b
    
    def train(self, epochs: int, learning_rate: float):
        cost_history = []
        for i in range(epochs):
            d_cost_w, d_cost_b = self.update_weights()
            self.params.weights -= (learning_rate * d_cost_w)
            self.params.bias -= (learning_rate * d_cost_b)
            cost_history.append(self.cost(self.params.features, self.params.labels))
            print(f'Epoch: {i}, Cost: {cost_history[-1]}')
        return cost_history


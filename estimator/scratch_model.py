from dataclasses import dataclass
import numpy as np

@dataclass
class ModelParams:
    weights: np.ndarray
    bias: float
    features: np.ndarray
    labels: np.ndarray


class PriceEstimator:
    def __init__(self, params: ModelParams):
        self.params = params

    def predict(self, feature: np.ndarray) -> float:
        return np.dot(feature, self.params.weights) + self.params.bias

    def cost(self, features: np.ndarray, labels: np.ndarray) -> float:
        row, _ = features.shape
        total_cost = 0
        for i in range(row):
            total_cost += (self.predict(features[i]) - labels[i]) ** 2
        return total_cost / 2*row 
    
    def update_weights(self):
        row, col = self.params.features.shape
        d_cost_w = np.zeros(col)
        d_cost_b = 0.
        for i in range(row):
            err = self.predict(self.params.features[i]) - self.params.labels[i]
            for j in range(col):
                d_cost_w[j] += err * self.params.features[i][j]
            d_cost_b += err
        return d_cost_w / row, d_cost_b / row
    
    def train(self, epochs: int, learning_rate: float):
        cost_history = []
        for i in range(epochs):
            d_cost_w, d_cost_b = self.update_weights()
            self.params.weights -= (learning_rate * d_cost_w)
            self.params.bias -= (learning_rate * d_cost_b)
            cost_history.append(self.cost(self.params.features, self.params.labels))
            print(f'Epoch: {i}, Cost: {cost_history[-1]}')
        return cost_history


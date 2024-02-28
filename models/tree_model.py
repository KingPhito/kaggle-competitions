from dataclasses import dataclass
from enum import Enum
import numpy as np

class ModelType(Enum):
    DECISION_TREE = 1
    RANDOM_FOREST = 2
    GRADIENT_BOOSTING = 3

@dataclass
class ModelParams:
    features: np.ndarray
    labels: np.ndarray
    type: ModelType
    max_depth: int = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: int = None
    n_estimators: int = 100
    learning_rate: float = 0.1
    lambda_: float = 0.0

class TreeModel:
    def __init__(self, params: ModelParams):
        self.params = params

    def get_purity(self, labels: np.ndarray) -> float:
        return np.max(np.bincount(labels)) / labels.size
    
    def get_entropy(self, labels: np.ndarray) -> float:
        _, counts = np.unique(labels, return_counts=True)
        prob = counts / labels.size
        return -np.sum(prob * np.log(prob))
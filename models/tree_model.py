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

@dataclass
class TreeNode:
    feature: int
    threshold: float
    left: 'TreeNode'
    right: 'TreeNode'
    value: float

class TreeModel:
    def __init__(self, params: ModelParams):
        self.params = params

    def get_purity(self, labels: np.ndarray) -> float:
        return np.max(np.bincount(labels)) / labels.size
    
    def get_entropy(self, labels: np.ndarray) -> float:
        purity = self.get_purity(labels)
        if purity == 0 or purity == 1:
            return 0
        return -purity * np.log2(purity) - (1 - purity) * np.log2(1 - purity)
    
    def split_data(self, feature: int, threshold: float, data: np.ndarray) -> np.ndarray:
        left = data[data[:, feature] <= threshold]
        right = data[data[:, feature] > threshold]
        return left, right
    
    def get_gain(self, feature: int, threshold: float, data: np.ndarray, parent_impurity: float) -> float:
        left, right = self.split_data(feature, threshold, data)
        left_impurity = self.get_entropy(left[:, -1])
        right_impurity = self.get_entropy(right[:, -1])
        n = data.shape[0]
        return parent_impurity - (left.shape[0] / n * left_impurity + right.shape[0] / n * right_impurity)
        
    def find_best_split(self, data: np.ndarray) -> tuple:
        best_gain = 0
        best_feature = 0
        best_threshold = 0
        impurity = self.get_entropy(data[:, -1])
        for feature in range(data.shape[1] - 1):
            thresholds = np.unique(data[:, feature])
            for threshold in thresholds:
                gain = self.get_gain(feature, threshold, data, impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, data: np.ndarray, depth: int = 0) -> TreeNode:
        labels = data[:, -1]
        if depth == self.params.max_depth or np.unique(labels).size == 1:
            return TreeNode(None, None, None, None, np.mean(labels))
        feature, threshold, _ = self.find_best_split(data)
        left, right = self.split_data(feature, threshold, data)
        left_tree = self.build_tree(left, depth + 1)
        right_tree = self.build_tree(right, depth + 1)
        return TreeNode(feature, threshold, left_tree, right_tree, np.mean(labels))
    
    def build_forest(self) -> list:
        trees = []
        match self.params.type:
            case ModelType.DECISION_TREE:
                tree = self.build_tree(np.column_stack((self.params.features, self.params.labels)))
                trees.append(tree)
                return trees
            case ModelType.RANDOM_FOREST:
                for _ in range(self.params.n_estimators):
                    indices = np.random.choice(self.params.features.shape[0], self.params.features.shape[0], replace=True)
                    tree = self.build_tree(np.column_stack((self.params.features[indices], self.params.labels[indices])))
                    trees.append(tree)
                    return trees
            case ModelType.GRADIENT_BOOSTING:
                pass
        
    def predict(self, features: np.ndarray) -> float:
        forest = self.build_forest()
        predictions = []
        for tree in forest:
            node = tree
            while node.left and node.right:
                if features[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return np.mean(predictions)
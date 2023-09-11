import numpy as np
import math

from src.CustomDecisionTree import CustomDecisionTree, DecisionTreeNode

class CustomRandomForestClassifier:

    def __init__(self, n_trees, max_depth=10, min_samples_split=2, min_impurity=1e-7, min_gain=1e-7):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.min_gain = min_gain
        self.trees = []

    def fit(self, X, y):
        self._grow_forest(X, y)

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        return np.round(np.mean(predictions, axis=0))

    def predict_proba(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict_proba(X))
        return np.mean(predictions, axis=0)


    def _grow_tree(self, X, y):

        X = np.array(X)
        y = np.array(y)

        tree_stump = CustomDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_impurity=self.min_impurity, min_gain=self.min_gain)
        tree_stump.fit(X, y)
        self.trees.append(tree_stump)

    def _grow_forest(self, X, y):
        for _ in range(self.n_trees):
            Xi, yi = CustomRandomForestClassifier.bootstrap_sample(X, y, X.shape[0])
            Xi = CustomRandomForestClassifier.sample_features(Xi, int(math.sqrt(X.shape[1])))
            self._grow_tree(Xi, yi)

    @staticmethod
    def bootstrap_sample(X, y, n_samples):
        sample_x = []
        sample_y = []
        for _ in range(n_samples):
            index = np.random.randint(0, len(X))
            sample_x.append(X[index])
            sample_y.append(y[index])
        return np.array(sample_x), np.array(sample_y)

    @staticmethod
    def sample_features(X, n_features):
        feature_indices = set(range(X.shape[1]))
        feature_indices = np.random.choice(list(feature_indices), n_features, replace=False)
        return X[:, feature_indices]




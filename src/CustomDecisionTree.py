import numpy as np
import math

class DecisionTreeNode:

    def __init__(self, feature_index, threshold, left, right, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return not self.left and not self.right


class CustomDecisionTree:

    def __init__(self, root, max_depth, min_samples_split=2, min_impurity=1e-7):
        self.root = root
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity

    def get_splitting_criterion(self, X, y):
        pass

    def get_best_split(self, X, y):
        pass

    def split(self, X, y, depth=0):
        pass

    def fit(self, X, y):
        pass

    @staticmethod
    def calculate_entropy(y):
        pos = sum(y)
        neg = len(y) - pos

        if pos == 0 or neg == 0:
            return 0

        return -pos/len(y) * math.log(pos/len(y), 2) - neg/len(y) * math.log(neg/len(y), 2)


    @staticmethod
    def calculate_gini_impurity(y):

        pos = sum(y)
        neg = len(y) - pos

        if pos == 0 or neg == 0:
            return 0

        return 1 - (pos/len(y))**2 - (neg/len(y))**2

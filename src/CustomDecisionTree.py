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

    def __init__(self, max_depth=10, min_samples_split=2, min_impurity=1e-7):
        self.root = DecisionTreeNode(None, None, None, None)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity

    def _grow_tree(self, X, y, node, depth):

        #reshape X to have dimensions 4, 100
        X = np.array(X)
        y = np.array(y)

        if depth >= self.max_depth:
            return

        feature, threshold, gain = self.get_best_split(X, y)

        if gain < self.min_impurity:
            return

        left_X, left_y, right_X, right_y = self.split(X, y, threshold, feature)

        node.feature_index = feature
        node.threshold = threshold
        node.left = DecisionTreeNode(None, None, None, None)
        node.right = DecisionTreeNode(None, None, None, None)

        self._grow_tree(left_X, left_y, node.left, depth+1)
        self._grow_tree(right_X, right_y, node.right, depth+1)

    def get_splitting_criterion(self, X, y, feature_index, parent_impurity):

        # Initialize variables to keep track of the best split
        best_gain = 0
        best_threshold = None

        for val in set(X.T[feature_index]):

            left_x, left_y, right_x, right_y = self.split(X, y, val, feature_index)
            entropy_left = self.calculate_entropy(left_y)
            entropy_right = self.calculate_entropy(right_y)

            # Calculate the information gain from this split
            info_gain = parent_impurity - (len(left_y)/len(y) * entropy_left + len(right_y)/len(y) * entropy_right)
            if info_gain > best_gain:
                best_threshold = val
                best_gain = max(info_gain, best_gain)

        return best_threshold, best_gain


    def get_best_split(self, X, y):

        parent_impurity = self.calculate_entropy(y)
        best_gain = 0
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            threshold, gain = self.get_splitting_criterion(X, y, feature_index, parent_impurity)
            if gain > self.min_impurity and gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold

        return best_feature_index, best_threshold, best_gain

    def split(self, X, y, threshold, feature_index):

        left_X = []
        left_y = []
        right_X = []
        right_y = []

        for i in range(X.shape[0]):
            if X.T[feature_index][i] < threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return left_X, left_y, right_X, right_y

    def fit(self, X, y):
        self._grow_tree(X, y, self.root, 0)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):

        if node.is_leaf_node():
            return node.value

        if x[node.feature_index] < node.threshold:
            return self._predict(x, node.left)

        return self._predict(x, node.right)

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

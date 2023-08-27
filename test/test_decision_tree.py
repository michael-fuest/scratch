import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.CustomDecisionTree import CustomDecisionTree


@pytest.fixture
def input_data() -> (np.array, np.array):
    x = np.array([np.random.randint(0, 10, size=4) for _ in range(100)])
    y = np.array([np.random.randint(0, 2) for _ in range(100)])
    return x, y

@pytest.fixture
def sklearn_decision_tree(input_data) -> DecisionTreeClassifier:
    X, y = input_data
    return DecisionTreeClassifier(min_samples_split=2, max_depth=10, min_impurity_decrease=1e-7).fit(X, y)

@pytest.fixture
def custom_decision_tree(input_data) ->  CustomDecisionTree:
    tree = CustomDecisionTree()
    tree.fit(input_data[0], input_data[1])
    return tree

def test_decision_tree_fit(
        custom_decision_tree,
        sklearn_decision_tree,
    ) -> None:
    """
    Tests the fit method of the LinearRegression class.
    :return: None
    """
    # Assert
    x_test = np.array([np.random.randint(0, 10, size=4) for _ in range(100)])
    custom_pred = custom_decision_tree.predict(x_test)
    sklearn_pred = sklearn_decision_tree.predict(x_test)
    same_prediction_percentage = np.sum(custom_pred == sklearn_pred) / len(custom_pred)
    assert same_prediction_percentage >= 0.89



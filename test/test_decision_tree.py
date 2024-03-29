import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.CustomDecisionTree import CustomDecisionTree


@pytest.fixture
def input_data() -> (np.array, np.array):
    x = np.array([np.random.randint(0, 10, size=8) for _ in range(1000)])
    y = np.array([np.random.randint(0, 2) for _ in range(1000)])
    return x, y

@pytest.fixture
def sklearn_decision_tree(input_data) -> DecisionTreeClassifier:
    X, y = input_data
    return DecisionTreeClassifier(min_samples_split=2, max_depth=5, min_impurity_decrease=1e-7).fit(X, y)

@pytest.fixture
def custom_decision_tree(input_data) ->  CustomDecisionTree:
    tree = CustomDecisionTree(max_depth=5, min_samples_split=2, min_impurity=1e-7, min_gain=1e-7)
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
    x_test = np.array([np.random.randint(0, 10, size=8) for _ in range(100)])
    custom_pred = custom_decision_tree.predict(x_test)
    custom_pred_proba = custom_decision_tree.predict_proba(x_test)
    sklearn_pred = sklearn_decision_tree.predict(x_test)
    sklearn_pred_proba = sklearn_decision_tree.predict_proba(x_test)
    pred_proba_diff = np.sum(np.abs(custom_pred_proba - sklearn_pred_proba)) / len(custom_pred_proba)
    same_prediction_percentage = np.sum(custom_pred == sklearn_pred) / len(custom_pred)
    assert same_prediction_percentage >= 0.89
    assert pred_proba_diff <= 0.05



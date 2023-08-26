import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier


from utils.classes import FileManager
from src.CustomDecisionTree import CustomDecisionTree


@pytest.fixture
def input_data() -> (np.array, np.array):
    x = np.array([np.random.randint(0, 10, size=4) for _ in range(100)])
    y = np.array([np.random.randint(0, 2) for _ in range(100)])
    return x, y

@pytest.fixture
def sklearn_decision_tree(input_data) -> DecisionTreeClassifier:
    X, y = input_data
    return DecisionTreeClassifier().fit(X, y)

@pytest.fixture
def custom_decision_tree(input_data) ->  CustomDecisionTree:
    return CustomDecisionTree().fit(input_data[0], input_data[1])

def test_decision_tree_fit(
        custom_decision_tree,
        sklearn_decision_tree,
    ) -> None:
    """
    Tests the fit method of the LinearRegression class.
    :return: None
    """
    # Assert
    assert custom_decision_tree.predict([1,2,3]) == sklearn_decision_tree.predict([1,2,3])


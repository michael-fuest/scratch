import pytest
import numpy as np

from sklearn.linear_model import LogisticRegression
from src.CustomLogisticRegression import CustomLogisticRegression


@pytest.fixture
def input_data() -> (np.array, np.array):
    x = np.array([np.random.randint(0, 10, size=8) for _ in range(1000)])
    y = np.array([np.random.randint(0, 2) for _ in range(1000)])
    return x, y
@pytest.fixture
def sklearn_logistic_regression(input_data) -> LogisticRegression:
    return LogisticRegression().fit(input_data[0], input_data[1])

@pytest.fixture
def custom_logistic_regression(input_data) -> CustomLogisticRegression:
    log_reg = CustomLogisticRegression()
    log_reg.fit(input_data[0], input_data[1])
    return log_reg

def test_logistic_regression_fit(
        custom_logistic_regression,
        sklearn_logistic_regression,
    ) -> None:
    """
    Tests the fit method of the LogisticRegression class.
    :return: None
    """

    x_test = np.array([np.random.randint(0, 10, size=8) for _ in range(100)])
    custom_pred = custom_logistic_regression.predict(x_test)
    custom_pred_proba = custom_logistic_regression.predict_proba(x_test)
    sklearn_pred = sklearn_logistic_regression.predict(x_test)
    sklearn_pred_proba = sklearn_logistic_regression.predict_proba(x_test)
    pred_proba_diff = np.sum(np.abs(custom_pred_proba - sklearn_pred_proba)) / len(custom_pred_proba)
    same_prediction_percentage = np.sum(custom_pred == sklearn_pred) / len(custom_pred)
    assert same_prediction_percentage >= 0.89
    assert pred_proba_diff <= 0.05


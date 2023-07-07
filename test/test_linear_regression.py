import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS


import sys

sys.path.append('/Users/michaelfuest/scratch/')
from utils.classes import FileManager
from src.models import CustomLinearRegression

@pytest.fixture
def file_manager():
    return FileManager()

@pytest.fixture
def housing_data_X(file_manager):
    return np.array(pd.read_csv(file_manager.test_data_input, usecols=['bedrooms', 'bathrooms']))

@pytest.fixture
def housing_data_y(file_manager):
    return np.array(pd.read_csv(file_manager.test_data_input, usecols=['price']))

@pytest.fixture
def sklearn_linear_regression(housing_data_X, housing_data_y):
    sklearn_model = LinearRegression().fit(housing_data_X, housing_data_y)
    return sklearn_model

@pytest.fixture
def statsmodels_linear_regression(housing_data_X, housing_data_y):
    statsmodels_model = OLS(housing_data_y, housing_data_X).fit()
    return statsmodels_model

@pytest.fixture
def custom_linear_regression(housing_data_X, housing_data_y):
    custom_model = CustomLinearRegression()
    custom_model.fit(housing_data_X, housing_data_y)
    return custom_model

def test_linear_regression_fit(custom_linear_regression, sklearn_linear_regression, statsmodels_linear_regression) -> None:
    """
    Tests the fit method of the LinearRegression class.
    :return: None
    """

    # Assert
    assert np.testing.assert_almost_equal(custom_linear_regression.theta[1:], sklearn_linear_regression.coef_, decimal=3) is None
    assert np.testing.assert_almost_equal(custom_linear_regression.theta[1:], sklearn_linear_regression.intercept_, decimal=3) is None

    return None

def test_standard_errors(custom_linear_regression, sklearn_linear_regression, statsmodels_linear_regression) -> None:
    """
    Tests the standard_errors method of the LinearRegression class.
    :return: None
    """

    # Assert
    assert custom_linear_regression.standard_errors == pytest.approx(statsmodels_linear_regression.bse, 1e-3)

    return None

def test_p_values(custom_linear_regression, sklearn_linear_regression, statsmodels_linear_regression) -> None:
    """
    Tests the p_values method of the LinearRegression class.
    :return: None
    """

    # Assert
    assert custom_linear_regression.p_values == pytest.approx(statsmodels_linear_regression.pvalues, 1e-3)

    return None








import pytest
import sklearn
import statsmodels
import pandas as pd
import numpy as np

import sys

sys.path.append('/Users/michaelfuest/scratch/')
from utils.classes import FileManager
from src.models import LinearRegression

@pytest.fixture
def file_manager():
    return FileManager()

@pytest.fixture
def housing_data_X(file_manager):
    return pd.read_csv(file_manager.test_data_input, usecols=['bedrooms', 'bathrms', 'stories'])

@pytest.fixture
def housing_data_y(file_manager):
    return pd.read_csv(file_manager.test_data_input, usecols=['price'])

@pytest.fixture
def sklearn_linear_regression():
    return sklearn.linear_model.LinearRegression()

@pytest.fixture
def statsmodels_linear_regression():
    return statsmodels.api.OLS

@pytest.fixture
def custom_linear_regression():
    return LinearRegression()

def test_linear_regression_fit() -> None:
    """
    Tests the fit method of the LinearRegression class.
    :return: None
    """
    # Act
    custom_linear_regression.fit(housing_data_X, housing_data_y)
    sklearn_linear_regression.fit(housing_data_X, housing_data_y)
    statsmodels_linear_regression.fit(housing_data_X, housing_data_y)

    # Assert
    assert custom_linear_regression.theta == pytest.approx(sklearn_linear_regression.coef_, 1e-10)
    assert custom_linear_regression.theta == pytest.approx(statsmodels_linear_regression.params, 1e-10)

    return None

def test_standard_errors() -> None:
    """
    Tests the standard_errors method of the LinearRegression class.
    :return: None
    """
   # Act
    custom_linear_regression.fit(housing_data_X, housing_data_y)
    sklearn_linear_regression.fit(housing_data_X, housing_data_y)
    statsmodels_linear_regression.fit(housing_data_X, housing_data_y)

    # Assert
    assert custom_linear_regression.standard_errors == pytest.approx(statsmodels_linear_regression.bse, 1e-10)

    return None

def test_p_values() -> None:
    """
    Tests the p_values method of the LinearRegression class.
    :return: None
    """
   # Act
    custom_linear_regression.fit(housing_data_X, housing_data_y)
    sklearn_linear_regression.fit(housing_data_X, housing_data_y)
    statsmodels_linear_regression.fit(housing_data_X, housing_data_y)

    # Assert
    assert custom_linear_regression.p_values == pytest.approx(statsmodels_linear_regression.pvalues, 1e-10)

    return None






import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression


from utils.classes import FileManager
from src.models import CustomLinearRegression

@pytest.fixture
def file_manager():
    return FileManager()


@pytest.fixture
def housing_data(file_manager) -> pd.DataFrame:
    return pd.read_csv(file_manager.test_data_input)

@pytest.fixture
def housing_data_x(file_manager, housing_data) -> np.array:
    return np.array(housing_data[['bedrooms', 'bathrooms', 'stories']])

@pytest.fixture
def housing_data_y(file_manager, housing_data) -> np.array:
    return np.array(housing_data['price'])

@pytest.fixture
def sklearn_linear_regression() -> LinearRegression:
    return LinearRegression()

@pytest.fixture
def statsmodels_linear_regression(housing_data_x, housing_data_y) -> sm.OLS:
    return sm.OLS(pd.DataFrame(housing_data_x), housing_data_y)

@pytest.fixture
def custom_linear_regression() -> CustomLinearRegression:
    return CustomLinearRegression()

def test_linear_regression_fit(
        custom_linear_regression,
        sklearn_linear_regression,
        statsmodels_linear_regression,
        housing_data_x,
        housing_data_y
    ) -> None:
    """
    Tests the fit method of the LinearRegression class.
    :return: None
    """
    # Act
    custom_linear_regression.fit(housing_data_x, housing_data_y)
    sklearn_linear_regression.fit(housing_data_x, housing_data_y)

    # Assert
    assert custom_linear_regression.coefficients == pytest.approx(sklearn_linear_regression.coef_, 1e-10)

def test_standard_errors() -> None:
    """
    Tests the standard_errors method of the LinearRegression class.
    :return: None
    """
   # Act
    custom_linear_regression.fit(housing_data_x, housing_data_y)
    sklearn_linear_regression.fit(housing_data_x, housing_data_y)
    statsmodels_linear_regression.fit(housing_data_x, housing_data_y)

    # Assert
    assert custom_linear_regression.standard_errors == pytest.approx(statsmodels_linear_regression.bse, 1e-10)

    return None

def test_p_values() -> None:
    """
    Tests the p_values method of the LinearRegression class.
    :return: None
    """
   # Act
    custom_linear_regression.fit(housing_data_x, housing_data_y)
    sklearn_linear_regression.fit(housing_data_x, housing_data_y)
    statsmodels_linear_regression.fit(housing_data_x, housing_data_y)

    # Assert
    assert custom_linear_regression.p_values == pytest.approx(statsmodels_linear_regression.pvalues, 1e-10)

    return None






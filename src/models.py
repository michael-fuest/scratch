import numpy as np
from scipy.special import gamma

class CustomLinearRegression():

    def __init__(self):
        self.theta = None
        self.coefficients = None
        self.intercept = None
        self.standard_errors = None
        self.p_values = None
        self.r_squared = None
        self.adjusted_r_squared = None

    def fit(self, X, y):
        """
        Fits a linear regression model using the normal equation.
        :param X: Numpy array of shape (n, m) containing the training examples.
        :param y: Numpy array of shape (n, 1) containing the target values.
        :return: None
        """
        n = X.shape[0]
        m = X.shape[1]
        X = np.insert(X, 0, 1, axis=1)
        X = X.T
        self.theta = np.linalg.inv(X.dot(X.T)).dot(X).dot(y)
        self.intercept = self.theta[0]
        self.coefficients = self.theta[1:]
        self.coefficients = self.coefficients.reshape(m,)
        self.calculate_r_squared(X, y)
        self.calculate_standard_errors(X, y)
        #self.calculate_p_values(X)
        

    def calculate_standard_errors(self, X, y):
        """
        Calculates the standard errors of the coefficients.
        :param X: Numpy array of shape (n, m) containing the training examples.
        :param y: Numpy array of shape (n, 1) containing the target values.
        :return: None
        """
        residuals = y - X.T.dot(self.theta)
        residual_variance = np.var(residuals, ddof=len(self.theta))
        cov_matrix = np.linalg.inv(np.dot(X, X.T))
        self.standard_errors = np.sqrt(np.diagonal(residual_variance * cov_matrix))




    def calculate_r_squared(self, X, y):
        """
        Calculates the coefficient of determination (R^2) for the model.
        :param X: Numpy array of shape (n, m) containing the training examples.
        :param y: Numpy array of shape (n, 1) containing the target values.
        :return: None
        """
        y_bar = np.mean(y)
        sum_of_total_squares = np.sum(np.square(y - y_bar))
        sum_of_residual_squares = np.sum(np.square(y - X.T.dot(self.theta)))
        self.r_squared = 1 - (sum_of_residual_squares / sum_of_total_squares)


    def calculate_p_values(self, X):
        """
        Calculates the p-values based on a two tailed hypothesis test for the coefficients.
        :param X: Numpy array of shape (n, m) containing the training examples.
        :param y: Numpy array of shape (n, 1) containing the target values.
        :return: None
        """
        degrees_of_freedom = X.shape[0] - X.shape[1] - 1

        assert degrees_of_freedom > 0, "The degrees of freedom must be greater than zero."

        t_stats = self.theta / self.standard_errors
        self.p_values = [CustomLinearRegression.get_students_t_pdf_value(np.abs(t_stat), degrees_of_freedom) for t_stat in t_stats]


    @staticmethod
    def get_students_t_pdf_value(value, degrees_of_freedom):
        """
        Calculates the probability density function value for a given value and degrees of freedom.
        :param value: The value to calculate the pdf for.
        :param degrees_of_freedom: The degrees of freedom.
        :return: The pdf value.
        """
        
        counter = gamma((degrees_of_freedom + 1) / 2)
        denominator = gamma(degrees_of_freedom / 2) * np.sqrt(degrees_of_freedom * np.pi)

        assert denominator != 0, "The denominator cannot be zero."

        first_term = counter / denominator
        second_term = (1 + (value**2 / degrees_of_freedom))**(-(degrees_of_freedom + 1) / 2)
        return first_term * second_term
    

    def predict(self, X):
        """
        Predicts the target values for the given examples.
        :param X: Numpy array of shape (n, m) containing the examples.
        :return: Numpy array of shape (n, 1) containing the predicted target values.
        """
        return X.dot(self.theta)


    

        




        

        
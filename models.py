import numpy as np
import pandas as pd

class LinearRegression():

    def __init__(self):
        self.theta = None
        self.standard_errors = None
        self.p_values = None
        self.r_squared = None
        self.adjusted_r_squared = None

    def fit(self, X, y):
        """
        Fits a linear regression model using the normal equation.
        :param X: Numpy array of shape (m, n) containing the training examples.
        :param y: Numpy array of shape (m, 1) containing the target values.
        :return: None
        """
        self.theta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(y)
        self.calculate_r_squared(self, X, y)
        self.calculate_standard_errors(self, X, y)
        self.calculate_p_values(self, X, y)
        

    
    def calculate_standard_errors(self, X, y):
        """
        Calculates the standard errors of the coefficients.
        :param X: Numpy array of shape (m, n) containing the training examples.
        :param y: Numpy array of shape (m, 1) containing the target values.
        :return: None
        """
        error = y - X.dot(self.theta)
        sigma = np.var(error)
        self.standard_errors = np.sqrt(np.diagonal(sigma * np.linalg.inv(np.transpose(X).dot(X))))



    def calculate_r_squared(self, X, y):
        """
        Calculates the coefficient of determination (R^2) for the model.
        :param X: Numpy array of shape (m, n) containing the training examples.
        :param y: Numpy array of shape (m, 1) containing the target values.
        :return: None
        """
        x_bar = np.mean(X)
        sum_of_total_squares = np.sum(np.square(X - x_bar))
        sum_of_residual_squares = np.sum(np.square(y - X.dot(self.theta)))
        self.r_squared = 1 - (sum_of_residual_squares / sum_of_total_squares)

    def calculate_p_values(self, X, y):
        """
        Calculates the p-values for the coefficients.
        :param X: Numpy array of shape (m, n) containing the training examples.
        :param y: Numpy array of shape (m, 1) containing the target values.
        :return: None
        """
        
        




        

        
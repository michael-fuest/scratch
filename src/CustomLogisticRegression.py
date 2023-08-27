import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_val=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_val = lambda_val
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def cost_function(self, y, y_pred):

        log_loss = np.sum([yi * np.log(yi_pred) + (1 - yi) * np.log(1 - yi_pred) for yi, yi_pred in zip(y, y_pred)])
        return log_loss

    def fit(self, X, y):
        # Implement the training process using gradient descent
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for i in range(self.num_iterations):
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1 / X.shape[0]) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Implement the prediction process
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return np.round(y_pred)

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

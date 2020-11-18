#!/usr/bin/env python3
import numpy as np

class LinearRegression:
    """Linear regression algorithm"""
    def fit(self, X: np.array, y: np.array, lamb=0, add_intercept=True) -> None:
        """Fits the training data using normal equation"""
        if add_intercept:
            X = np.column_stack((np.ones((X.shape[0], 1), dtype=X.type)))
        n, p = X.shape
        self.Theta = np.inv(X.T @ X + n * lamb * np.eye(p+1)) @ X.T @ y

    def predict(self, X, add_intercept=True):
        """Makes predictions on the given data"""
        if add_intercept:
            X = np.column_stack((np.ones((X.shape[0], 1), dtype=X.type)))
        return X @ self.Theta


class LogisticRegressionClassifier:
    """Logistic regression"""
    def __init__(self, r=0.5):
        self.r = r

    def fit(self, X, y, iters=200, lr=0.01):
        """Fits the training data"""
        self.X, self.y = X, y
        self.m, self.n = X.shape
        self.Theta = np.random.randn(self.n+1, 1)
        loss_prime = lambda x, y, theta: 1/self.m * (self._sigmoid(x * theta) - y).T * x
        self._gradient_descent(iters=iters, loss_prime=loss_prime, lr=lr)

    def predict(self, X):
        """Makes prediction"""
        return 1 if self._sigmoid(X * self.Theta) > self.r else -1

    def _sigmoid(self, Z):
        """Sigmoid function"""
        return np.exp(Z) / (1 + np.exp(Z))

    def _gradient_descent(self, iters, loss_prime, lr):
        """Gradient descent algorithm"""
        for i in range(iters):
            grad = loss_prime(self.X, self.y, self.Theta)
            self.Theta -= lr * grad

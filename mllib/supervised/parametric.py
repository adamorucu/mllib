#!/usr/bin/env python3
import numpy as np


class LinearRegression:
    """Linear regression algorithm"""
    def fit(self, X, y, lamb=0, add_intercept=True, iters=1000, lr=0.006):
        """Fits the training data using normal equation"""
        if add_intercept:
            X = np.column_stack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        n, p = X.shape
        self.X = X
        self.y = y
        self.Theta = np.random.randn(n + 1, 1)
        #         self.Theta = np.linalg.inv(X.T @ X) @ X.T @ y
        loss_prime = lambda x, y, theta: (x @ theta - y).T @ x
        self._gradient_descent(iters=iters, loss_prime=loss_prime, lr=lr)

    def predict(self, X, add_intercept=True):
        """Makes predictions on the given data"""
        if add_intercept:
            X = np.column_stack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        return X @ self.Theta

    def _gradient_descent(self, iters, loss_prime, lr):
        """Gradient descent algorithm"""
        for _ in range(iters):
            grad = loss_prime(self.X, self.y, self.Theta)
            self.Theta -= lr * grad.T


class LogReg:
    def __init__(self, r=0.5):
        self.r = r

    def loss(self, x, y):
        expo = np.exp(self.Theta @ x)
        if y == 1:
            return expo/(1+expo)
        else:
            return 1/(1+expo)

class LogisticRegressionClassifier:
    """Logistic regression"""
    def __init__(self, r=0.5):
        self.r = r

    def fit(self, X, y, iters=500, lr=0.01):
        """Fits the training data"""
        self.y = y
        self.X = np.column_stack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        self.m, self.n = X.shape
        self.Theta = np.random.randn(self.n + 1, 1)
        loss_prime = lambda x, y, theta: 1 / self.m * (self._sigmoid(x @ theta)
                                                       - y).T @ x
        self._gradient_descent(iters=iters, loss_prime=loss_prime, lr=lr)

    def predict(self, X):
        """Makes prediction"""
        X = np.column_stack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        return np.array([[1 if res > self.r else -1]
                         for res in self._sigmoid(X @ self.Theta)])

    def _sigmoid(self, Z):
        """Sigmoid function"""
        return np.exp(Z) / (1 + np.exp(Z))

    def _gradient_descent(self, iters, loss_prime, lr):
        """Gradient descent algorithm"""
        for _ in range(iters):
            grad = loss_prime(self.X, self.y, self.Theta)
            self.Theta -= lr * grad.T

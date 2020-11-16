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


class LogisticRegression:
    """Logistic regression"""
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _gradient_descent(self, loss, ...):
        pass


class kNearestNeighbors:
    """kNN Algorithm"""
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict_one(self, x_hat, ptype="classification"):
        neighbor_dists = []
        for x in self.X:
            neighbor_dists.append(np.linalg.norm(x_hat-x))

        nn = [y for _, y in sorted(zip(neighbor_dists, self.y))][:self.k]
        if ptype = 'classification':
            return max(nn, key=nn.count)
        else ptype = 'regression':
            return sum(nn)/len(nn)

    def predict(self, X, ptype="classification"):
        pred = []
        for x_hat in X:
            pred.append(self.predict_one(x_hat, ptype))
        return np.array(pred)

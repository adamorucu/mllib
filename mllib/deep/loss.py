import numpy as np

class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError


class SquaredError(Loss):
    def loss(self, predicted, true):
        return np.sum((predicted - true) ** 2)

    def grad(self, predicted, true):
        return 2 * (predicted - true)

import numpy as np

def gradient_descent(self, iters, loss_prime, lr):
    """Gradient descent algorithm"""
    for i in range(iters):
        grad = loss_prime(self.X, self.y, self.Theta)
        self.Theta -= lr * grad.T
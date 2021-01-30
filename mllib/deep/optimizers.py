import numpy as np

from .neuralnet import NeuralNet

class Optimizer:
    def step(self, nn):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, nn):
        for param, grad in nn.get_params_and_grads():
            param -= self.lr * grad
            

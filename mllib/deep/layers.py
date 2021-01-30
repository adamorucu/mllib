import numpy as np

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size, output_size):
        """
        intput_size: (batch_size, input_size)
        output_size: (batch_size, output_size)
        """
        super().__init__()
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad):
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T

class Activation(Layer):
    def __init__(self, func, func_prime):
        super().__init__()
        self.func = func
        self.func_prime = func_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.func_prime(self.inputs) * grad

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - tanh(x) ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

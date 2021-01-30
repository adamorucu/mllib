import sys
sys.path.append('../')

import numpy as np

from deep.train import train
from deep.neuralnet import NeuralNet
from deep.layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    ])

targets = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0]
    ])

nn = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(nn, inputs, targets)
for x, y in zip(inputs,targets):
    preds = nn.forward(x)
    print(x, preds, y)

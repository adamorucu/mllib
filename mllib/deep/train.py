import numpy as np

from .neuralnet import NeuralNet
from .loss import SquaredError
from .optimizers import SGD
from .data import BatchIterator


def train(nn, inputs, targets, num_epochs=5000, iterator=BatchIterator(), loss=SquaredError(), optimizer=SGD()):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            preds = nn.forward(batch['inputs'])
            epoch_loss += loss.loss(preds, batch['targets'])
            grad = loss.grad(preds, batch['targets'])
            nn.backward(grad)
            optimizer.step(nn)
        print(f'Epoch {epoch}, loss: {epoch_loss}')

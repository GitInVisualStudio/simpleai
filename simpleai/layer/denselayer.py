import numpy as np
from .layer import *

class DenseLayer(Layer):

    def __init__(self, size, connections, ac_f):
        self.ac_f = ac_f
        weights = np.random.rand(connections, size) * 2 - 1
        biases = np.random.rand(connections) * 2 - 1
        super().__init__(("w", weights), ("b", biases))
    
    def forward(self, input):
        self.activation = input
        self.output = self.ac_f(self.parameter["w"].dot(input) + self.parameter["b"], False)
        return self.output

    def backward(self, error):
        delta = self.ac_f(self.output, True) * error
        self.deriv["b"] += delta
        self.deriv["w"] += np.outer(delta, self.activation)
        return delta.dot(self.parameter["w"])

    def get_gradient(self, error):
        delta = self.ac_f(self.output, True) * error
        return delta.dot(self.parameter["w"])

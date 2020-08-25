import numpy as np

class Layer:

    def __init__(self, *parameter):
        self.parameter = dict(parameter)
        self.deriv = dict([(key, np.zeros(value.shape)) for key, value in parameter])

    def forward(self, input):
        pass

    def backward(self, error):
        pass

    def get_gradient(self, error):
        pass


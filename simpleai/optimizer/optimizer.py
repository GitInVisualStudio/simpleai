import numpy as np

class Optimizer:

    def optimize(self, value):            
        pass

class SGD(Optimizer):

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def optimize(self, value):
        return value * self.learning_rate    

class Momentum(Optimizer):

    def __init__(self, learning_rate=0.1, b=0.9):
        self.learning_rate = learning_rate
        self.b = b
        self.v_value = None

    def optimize(self, value):
        
        if self.v_value is None:
            self.v_value = (1 - self.b) * value

        self.v_value = self.v_value * self.b + (1 - self.b) * value
        return self.learning_rate * self.v_value

class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-5):
        self.b1 = b1
        self.b2 = b2
        self.learning_rate = learning_rate
        self.e = e
        self.r_value = None
        self.v_value = None

    def optimize(self, value):

        if self.r_value is None:
            self.r_value = value * value * (1 - self.b2)
            self.v_value = value * (1 - self.b1)

        self.v_value = self.v_value * self.b1 + value * (1 - self.b1)
        self.r_value = self.r_value * self.b2 + (value * value) * (1 - self.b2)
        return self.learning_rate * self.v_value / (np.sqrt(self.r_value) + self.e)

class RMSProp(Optimizer):

    def __init__(self,  learning_rate=0.001, b=0.999, e=1e-5):
        self.learning_rate = learning_rate
        self.b = b
        self.e = e
        self.r_value = None

    def optimize(self, value):

        if self.r_value is None:
            self.r_value = value * value * (1 - self.b)

        self.r_value = self.r_value * self.b + (value * value) * (1 - self.b)
        return self.learning_rate * value / (np.sqrt(self.r_value) + self.e)

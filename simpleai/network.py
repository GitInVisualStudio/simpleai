import numpy as np
import time
from simpleai.utils import get_function
import copy

class Network:

    def __init__(self, optimizer, loss_func, track=["loss"]):
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.layer = []
        self.track = []
        for _track in track:
            self.track.append(get_function(_track))

    def add_layer(self, layer):
        self.layer.append(layer)

    def compile(self):
        self.deriv = []
        for layer in self.layer:
            for key in layer.deriv.keys():
                self.deriv.append((layer, key, copy.copy(self.optimizer)))

    def optimize(self):
        for layer, key, optimizer in self.deriv:
            value = layer.deriv[key]
            value = optimizer.optimize(value)
            layer.parameter[key] -= value
            layer.deriv[key] = np.zeros(value.shape)

    def feed_forward(self, input):
        for layer in self.layer:
            input = layer.forward(input)
        return input

    @property
    def output(self):
        return self.layer[-1].output

    @property
    def prediction(self):
        output = self.output
        return np.where(output == output.max())[0][0]
    
    def get_gradient(self, input, output):
        model_output = self.feed_forward(input)
        gradient = self.loss_func(model_output, output)
        for layer in reversed(self.layer):
            gradient = layer.get_gradient(gradient)
        return gradient

    @property
    def parameter(self):
        parameter = []
        for layer in self.layer:
            parameter.append(list(layer.parameter.values()))
        return pramaeter

    def feed_backward(self, input, output):
        model_output = self.feed_forward(input)
        loss = self.loss_func(model_output, output)
        error = loss
        for layer in reversed(self.layer):
            error = layer.backward(error)
        return [_track(model_output, output) for _track in self.track]

    def fit(self, epochs, train_x, train_y, batch_size=16):
        track_histroy = np.zeros(len(self.track))
        for epoch in range(epochs):
            k = 0
            track_values = np.zeros(len(self.track))
            start = time.time()
            for input, output in zip(train_x, train_y):
                k += 1
                values = self.feed_backward(input, output)
                track_values += values
                if k % batch_size == 0:
                    self.optimize()
            log = f"Time: {time.time() - start}"
            for i, track in enumerate(self.track, 0):
                log += f" {track.__name__}: {track_values[i] / k}"
            print(log)

import numpy as np

def get_mnist(length=55000):
    inputs = np.genfromtxt("train.csv", delimiter=',', dtype=int, max_rows=length)
    outputs = []
    for i in range(len(inputs)):
        vector = inputs[i]
        output = np.zeros(10)
        output[vector[0]] = 1
        outputs.append(output)
    inputs = np.delete(inputs, 0, 1) / 255
    return inputs, outputs

def sigmoid(value, deriv):
    if deriv:
        return value * (1 - value)
    try:
        output = 1 / (1 + np.exp(-value))
    except:
        print(value.shape)
    return output

def relu(value, deriv):
    if deriv:
        value[value > 0] = 1
    else:
        value[value < 0] = 0
    return value

def error(output, actual):
    return output - actual

def log_error(output, actual):
    return -np.log(1 - output + actual)

def loss(output, actual):
    return np.abs(output - actual).sum()

def acc(output, actual):
    index = np.amax(output)
    index = np.where(output == index)[0][0]
    return actual[index] == 1

def get_function(name):
    return globals()[name]

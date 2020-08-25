from simpleai.layer import DenseLayer
from simpleai.utils import sigmoid, error
from simpleai.optimizer import Adam
from simpleai import Network
from simpleai import utils

def create_network():
    optimizer = Adam(learning_rate=0.003)
    network = Network(optimizer, loss_func=error, track=["loss" , "acc"])

    #create new layer
    network.add_layer(DenseLayer(784, 16, sigmoid))
    network.add_layer(DenseLayer(16, 16, sigmoid))
    network.add_layer(DenseLayer(16, 10, sigmoid))

    #compile the network
    network.compile()
    return network

if __name__ == '__main__':
    network = create_network()
    train_x, train_y = utils.get_mnist(10000)
    network.fit(epochs=5, train_x=train_x, train_y=train_y, batch_size=1)

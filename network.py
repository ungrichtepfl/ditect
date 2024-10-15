"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

from __future__ import annotations

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

WEIGHTS = [
    np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
    ),
    np.array(
        [
            [0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1],
        ]
    ),
]
BIASES = [
    np.array([[0.1, 0.2, 0.3]]).T,
    np.array([[0.4, 0.5]]).T,
]


class Network(object):
    def __init__(self, biases: list[np.ndarray], weights: list[np.ndarray]):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        assert len(weights) == len(biases)
        self.num_layers = len(weights) + 1
        self.sizes = [w.shape[1] for w in weights] + [weights[-1].shape[0]]
        self.weights = weights
        self.biases = biases
        # self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # self.weights = [np.random.randn(y, x)
        #                 for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        layers = [(a, a)]
        for b, w in zip(self.biases, self.weights):
            input = np.dot(w, a) + b
            a = sigmoid(input)
            layers.append((input, a))
        return a, layers

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y, zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)[0]), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y, z):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y) * sigmoid_prime(z)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def main():
    print("---- FEEDFORWARD ----")
    network = Network(BIASES, WEIGHTS)
    input = np.array([[0.1, 0.2]]).T
    _, layers = network.feedforward(input)
    for i, (inp, a) in enumerate(layers, start=1):
        print(f"Input {i}: {inp.T}")
        print(f"Activation {i}: {a.T}")
    print("----------------------")
    print("------ LAST ERROR ------")
    a = 0.3
    z = 0.5
    y = 0.3
    o = network.cost_derivative(a, y, z)
    print(f"Last cost: {o}")
    a = 0.8
    z = 0.3
    y = 0.1
    o = network.cost_derivative(a, y, z)
    print(f"Last cost: {o}")
    a = 0.7
    z = 0.8
    y = 0.9
    o = network.cost_derivative(a, y, z)
    print(f"Last cost: {o}")

    print("----------------------")
    print("------ BACKPROP 1 ------")
    x = np.array([[0.2, 0.1]]).T
    y = np.array([[0.5, -0.3]]).T
    nablas_b, nablas_w = network.backprop(x, y)
    for i, (nabla_b, nabla_w) in enumerate(zip(nablas_b, nablas_w), start=1):
        print(f"Errors b {i}: {nabla_b.T}")
        print(f"Errors w {i}:\n{nabla_w}")
    print("----------------------")

    print("------ Mini Batch ------")
    xs = [np.array([[0.3, 0.2]]).T, np.array([[0.4, 0.5]]).T]
    ys = [np.array([[0.1, -0.2]]).T, np.array([[-0.3, 0.7]]).T]
    mini_batch = list(zip(xs, ys))
    network = Network(BIASES, WEIGHTS)
    eta = 0.8
    network.update_mini_batch(mini_batch, eta)
    for i, (bias, weight) in enumerate(zip(network.biases, network.weights), start=1):
        print(f"Bias {i}: {bias.T}")
        print(f"Weight {i}: {weight}")
    network.update_mini_batch(mini_batch, eta)
    for i, (bias, weight) in enumerate(zip(network.biases, network.weights), start=1):
        print(f"Bias {i}: {bias.T}")
        print(f"Weight {i}: {weight}")
    print("----------------------")


if __name__ == "__main__":
    main()

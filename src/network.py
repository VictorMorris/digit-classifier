import random
import numpy as np

class Network:
    # Initializes a network with random weights and biases
    def __init__(self, sizes:list[int])->None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        # self.biases[layer(starting at layer 2)][node] = bias
        self.biases = [np.random.default_rng().standard_normal(size=(y, 1)) for y in sizes[1:]]
        # self.weights[layer][node from layer-1][node from layer] = weight
        self.weights = [np.random.default_rng().standard_normal(size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

    def print_biases(self):
        layer = 0
        while layer <= self.num_layers-2:
            print(f"----- Layer {layer+2} biases -----")
            for neuron in range(len(self.biases[layer])):
                print(f"The bias for neuron {neuron+1} is {self.biases[layer][neuron]}")
            layer += 1


    def print_weights(self):
        layer = 0
        while layer <= self.num_layers-2:
            print(f"----- Weights from layer {layer} to layer {layer+1} -----")
            for col in range(len(self.weights[layer][0])):
                for row in range(len(self.weights[layer])):
                    print(f"The weight from N{col+1}(layer={layer}) to N{row+1}(layer={layer+1}) is {self.weights[layer][row][col]}")
            layer += 1

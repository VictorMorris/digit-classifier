import os
import random
import numpy as np


class Network:
    def __init__(self, sizes: list[int]) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes

        # A layer of biases is stored for every layer except the input layer
        # Each layer of biases are stored in a column matrix
        self.biases = [np.random.default_rng().standard_normal(size=(y, 1)) for y in sizes[1:]]

        # A matrix of weights is stored for every neuron connection between every layer
        # Has a row for every neuron in the following layer
        # Has a column for every nueron in the layer
        self.weights = [np.random.default_rng().standard_normal(size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

        self.test_errors = np.array([])

    def save(self, filepath):
        """Save the network's weights and biases to a .npz file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        weight_keys = {f'w{i}': w for i, w in enumerate(self.weights)}
        bias_keys = {f'b{i}': b for i, b in enumerate(self.biases)}
        np.savez(filepath, **weight_keys, **bias_keys)

    @classmethod
    def load(cls, filepath):
        """Load a network from a .npz file. Assumes fixed architecture [784, 16, 16, 10]."""
        sizes = [784, 16, 16, 10]
        net = cls(sizes)
        data = np.load(filepath)
        num_weight_arrays = len(sizes) - 1
        net.weights = [data[f'w{i}'] for i in range(num_weight_arrays)]
        net.biases = [data[f'b{i}'] for i in range(num_weight_arrays)]
        return net

    def feed_forward(self, data):
        a = data
        for w, b in zip(self.weights, self.biases):
            # The activation of neurons in the next layer is calculated by:
            # Weights (size of layer)
            a = self.sigmoid(w @ a + b)
        # The final activations of the network
        # A column matrix of length sizes[-1] (One activation for every neuron in the output layer)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, epoch_callback=None):
        """Stocastic Gradient Descent"""
        if test_data: n_test = len(test_data)
        n = len(training_data) # Total samples
        # Evaluate before any training to get the initial error rate
        if test_data:
            initial_error = 100*(1-(self.evaluate(test_data)/n_test))
            self.test_errors = np.append(self.test_errors, initial_error)
            print(f"Epoch 0 (untrained): {self.evaluate(test_data)} / {n_test}")
            if epoch_callback:
                epoch_callback(initial_error)
        for j in range(epochs):
            random.shuffle(training_data) # Shuffle is to break ordering bias
            # Slices training data into batches of size mini_batch_size
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) # One gradient step per batch
            # Update the graph for every epoch completed
            if test_data:
                test_error = 100*(1-(self.evaluate(test_data)/n_test))
                self.test_errors = np.append(self.test_errors, test_error)
                print(f"Epoch {j+1}: {self.evaluate(test_data)} / {n_test}")
                if epoch_callback:
                    epoch_callback(test_error)
            else:
                print(f"Epoch {j+1} complete")
                if epoch_callback:
                    epoch_callback(None)

    def update_mini_batch(self, mini_batch, eta):
        """Apply one gradient descent step"""
        # These will accumulate the total gradient across the mini-batch
        nabla_b = [np.zeros(b.shape) for b in self.biases] # List of zero arrays, same shape as self.biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] # List of zero arrays, same shape as self.weights
        for x, y in mini_batch: # Loop each sample
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # Get gradients for this sample
            # Accumulates the gradients
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # At this point nabla_b and nabla_w hold the sum of the gradiants from every sample
        # eta / len(mini_batch) gets the average gradient by dividing by batch size
        # Subtract the gradient for each weight and bias matrices
        # Gradient descent update: w = w - eta * gradient
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Compute the exact gradient of the loss"""
        # These will accumulate the total gradient across the mini-batch
        nabla_b = [np.zeros(b.shape) for b in self.biases] # List of zero arrays, same shape as self.biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] # List of zero arrays, same shape as self.weights
        
        # Forward pass
        activation = x # Start with the input
        activations = [x] # Store all the layer activations - needed for backward pass
        zs = [] # Store all the pre-activation values (z=w*a + b)
        for b, w in zip(self.biases, self.weights):
            z = w @ activation + b #Pre activation
            zs.append(z) # Save the z for this layer
            activation = self.sigmoid(z)
            activations.append(activation) # Save the activation for this layer

        # Backward pass
        # Tells how much each output neuron contributed to the error
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta # Output layer bias gradient
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # Output layer weight gradient
        
        # Propagate the rror signal backward using the chain rule
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # Propagate delta backward
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Accuracy check"""
        test_results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results) # Count correct predictions
    
    def cost_derivative(self, output_activations, y):
        """Computes the derivative of the mean squared error loss"""
        return(output_activations-y)
    

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def print_biases(self):
        layer = 0
        while layer <= self.num_layers - 2:
            print(f"----- Layer {layer+2} biases -----")
            for neuron in range(len(self.biases[layer])):
                print(f"The bias for neuron {neuron+1} is {self.biases[layer][neuron]}")
            layer += 1

    def print_weights(self):
        layer = 0
        while layer <= self.num_layers - 2:
            print(f"----- Weights from layer {layer} to layer {layer+1} -----")
            for col in range(len(self.weights[layer][0])):
                for row in range(len(self.weights[layer])):
                    print(f"The weight from N{col+1}(layer={layer}) to N{row+1}(layer={layer+1}) is {self.weights[layer][row][col]}")
            layer += 1

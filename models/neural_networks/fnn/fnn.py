'''
Feedforward Neural Network (vanilla NN)

Classes:
- Neuron
- NeuralNetwork
- Layer
'''
import numpy as np
import json
import os

from utils.data import DataLoader
from utils.metrics import CrossEntropyLoss, MeanSquaredError, Accuracy
from utils.activation import Activation
        
class Layer:
    def __init__(self, num_neurons: int, num_inputs: int, activation: str):
        '''
        An array of neurons representing a layer in the network

        Parameters:
        - num_neurons: Number of neurons in the layer being created
        - num_inputs: Number of neurons in the previous layer
        - activation: String defining the activation function used in the layer ['sigmoid', 'relu', 'leaky']
        '''
        self.num_neurons = num_neurons

        # Arrays to store weights/biases for each neuron in the layer
        '''
        He Initialization:
        - W[l] = np.random.randn(size_l, size_l-1) * np.sqrt(2/size_l-1)
        '''
        self.W = np.random.randn(num_neurons, num_inputs) * np.sqrt(2/num_inputs)
        self.B = np.random.randn(num_neurons) * np.sqrt(2/num_inputs)

        if activation == 'sigmoid':
            self.activation = Activation('sigmoid')
        elif activation == 'relu':
            self.activation = Activation('relu')
        elif activation == 'softmax':
            self.activation = Activation('softmax')
        elif activation == 'leaky':
            self.activation = Activation('leaky')
        else:
            print("Invalid activation function")


class NeuralNetwork:
    def __init__(self, layers: list[int], activation: str, in_features: int, num_classes: int, lr: float = 0.01):
        '''
        Network built as an array of layers

        Parameters:
        - layers: Array describing the number of neurons in each layer
            * e.g. [1, 2, 3] means there are 3 hidden layers, with 1, 2, and 3 neurons respectively
        - activation: String defining the activation function used in hidden layers ['sigmoid', 'relu', 'leaky']
        - in_features: Number of input features
        - num_classes: Number of output classes
        - lr: Learning rate for training (0, 1)
        '''
        self.in_features = in_features
        self.num_classes = num_classes

        self.lr = lr

        if activation not in ['sigmoid', 'relu', 'leaky']:
            print("Error: Invalid hidden layer activation function")
            return
        self.activation = activation
        
        # -- Array to hold Layer objects
        self.layers = []

        # -- Add hidden layers
        num_inputs = in_features

        for num_neurons in layers:
            self.layers.append(Layer(num_neurons, num_inputs, activation))
            # Update num_inputs for next layer
            num_inputs = num_neurons

        # -- Add output layer
        if num_classes == 2:
            self.layers.append(Layer(num_neurons=1, num_inputs=num_inputs, activation='sigmoid'))
        else:
            self.layers.append(Layer(num_neurons=num_classes, num_inputs=num_inputs, activation='softmax'))

    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Passes an input array into the network and returns the network's output (0, 1)
        - X: numpy array of a single sample of input data (batch_size, x)
        '''
        # Save input for backprop
        self.X = X

        for layer in self.layers:
            Z = np.dot(X, layer.W.T) + layer.B

            A = layer.activation(Z)

            layer.Z = Z
            layer.A = A

            # Next layer in (X) = Current layer out (A)
            X = A
    
        return self.layers[-1].A
    
    def backward(self, y: np.ndarray, y_hat: np.ndarray):
        '''
        Performs backpropagation and applies updates to weights
        - For a batch: Averages the update gradients of all samples in the batch

        Track the error terms (δ) for each layer, then use it to calculate the weight gradient
        - dL/dw_ij = (∑ δ_k * w_jk) * da/dz_j * dz_j/dw_ij
                   = δ_j * a_i
            * Where w_ij is the weight from neuron i to neuron j
                    δ_k is the error term for the kth neuron in the kth layer (layer after j)
                    a_i is the output of neuron i

        - y: Numpy array of correct labels for the batch
        - y_hat: Numpy array of network predictions for the batch
        '''
        batch_size = y.shape[0]

        # Lists to store δ_i, ∂w_i, and ∂b_i for each layer i
        delta = []
        der_w = []
        der_b = []

        # -- Output layer
        '''
        Binary case:
        - y: (s, )
        Multiclass case:
        - y: (s, c)
        '''
        o = self.layers[-1]

        # -- Binary
        if self.num_classes == 2:
            dL = (y_hat - y.reshape(-1, 1))
            delta_o = np.array(dL * o.activation.der(o.Z))
        # -- Multiclass
        else:
            # Convert y to one hot encoding for softmax, e.g. [0 1] for y = 1
            y = np.eye(self.num_classes)[y.astype(int)]
            # δ_o = y_hat - y for backprop with softmax/cross entropy loss
            delta_o = y_hat - y

        # ∂W_o = δ_o.T * A_i (averaged over batch)
        der_w_o = np.dot(delta_o.T, self.layers[-2].A) / batch_size
        # ∂B_o = δ_o
        der_b_o = np.sum(delta_o, axis=0) / batch_size

        # Store δ_o, ∂W_o, and ∂B_o
        delta.insert(0, delta_o)
        der_w.insert(0, der_w_o)
        der_b.insert(0, der_b_o)

        '''
        For each layer j back from the output layer: 
        - delta_j = delta_k * W_k.T * a'(z_j)
        - der_w_j = δ_k * a_i
        '''
        for layer in range(len(self.layers) - 2, -1, -1):
            j = self.layers[layer]
            k = self.layers[layer + 1]

            # Succeeding layer
            delta_k = delta[0]
            # δ_j = ∑(δ_k * W_k) * ∂A/∂Z
            delta_j = np.dot(delta_k, k.W) * j.activation.der(j.Z)
            # ∂B_j = δ_j (averaged over batch)
            der_b_j = np.sum(delta_j, axis=0) / batch_size

            # -- Input layer
            if layer == 0:
                # ∂W_j = δ_j.T * X (averaged over batch)
                der_w_j = np.dot(delta_j.T, self.X) / batch_size
            # -- Hidden layer
            else:
                i = self.layers[layer - 1]
                # # ∂W_j = δ_j.T * A_i (averaged over batch)
                der_w_j = np.dot(delta_j.T, i.A) / batch_size

            # Store δ_j, ∂W_j, and ∂B_j
            delta.insert(0, delta_j)
            der_w.insert(0, der_w_j)
            der_b.insert(0, der_b_j)

        '''
        Update weights
        - Average the weight derivative among the batch samples for each weight
        '''
        for i, layer in enumerate(self.layers):
            layer.W = layer.W - self.lr * der_w[i]
            layer.B = layer.B - self.lr * der_b[i]


    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs=100):
        print(f"Initial Train Accuracy: {self.test(train_loader)[0] * 100:.2f}%")

        for epoch in range(epochs):
            for X, y in train_loader:
                y_hat = self.forward(X)
                self.backward(y, y_hat)

            loss = CrossEntropyLoss(y_hat, y)
            print(f"Epoch {epoch} -- Loss: {loss}, Train Accuracy: {self.test(train_loader)[0] * 100:.2f}%")


    def test(self, loader: DataLoader):
        total_correct = 0
        total_incorrect = 0
        total_accuracy = 0

        for i, (X, y) in enumerate(loader):
            y_hat = self.forward(X)

            batch_accuracy, batch_correct, batch_incorrect = Accuracy(y_hat, y)

            total_correct += batch_correct
            total_incorrect += batch_incorrect
            total_accuracy += batch_accuracy

        return total_accuracy / (i + 1), total_correct, total_incorrect

    def save(self, filename: str) -> None:
        model_params = {'layers': [layer.num_neurons for layer in self.layers[:-1]],
                        'in_features': self.in_features,
                        'num_classes': self.num_classes,
                        'activation': self.activation,
                        'lr': self.lr,
                        'W': [layer.W.tolist() for layer in self.layers],
                        'B': [layer.B.tolist() for layer in self.layers]}
        
        # Create saved_models directory
        os.makedirs('models/fnn/saved_models', exist_ok=True)
        
        with open(f'models/fnn/saved_models/{filename}.json', 'w') as file:
            json.dump(model_params, file)

    @staticmethod
    def load(filepath: str) -> 'NeuralNetwork':
        with open(filepath, 'r') as file:
            model_params = json.load(file)

        layers = model_params['layers']
        activation = model_params['activation']
        lr = model_params['lr']
        W = model_params['W']
        B = model_params['B']
        in_features = model_params['in_features']
        num_classes = model_params['num_classes']

        model = NeuralNetwork(layers=layers,
                              activation=activation,
                              in_features=in_features,
                              num_classes=num_classes,
                              lr=lr)

        for i, layer in enumerate(model.layers):
            layer.W = np.array(W[i])
            layer.B = np.array(B[i])

        return model

        
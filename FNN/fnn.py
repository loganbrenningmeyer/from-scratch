'''
Feedforward Neural Network (vanilla NN)

Classes:
- Neuron
- NeuralNetwork
- Layer
'''
import numpy as np

from data import Dataset, DataLoader
from loss import CrossEntropyLoss, MeanSquaredError

'''
Stores multiple activation functions and their derivative functions
'''
class Activation:

    def __init__(self, activation: str):
        if activation == 'sigmoid':
            self.output = self.sigmoid
            self.der = self.d_sigmoid
        elif activation == 'relu':
            self.output = self.relu
            self.der = self.d_relu
        elif activation == 'softmax':
            self.output = self.softmax
            self.der = self.d_softmax
        elif activation == 'leaky':
            self.output = self.leaky_relu
            self.der = self.d_leaky_relu

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return self.output(Z)
    
    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    @staticmethod
    def d_softmax(Z: np.ndarray) -> np.ndarray:
        jacobian = np.empty(shape=(Z.shape[0], Z.shape[1], Z.shape[1]))

        for i, s in enumerate(Z):
            jacobian[i] = np.diag(s) - np.outer(s, s)

        return jacobian

    @staticmethod
    def sigmoid(Z: np.ndarray) -> np.ndarray:
        Z = np.clip(Z, -500, 500)  # Clip values to avoid overflow
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def d_sigmoid(Z: np.ndarray) -> np.ndarray:
        return Activation.sigmoid(Z) * (1 - Activation.sigmoid(Z))
    
    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    @staticmethod
    def d_relu(Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)
    
    @staticmethod
    def leaky_relu(Z, alpha=0.01):
        return np.maximum(alpha * Z, Z)
    
    @staticmethod
    def d_leaky_relu(Z, alpha=0.01):
        dZ = np.ones_like(Z)
        dZ[Z < 0] = alpha
        return dZ
        
'''
An array of neurons representing a layer in the network
- num_neurons: Number of neurons in the layer being created
- num_inputs: Number of neurons in the previous layer
'''
class Layer:
    def __init__(self, num_neurons: int, num_inputs: int, activation: str):
        self.num_neurons = num_neurons

        # -- Arrays to store weights/biases for each neuron in the layer
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

'''
Network built as an array of layers
- layers: Array describing the number of neurons in each layer
  i.e. [1, 2, 3] means there are 3 hidden layers, with 1, 2, and 3 neurons respectively
'''
class NeuralNetwork:
    def __init__(self, layers: list[int], activation: str, in_features: int, classes: int, lr: float = 0.01):
        self.in_features = in_features
        self.classes = classes
        self.lr = lr

        self.layers = []

        # -- Will update first layer's num_inputs when data is provided
        num_inputs = in_features

        for num_neurons in layers:
            self.layers.append(Layer(num_neurons, num_inputs, activation))
            # -- Update num_inputs for next layer
            num_inputs = num_neurons

        # -- Add output layer
        if classes == 2:
            self.layers.append(Layer(num_neurons=1, num_inputs=num_inputs, activation='sigmoid'))
        else:
            self.layers.append(Layer(num_neurons=classes, num_inputs=num_inputs, activation='softmax'))

    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Passes an input array into the network and returns the network's output (0, 1)
        - X: numpy array of a single sample of input data (batch_size, x)
        '''
        # -- Save for backprop
        self.X = X

        for i, layer in enumerate(self.layers):
            Z = np.dot(X, layer.W.T) + layer.B

            A = layer.activation(Z)

            # -- Save weighted sum/output for backprop
            layer.Z = Z
            layer.A = A

            # -- Update X for the next layer
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

        '''
        d_output = dL/da_o * da_o/dz_o * dz_o/dw_i
                 = δ_o * a_i
        '''
        num_samples = y.shape[0]

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
        if self.classes == 2:
            dL = (y_hat - y.reshape(-1, 1))
            delta_o = np.array(dL * o.activation.der(o.Z))
        # -- Multiclass
        else:
            # Convert y to one hot encoding for softmax, e.g. [0 1] for y = 1
            one_hot = np.zeros(shape=(y.shape[0], self.classes))
            one_hot[np.arange(y.shape[0]), y.astype(dtype=int)] = 1
            y = one_hot

            delta_o = y_hat - y

        der_w_o = np.dot(delta_o.T, self.layers[-2].A) / num_samples
        der_b_o = np.sum(delta_o, axis=0) / num_samples

        delta.insert(0, delta_o)
        der_w.insert(0, der_w_o)
        der_b.insert(0, der_b_o)

        '''
        For each layer j back from the output layer: 
        - delta_j = delta_k * W_k.T * a'(z_j)
        - der_w_j = δ_k * a_i
        '''
        for layer in range(len(self.layers) - 2, -1, -1):
            i = self.layers[layer - 1]
            j = self.layers[layer]
            k = self.layers[layer + 1]

            delta_k = delta[0]

            delta_j = np.dot(delta_k, k.W) * j.activation.der(j.Z)

            der_b_j = np.sum(delta_j) / num_samples

            if layer == 0:
                der_w_j = np.dot(delta_j.T, self.X) / num_samples
            else:
                der_w_j = np.dot(delta_j.T, i.A) / num_samples

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
        for epoch in range(epochs):
            for i, (X, y) in enumerate(train_loader):

                y_hat = self.forward(X)
                self.backward(y, y_hat)

            if epoch % 100 == 0:
                loss = MeanSquaredError(y_hat, y, self.classes)
                print(f"Epoch {epoch}, Loss: {loss}")
                self.test(test_loader)


    def test(self, loader: DataLoader):
        total_correct = 0
        total_samples = 0

        for i, (X, y) in enumerate(loader):
            y_hat = self.forward(X)

            if self.classes == 2:
                y_hat_class = np.round(y_hat)
            else:
                y_hat_class = np.argmax(y_hat, axis=1)

            correct = np.sum(y_hat_class.T == y)
            total_correct += correct
            total_samples += len(y)

        print(f"Accuracy: {total_correct / total_samples * 100:.2f}%; {total_correct} correct, {total_samples - total_correct} wrong")
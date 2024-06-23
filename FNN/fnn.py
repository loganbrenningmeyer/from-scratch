'''
Feedforward Neural Network (vanilla NN)

Classes:
- Neuron
- NeuralNetwork
- Layer
'''
import numpy as np
import math

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

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.output(z)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def d_sigmoid(z: np.ndarray) -> np.ndarray:
        return Activation.sigmoid(z) * (1 - Activation.sigmoid(z))

    @staticmethod
    def d_relu(z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1, 0)

class DataLoader:
    def __init__(self, dataset: np.ndarray, batch_size: int = 8, shuffle: bool = True):
        '''
        - dataset: Numpy array of X, y sample/target pairs
        - batch_size: Size of a training batch within the dataset
        - train_size: Proportion of the full dataset allocated to training
        '''
        self.current_batch = 0

        if shuffle:
            np.random.shuffle(X)

        #print(dataset)
        #print(dataset.shape)

        # -- Split into batches
        self.dataset = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch < len(self.dataset):
            batch = self.dataset[self.current_batch]
            X = np.array([x for x, y in batch], dtype=np.float32)
            y = np.array([y for x, y in batch], dtype=np.double)
            self.current_batch += 1
            return X, y
        else:
            raise StopIteration

'''
A singular neuron in the network with its own weights and bias
- num_inputs: Number of input connections into the neuron (# neurons in previous layer)
- activation: String defining which activation function to use
'''
class Neuron:
    def __init__(self, num_inputs: int, activation: str):
        '''
        Randomly initialize weights/bias
        '''
        # self.weights = np.random.randn(num_inputs)
        # self.bias = np.random.randn()

        # self.weights = np.empty(shape=())

        # self.a = np.array([])
        # self.z = np.array([])

        if activation == 'sigmoid':
            self.activation = Activation('sigmoid')
        elif activation == 'relu':
            self.activation = Activation('relu')
        else:
            print("Invalid activation function")

    def activate(self, z: float) -> float:
        a = self.activation(z)
        self.a = np.append(self.a, a)
        return a

    def weighted_sum(self, X: np.ndarray) -> float:
        z = np.dot(X, self.weights) + self.bias
        self.z = np.append(self.z, z)
        return z
        
'''
An array of neurons representing a layer in the network
- num_neurons: Number of neurons in the layer being created
- num_inputs: Number of neurons in the previous layer
'''
class Layer:
    def __init__(self, num_neurons: int, num_inputs: int, activation: str):
        self.neurons = [Neuron(num_inputs, activation) for _ in range(num_neurons)]

        # -- Arrays to store weights/biases for each neuron in the layer
        self.W = np.random.normal(size=(num_neurons, num_inputs))
        self.B = np.random.normal(size=num_neurons)
        # -- Store each neuron's weighted sum (z) and output (a)
        # self.Z = np.empty(shape=num_neurons)
        # self.A = np.empty(shape=num_neurons)
        if activation == 'sigmoid':
            self.activation = Activation('sigmoid')
        elif activation == 'relu':
            self.activation = Activation('relu')
        else:
            print("Invalid activation function")

    # def weighted_sums(self):
    #     return np.array([n.z for n in self.neurons])

    # def outputs(self):
    #     return np.array([n.a for n in self.neurons])

'''
Network built as an array of layers
- layers: Array describing the number of neurons in each layer
  i.e. [1, 2, 3] means there are 3 hidden layers, with 1, 2, and 3 neurons respectively
'''
class NeuralNetwork:
    def __init__(self, layers: list[int], activation: str, lr: float = 0.01):

        self.lr = lr

        self.layers = []

        # -- Will update first layer's num_inputs when data is provided
        num_inputs = 2

        for num_neurons in layers:
            self.layers.append(Layer(num_neurons, num_inputs, activation))
            # -- Update num_inputs for next layer
            num_inputs = num_neurons

        # -- Add output layer
        self.layers.append(Layer(num_neurons=1, num_inputs=num_inputs, activation='sigmoid'))

    def forward(self, X: np.ndarray) -> float:
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
        print(f"\nLoss: {np.average(y_hat - y)}")

        delta = []
        der_w = []

        # -- Output layer
        delta_o = np.array((y_hat - y) * (y_hat * (1 - y_hat))).reshape(-1, 1)

        der_w_o = np.einsum('is,so->soi', self.layers[-2].A.T, delta_o)

        delta.append(delta_o)
        der_w.append(der_w_o)

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

            if layer == 0:
                der_w_j = np.einsum('is,sj->sji', self.X.T, delta_j)
            else:
                der_w_j = np.einsum('is,sj->sji', i.A.T, delta_j)

            delta.insert(0, delta_j)
            der_w.insert(0, der_w_j)

        '''
        Update weights
        - Average the weight derivative among the batch samples for each weight
        '''
        for i, layer in enumerate(self.layers):
            
            batch_der_w = np.average(der_w[i], axis=0)
            batch_der_b = np.average(delta[i], axis=0)

            layer.W = layer.W - self.lr * batch_der_w
            layer.B = layer.B - self.lr * batch_der_b

    def train(self, loader: DataLoader, epochs=100):
        for _ in range(epochs):
            for X, y in loader:

                y_hat = self.forward(X)
                self.backward(y, y_hat)

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

dataset = np.array([(x, yi) for x, yi in zip(X, y)], dtype=object)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

nn = NeuralNetwork(layers=[2, 2], activation='relu', lr=0.1)
nn.train(loader, epochs=10000)

def evaluate_nn(nn, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        output = nn.forward(np.array([xi]))
        predicted = 1 if output >= 0.5 else 0
        print(f"Input: {xi}, Predicted: {predicted}, Actual: {yi[0]}")
        if predicted == yi[0]:
            correct += 1
    accuracy = correct / len(X)
    print(f"Accuracy: {accuracy * 100}%")

evaluate_nn(nn, X, y)

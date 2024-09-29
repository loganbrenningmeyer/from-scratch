import numpy as np
import torch
import torch.nn.functional as F

from utils.activation import Activation

from models.neural_networks.fnn.fnn import NeuralNetwork
'''
CNN Architecture:

Input: 
- X: Images of shape (width, height, channels)
- Ensure images are of fixed size, if not, resize and center crop to desired size

Convolutional Layer:
- Apply kernels to the inputs to extract features
- Activation function: Apply ReLU to feature maps

Pooling Layer:
- Max pooling/average pooling to reduce dimensionality of feature maps

Flattening:
- Convert 3D feature maps to 1D vectors

Fully Connected Layer:
- Input the 1D vectors 
- Output predicted classification

Backpropagation:
- Backprop through FCL
- Backprop through pooling/convolutional layers
'''
class ConvNeuralNetwork:
    def __init__(self):
        '''
        CNN built as an array of layers

        - Initialize empty list of layers
        '''
        # -- Array to hold ConvLayer/PoolLayer objects
        self.layers = []

    def add_conv(self, num_filters, filter_size):
        '''
        Appends a new ConvLayer to the layers array
        
        Parameters:
        - num_filters: # of filters in the ConvLayer
        - filter_size: size k of each filter of shape (k, k)
        
        Determines in_channels based on the out_channels of the previous layer
        '''
        # -- Determine in_channels based on previous layers
        if len(self.layers) != 0:
            # Previous PoolLayer (out_channels = in_channels)
            if type(self.layers[-1]) == PoolLayer:
                in_channels = self.layers[-1].in_channels
            # Previous ConvLayer (out_channels = num_filters)
            elif type(self.layers[-1]) == ConvLayer:
                in_channels = self.layers[-1].num_filters
        else:
            # -- First ConvLayer, so 3 channels for RGB
            in_channels = 3

        self.layers.append(ConvLayer(in_channels, num_filters, filter_size))

    def add_pool(self, method, filter_size):
        '''
        Appends a new PoolLayer to the layers array

        Parameters:
        - method: Pooling method ('max' or 'avg')
        - filter_size: size k of pooling filter of shape (k, k)
        '''
        # -- Determine in_channels based on previous layers
        if len(self.layers) != 0:
            # Previous PoolLayer (out_channels = in_channels)
            if type(self.layers[-1]) == PoolLayer:
                in_channels = self.layers[-1].in_channels
            # Previous ConvLayer (out_channels = num_filters)
            elif type(self.layers[-1]) == ConvLayer:
                in_channels = self.layers[-1].num_filters
        else:
            # -- First ConvLayer, so 3 channels for RGB
            in_channels = 3

        self.layers.append(PoolLayer(in_channels, method, filter_size))

    def add_fcl(self, sample_shape, 
                      layers,
                      num_classes,
                      activation='relu',
                      lr=0.01):
        '''
        Initializes a FCL for the final decision-making
        layer of the CNN
        '''
        # -- Get sample output shape
        sample = np.ones(shape=sample_shape)
        output_shape = self.extract_features(sample).shape

        # -- Determine # input features based on final output shape (channels * x * y)
        in_features = output_shape[1] * output_shape[2] * output_shape[3]

        # -- Initialize FNN
        self.fnn = NeuralNetwork(layers=layers,
                                 activation=activation,
                                 in_features=in_features,
                                 num_classes=num_classes,
                                 lr=lr)

    def extract_features(self, X):
        '''
        Pass input X through each layer of the Convolutional Network
        '''
        # -- Pass through ConvLayers and PoolLayers
        for layer in self.layers:
            output = layer.forward(X)

            X = output

        return X
    
    def classify(self, X):
        '''
        Pass extracted features X through the FCL to classify
        '''
        # -- Pass through FCL
        return self.fnn.forward(X)
    
    def forward(self, X):
        '''
        Pass input X through both extract_features (ConvLayers/PoolLayers)
        and classify (FCL)
        '''
        # -- Extract features
        features = self.extract_features(X)

        # -- Flatten features (preserve batch dimension)
        features = np.reshape(features, (features.shape[0], -1))

        # -- Feed into FCL and return decision
        return self.classify(features)


class ConvLayer:
    def __init__(self, in_channels, num_filters, filter_size):
        '''
        Randomly initializes kernels with a Gaussian distribution

        Given num_filters (N_k), filter_size (k), and number of color channels (3),
        the shape of the kernels for a layer is:
        - (N_k, 3, k, k)

        i.e. filters[i] = ith kernel
             filters[i][j] = ith kernel, jth channel
        '''
        # -- Set attributes
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_size = filter_size

        # -- He initialization of filters with shape (N_k, 3, k, k)
        self.filters = np.random.normal(loc=0, 
                                        scale=np.sqrt(2.0 / (3 * filter_size**2)), 
                                        size=(num_filters, in_channels, filter_size, filter_size))
        
        # -- Zero initialization of biases
        self.bias = np.zeros(num_filters)

        # -- ReLU activation
        self.activation = Activation('relu')

    def forward(self, X):
        '''
        Given some input batch of matrices X with shape (batch_size, m, n), 
        convolve the filters over the image return N_k feature maps
        '''
        # -- Convolve the filters over the image X and add bias
        Z = np.array(F.conv2d(input=torch.from_numpy(X), 
                              weight=torch.from_numpy(self.filters), 
                              bias=torch.from_numpy(self.bias),
                              padding=1))

        # -- Apply ReLU activation
        A = self.activation(Z)

        self.Z = Z
        self.A = A

        return A
    
class PoolLayer:
    def __init__(self, in_channels, method, filter_size):
        self.in_channels = in_channels
        self.method = method
        self.filter_size = filter_size

    def forward(self, X):
        # -- Apply pooling (max or avg)
        if self.method == 'max':
            P = np.array(F.max_pool2d(input=torch.from_numpy(X),
                                           kernel_size=self.filter_size,
                                           stride=self.filter_size))
        else:
            P = np.array(F.avg_pool2d(input=torch.from_numpy(X),
                                           kernel_size=self.filter_size,
                                           stride=self.filter_size))
        
        self.P = P

        return P


# if __name__ == "__main__":
#     img = np.ones(shape=(2, 3, 100, 100))

#     layer = ConvLayer(32, 3)
#     maps = layer.forward(img)
#     print(f"maps.shape: {maps.shape}")

#     pool = PoolLayer('max', 2)
#     P = pool.forward(maps)
#     print(f"P.shape: {P.shape}")

#     layer2 = ConvLayer(64, 3)
#     maps2 = layer2.forward()
#     print(f"maps2.shape: {maps2.shape}")


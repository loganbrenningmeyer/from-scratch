import numpy as np
import torch
import torch.nn.functional as F

from utils.activation import Activation
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
class ConvLayer:
    def __init__(self, num_filters, filter_size):
        '''
        Randomly initializes kernels with a Gaussian distribution

        Given num_filters (N_k), filter_size (k), and number of color channels (3),
        the shape of the kernels for a layer is:
        - (N_k, 3, k, k)

        i.e. filters[i] = ith kernel
             filters[i][j] = ith kernel, jth channel
        '''
        # -- He initialization of filters with shape (N_k, 3, k, k)
        self.filters = np.random.normal(loc=0, 
                                        scale=np.sqrt(2.0 / (3 * filter_size**2)), 
                                        size=(num_filters, 3, filter_size, filter_size))
        
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
    def __init__(self, type, filter_size):
        if type == 'max':
            self.pool = F.max_pool2d()
        elif type == 'avg':
            self.pool = F.avg_pool2d()

        self.filter_size = filter_size

    def forward(self, X):
        # -- Apply pooling 
        self.P = np.array(self.pool(input=torch.from_numpy(X),
                                    kernel_size=self.filter_size,
                                    stride=self.filter_size))
        
        return self.P


if __name__ == "__main__":
    img = np.ones(shape=(2, 3, 100, 100))
    layer = ConvLayer(32, 3)
    maps = layer.forward(img)
    print(f"maps.shape: {maps.shape}")
    pool = PoolLayer('max', 2)
    P = pool.forward(maps)
    print(f"P.shape: {P.shape}")


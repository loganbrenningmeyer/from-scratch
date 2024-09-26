import numpy as np

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
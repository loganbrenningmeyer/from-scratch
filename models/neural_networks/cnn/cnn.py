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

def convolution(X: np.ndarray, kernel: np.ndarray, padding: str, stride: int) -> np.ndarray:
    '''
    Create new convolved X array
    - New size: 
        * H_2 = (H_1 - K + 2*P) / S + 1
        * W_2 = (W_1 - K + 2*P) / S + 1
    '''
    pad = int((kernel.shape[0] - 1) / 2)

    X_conv = np.zeros(shape=((X.shape[0] - kernel.shape[0] + 2 * pad) / stride + 1, 
                             (X.shape[1] - kernel.shape[1] + 2 * pad) / stride + 1))
    '''
    Pad X depending on padding type
    - full: Append 0s all around X
    - same: Append 0s on the left and top of X
    - valid: No padding applied

    kernel: nxn (odd-dimensions) square matrix
    '''
    if padding == 'same':
        # Calculate # of pixels to pad
        pad = int((kernel.shape[0] - 1) / 2)
        # -- no pad for channel axis itself, pad on each side for each image axis
        X = np.pad(X, ((0, 0), (pad, pad), (pad, pad)))

    # -- Convolve
    '''
    Use slicing to pass the kernel over sections of X
    - X_ij = X[i : i + k, j : j + k]
        * i,j is the upper left index of the section
        * k is the size of the kernel
    
    Then perform in-place multiplication with the kernel (section & kernel same dimensions)
    and np.sum the result

    X: 5x5
    k: 3x3
    i [0, 1, 2]
    '''

    for i in range(X.shape[0] - kernel.shape[0] + 1):
        for j in range(X.shape[1] - kernel.shape[1] + 1):
            # Slice X_ij
            X_ij = X[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            # Multiply X and k
            X_conv[i, j] = np.dot(X_ij, kernel)
            print(f"X_conv: {X_conv}")
            

    for c in range(X.shape[0]):
        X[c] = np.convolve(X[c], kernel)

    print(X)
    
if __name__ == "__main__":
    # X.shape: (channels, height, width)
    X = np.array([[[1, 2, 3, 4 ,5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25]]] * 3)
    
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    
    convolution(X, kernel, 'same', 1)
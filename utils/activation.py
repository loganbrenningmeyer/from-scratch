import numpy as np

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
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True) + 1e-15
    
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
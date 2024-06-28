import numpy as np
from fnn import Layer, NeuralNetwork

'''
Input data
'''
X = np.array([[0, 0],
              [1, 1],
              [2, 2]])

y = np.array([0, 1, 2])

num_samples = X.shape[0]
input_size = X.shape[1]
classes = 3

input_layer = Layer(num_neurons=2, num_inputs=input_size, activation='relu')
output_layer = Layer(num_neurons=classes, num_inputs=2, activation='softmax')

print(input_layer.W)
print(input_layer.B)
print(f"input_layer.W.shape: {input_layer.W.shape}")
print(f"input_layer.B.shape: {input_layer.B.shape}")

print(output_layer.W)
print(output_layer.B)
print(f"output_layer.W.shape: {output_layer.W.shape}")
print(f"output_layer.B.shape: {output_layer.B.shape}")

'''
Forward pass
'''
Z_i = np.dot(X, input_layer.W.T)
print(f"Z_i after X * W.T: {Z_i}, shape: {Z_i.shape}")

Z_i = Z_i + input_layer.B
print(f"Z_i after X * W.T + B: {Z_i}, shape: {Z_i.shape}")

A_i = input_layer.activation(Z_i)
print(f"A_i: {A_i}, shape: {A_i.shape}")

Z_o = np.dot(A_i, output_layer.W.T)
print(f"Z_o after X * W.T: {Z_o}, shape: {Z_o.shape}")

Z_o = Z_o + output_layer.B
print(f"Z_o after X * W.T + B: {Z_o}, shape: {Z_o.shape}")

A_o = output_layer.activation(Z_o)
print(f"A_o: {A_o}, shape: {A_o.shape}")

one_hot = np.zeros(shape=(y.shape[0], classes))
one_hot[np.arange(y.shape[0]), y.astype(dtype=int)] = 1
y = one_hot
print(f"y: {y}, shape: {y.shape}")

delta_o = A_o - y
print(f"dZ_o: {delta_o}, shape: {delta_o.shape}")

dW_o = np.dot(delta_o.T, A_i)
print(f"dW_o: {dW_o}, shape: {dW_o.shape}")

dW_o /= num_samples
print(f"dW_o avg: {dW_o}, shape: {dW_o.shape}")

import tensorflow as tf
import numpy as np
import pandas as pd

from models.neural_networks.fnn.fnn import NeuralNetwork
from utils.data import Dataset, DataLoader

from sklearn.datasets import load_iris
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data_id = 3

    if data_id == 0:
        iris = load_iris()
        X = iris.data
        y = iris.target
        # y = [0 if yi == 0 else 1 for yi in y]
        classes = 3
    elif data_id == 1:
        csv = pd.read_csv('data/student_performance_data.csv')
        data = np.array(csv)
        X = data[:,:-1]
        y = data.T[-1]
        # y = np.array([0 if yi < 3 else 1 for yi in y])
        classes = 5
    elif data_id == 2:
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        X = np.concatenate((x_train, x_test), axis=0)
        X = np.array([x.flatten() for x in X])
        y = np.concatenate((y_train, y_test), axis=0)
        # y = [0 if yi < 3 else 1 if yi < 6 else 2 for yi in y]
        classes = 10
    elif data_id == 3:
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        X = np.concatenate((x_train, x_test), axis=0)
        X = np.array([x.flatten() for x in X])
        y = np.concatenate((y_train, y_test), axis=0)
        classes = 10
    dataset = Dataset(X, y)

    train_loader = DataLoader(dataset.train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset.test_dataset, batch_size=1, shuffle=False)

    model = NeuralNetwork(layers=[128, 64, 32],
                          activation='leaky',
                          in_features=X.shape[1],
                          num_classes=classes,
                          lr=0.01)
    
    model.train(train_loader, test_loader, epochs=10)

    accuracy, correct, incorrect = model.test(test_loader)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    model.save('model')

    loaded_model = model.load('models/fnn/saved_models/model.json')

    print("Testing loaded model:")
    loaded_accuracy, _, _ = loaded_model.test(test_loader)
    print(f"Loaded test accuracy: {loaded_accuracy * 100:.2f}%")
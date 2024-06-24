import tensorflow as tf
import numpy as np
from fnn import NeuralNetwork, DataLoader, Activation

if __name__ == "__main__":

    # softmax = Activation('softmax')
    # Z = np.array([2.0, 1.0, 0.1])
    # print(softmax(Z))

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(y_train.shape)

    in_features = len(x_train[0].flatten())
    classes = 10

    print(f"in_features: {in_features}")
    print(f"classes: {classes}")

    # -- Filter the data for binary classification (digits 0 and 1)
    # train_filter = np.where((y_train == 0) | (y_train == 1))
    # test_filter = np.where((y_test == 0) | (y_test == 1))

    # x_train, y_train = x_train[train_filter], y_train[train_filter]
    # x_test, y_test = x_test[test_filter], y_test[test_filter]

    # -- Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_dataset = np.array([(x.flatten(), yi) for x, yi in zip(x_train, y_train)], dtype=object)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    test_dataset = np.array([(x.flatten(), yi) for x, yi in zip(x_train, y_train)], dtype=object)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    # -- Initialize neural network
    model = NeuralNetwork(layers=[128, 64, 32], activation='relu', lr=0.1, in_features=in_features, classes=classes)

    # -- Train the network
    model.train(train_loader, epochs=5)
    model.test(test_loader)
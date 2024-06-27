import tensorflow as tf
import numpy as np
from fnn import NeuralNetwork, DataLoader, Dataset, Activation
import pandas as pd

from sklearn.datasets import load_iris

if __name__ == "__main__":

    data_id = 2

    if data_id == 0:
        iris = load_iris()
        X = iris.data
        y = iris.target
        y = [0 if yi == 0 else 1 for yi in y]
        classes = 2
    elif data_id == 1:
        csv = pd.read_csv('student_performance_data.csv')
        data = np.array(csv)
        X = data[:,:-1]
        y = data.T[-1]
        y = np.array([0 if yi < 3 else 1 for yi in y])
        classes = 2
    elif data_id == 2:
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        X = np.concatenate((x_train, x_test), axis=0) / 255.0
        X = np.array([x.flatten() for x in X])
        y = np.concatenate((y_train, y_test), axis=0) / 255.0
        y = [0 if yi < 5 else 1 for yi in y]
        classes = 2

    dataset = Dataset(X, y)

    train_loader = DataLoader(dataset.train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset.test_dataset, batch_size=1, shuffle=False)

    model = NeuralNetwork(layers=[64, 32],
                          activation='relu',
                          in_features=X.shape[1],
                          classes=classes,
                          lr=0.001)
    
    model.train(train_loader, test_loader, epochs=10000)

    model.test(test_loader)

    # csv = pd.read_csv('student_performance_data.csv')
    # data = np.array(csv)
    # print(data)
    # # print(data.shape)
    # X = data[:,:-1]
    # y = data.T[-1]
    # # print(y)
    # # print(y.shape)
    
    # dataset = np.array([(x, yi) for x, yi in zip(X, y)], dtype=object)
    # # print(dataset.shape)
    # np.random.shuffle(dataset)

    # train_size = int(0.8*len(dataset))

    # train_dataset = dataset[:train_size]
    # test_dataset = dataset[train_size:]

    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model = NeuralNetwork(layers=[64, 32], activation='relu', in_features=14, classes=5, lr=0.0001)

    # model.train(train_loader, test_loader, epochs=10000)

    # model.test(test_loader)

    # print(dataset)

    # softmax = Activation('softmax')
    # Z = np.array([2.0, 1.0, 0.1])
    # print(softmax(Z))

    # mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print(x_train.shape)
    # print(y_train.shape)

    # in_features = len(x_train[0].flatten())
    # classes = 2

    # print(f"in_features: {in_features}")
    # print(f"classes: {classes}")

    # # -- Filter the data for binary classification (digits 0 and 1)
    # y_train = [0 if y_i < 5 else 1 for y_i in y_train]
    # y_test  = [0 if y_i < 5 else 1 for y_i in y_test]
    # # train_filter = np.where((y_train == 0) | (y_train == 1))
    # # test_filter = np.where((y_test == 0) | (y_test == 1))

    # # x_train, y_train = x_train[train_filter], y_train[train_filter]
    # # x_test, y_test = x_test[test_filter], y_test[test_filter]

    # # -- Normalize the data
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    # train_dataset = np.array([(x.flatten(), yi) for x, yi in zip(x_train, y_train)], dtype=object)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    # test_dataset = np.array([(x.flatten(), yi) for x, yi in zip(x_test, y_test)], dtype=object)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

    # # -- Initialize neural network
    # model = NeuralNetwork(layers=[64, 32], activation='relu', lr=0.001, in_features=in_features, classes=classes)

    # # -- Train the network
    # model.train(train_loader, test_loader, epochs=50)
    # print("Test:")
    # model.test(test_loader)
    # print("Train:")
    # model.test(train_loader)
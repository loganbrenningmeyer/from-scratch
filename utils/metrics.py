import numpy as np

def CrossEntropyLoss(y_hat: np.ndarray, y: np.ndarray) -> float:
    '''
    Computes the cross entropy loss of a batch for both binary/multiclass

    Parameters:
    - y_hat (np.ndarray): Model's predicted probabilities
        * Binary: shape (s, 1) where s is the # of samples
        * Multiclass: shape (s, c) where c is the # of classes
    - y (np.ndarray): Target labels 
        * Binary: shape (s, )
        * Multiclass: shape (s, )

    Returns:
    - loss (float)
    '''
    output_size = y_hat.shape[1]

    # -- Clip to avoid log(0)
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    if output_size == 1:
        y_hat = y_hat.reshape(-1, )
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    else:
        # -- One hot encode y
        y = np.eye(output_size)[y.astype(int)]
        loss = -np.mean(np.sum(y * np.log(y_hat), axis=1))

    return loss

def MeanSquaredError(y_hat: np.ndarray, y: np.ndarray) -> float:
    '''
    Computes mean squared error loss of a batch for both binary/multiclass

    Parameters:
    - y_hat (np.ndarray): Model's predicted probabilities
        * Binary: shape (s, 1) where s is the # of samples
        * Multiclass: shape (s, c) where c is the # of classes
    - y (np.ndarray): Target labels 
        * Binary: shape (s, )
        * Multiclass: shape (s, )

    Returns:
    - loss (float)
    '''
    output_size = y_hat.shape[1]
    
    if output_size == 1:
        y_hat = y_hat.reshape(-1, )
        loss = np.mean((y - y_hat) ** 2)
    else:
        # -- One hot encode y
        y = np.eye(output_size)[y.astype(int)]
        loss = np.mean((y - y_hat) ** 2)

    return loss

def Accuracy(y_hat: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    output_size = y_hat.shape[1]

    if output_size == 1:
        y_hat = y_hat.reshape(-1, )
        y_hat = np.round(y_hat)
    else:
        y_hat = np.argmax(y_hat, axis=1)

    correct = np.sum(y_hat == y)
    incorrect = len(y) - correct
    accuracy = correct / len(y)

    return accuracy, correct, incorrect

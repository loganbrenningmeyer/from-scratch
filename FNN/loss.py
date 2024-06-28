import numpy as np


def CrossEntropyLoss(y_hat: np.ndarray, y: np.ndarray, num_classes: int) -> float:
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
    # -- Clip to avoid log(0)
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    if num_classes == 2:
        y_hat = y_hat.reshape(-1, )
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    else:
        batch_size = y.shape[0]
        loss = -np.mean(np.sum(y * np.log(y_hat[np.arange(batch_size), y.astype(int)])))

    return loss

def MeanSquaredError(y_hat: np.ndarray, y: np.ndarray, num_classes: int) -> float:
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
    if num_classes == 2:
        y_hat = y_hat.reshape(-1, )
        loss = np.mean((y - y_hat) ** 2)
    else:
        # -- One hot encode y
        y = np.eye(num_classes)[y.astype(int)]
        loss = np.mean((y - y_hat) ** 2)

    return loss


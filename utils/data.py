import numpy as np

class Dataset:
    def __init__(self, X, y, train_ratio=0.8):
        # -- Normalize X
        # X.shape = (S, c)
        X = (X - np.min(X, axis=0)) / ((np.max(X, axis=0) - np.min(X, axis=0)) + 1e-8)

        dataset = np.array([(x, yi) for x, yi in zip(X, y)], dtype=object)
        np.random.shuffle(dataset)

        train_size = int(0.8*len(dataset))

        self.train_dataset = dataset[:train_size]
        self.test_dataset = dataset[train_size:]
        

class DataLoader:
    def __init__(self, dataset: np.ndarray, batch_size: int = 8, shuffle: bool = True):
        '''
        - dataset: Numpy array of X, y sample/target pairs
        - batch_size: Size of a training batch within the dataset
        - train_size: Proportion of the full dataset allocated to training
        '''
        self.current_batch = 0

        if shuffle:
            np.random.shuffle(dataset)

        # -- Split into batches
        self.dataset = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch < len(self.dataset):
            batch = self.dataset[self.current_batch]
            X = np.array([x for x, y in batch], dtype=np.float32)
            y = np.array([y for x, y in batch], dtype=np.double)
            self.current_batch += 1
            return X, y
        else:
            raise StopIteration
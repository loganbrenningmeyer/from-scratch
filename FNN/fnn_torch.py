import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import numpy as np
import tensorflow as tf

class BasicNN(L.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(BasicNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Linear(32, output_dim),
            nn.Softmax()
        )

    def forward(self, X):
        a1 = self.layer1(X)
        a2 = self.layer2(a1)
        a3 = self.layer3(a2)
        return self.output(a3)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.calculate_accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.calculate_accuracy(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)

    def calculate_accuracy(self, y_hat, y):
        _, predicted = torch.max(y_hat, dim=1)
        correct = (predicted == y).float()
        accuracy = correct.sum() / len(correct)
        return accuracy

class StudentPerformanceDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class MnistDataset(Dataset):
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        X = np.concatenate((x_train, x_test), axis=0) / 255.0
        self.X = torch.tensor([x.flatten() for x in X])
        y = torch.tensor(np.concatenate((y_train, y_test), axis=0) / 255.0
    
def create_dataloaders(csv_file, batch_size=32, split_ratio=0.8):
    dataset = StudentPerformanceDataset(csv_file)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    # csv_file = 'student_performance_data.csv'
    # train_loader, val_loader = create_dataloaders(csv_file, batch_size=32)

    
    # y = [0 if yi < 5 else 1 for yi in y]
    classes = 10

    dataset = Dataset(X, y)

    train_loader = DataLoader(dataset.train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset.test_dataset, batch_size=1, shuffle=False)

    model = BasicNN(input_dim=784, output_dim=classes)

    logger = CSVLogger("logs", name="my_model")

    # Create a trainer with the logger
    trainer = L.Trainer(max_epochs=100, logger=logger)
    trainer.fit(model, train_loader, test_loader)

    # After training, print the final accuracy
    train_accuracy = trainer.callback_metrics['train_accuracy_epoch']
    val_accuracy = trainer.callback_metrics['val_accuracy_epoch']
    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    # Print the path to the log file
    print(f"Log file is saved at: {logger.log_dir}")
    
    # model.train(train_loader, test_loader, epochs=1000)

    # model.test(test_loader)

    # input_dim = train_loader.dataset[0][0].shape[0]
    # output_dim = len(torch.unique(train_loader.dataset.dataset.y))

    # model = BasicNN(input_dim, output_dim)

    # # Create a logger
    # logger = CSVLogger("logs", name="my_model")

    # # Create a trainer with the logger
    # trainer = L.Trainer(max_epochs=100, logger=logger)
    # trainer.fit(model, train_loader, val_loader)

    # # After training, print the final accuracy
    # train_accuracy = trainer.callback_metrics['train_accuracy_epoch']
    # val_accuracy = trainer.callback_metrics['val_accuracy_epoch']
    # print(f"Final Training Accuracy: {train_accuracy:.4f}")
    # print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    # # Print the path to the log file
    # print(f"Log file is saved at: {logger.log_dir}")
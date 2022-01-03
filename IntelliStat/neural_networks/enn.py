from typing import Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from IntelliStat.neural_networks import BaseNeuralNetwork


class ENN(BaseNeuralNetwork):
    def __init__(self, n_in: int, n_hidden1: int, n_hidden2: int, n_hidden3: int, n_out: int, learning_rate: float):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden1, n_hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden2, n_hidden3),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden3, n_out)
        )
        self.optimizer = Adam(self.model.parameters(), learning_rate)
        self.loss_vector = []

    def train(self, dataset: Dataset, epochs: int, batch: Optional[int]):
        data_train = DataLoader(dataset=dataset, batch_size=batch, shuffle=True)
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print("Training epoch: ", epoch)
            for batch in data_train:
                x_train, y_train = batch['input'], batch['output']
                self.train_step(x_train, y_train, criterion)
            loss = criterion(self.model(x_train.float()), y_train.float())
            self.loss_vector.append(loss.item())

    def train_step(self, x, y, criterion):
        self.model.zero_grad()
        x = x.float()
        y = y.float()

        output = self.model(x)
        loss = criterion(output, y)
        loss.backward()
        self.optimizer.step()

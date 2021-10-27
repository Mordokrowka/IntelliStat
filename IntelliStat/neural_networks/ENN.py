from typing import Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


class ENN:
    def __init__(self, n_in: int, n_hidden1: int, n_hidden2: int, n_hidden3: int, n_out: int):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden1, n_hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden2, n_hidden3),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden3, n_out)
        )
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss_vector = []

    def train_step(self, x, y, criterion):
        self.model.zero_grad()
        x = x.float()
        y = y.float()

        output = self.model(x)
        loss = criterion(output, y)
        loss.backward()
        self.optimizer.step()

    def train(self, data: Dataset, epochs: int, batch: Optional[int]):
        data_train = DataLoader(dataset=data, batch_size=batch, shuffle=True)
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print("Training epoch: ", epoch)
            for dummy, batch in enumerate(data_train):
                x_train, y_train = batch['input'], batch['output']
                self.train_step(x_train, y_train, criterion)
            loss = criterion(self.model(x_train.float()), y_train.float())
            self.loss_vector.append(loss.item())

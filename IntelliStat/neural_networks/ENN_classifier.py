import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


class ENN_Classifier:
    def __init__(self, n_in, n_hidden1, n_hidden2, n_hidden3, n_out):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden1, n_hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden2, n_hidden3),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden3, n_out),
            torch.nn.Softmax(dim=1)
        )
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss_vector = []

    def train_step(self, x, y, criterion):
        self.model.zero_grad()

        x = x.float()

        output = self.model(x)
        loss = criterion(output, y)
        loss.backward()
        self.optimizer.step()

    def train(self, data, epochs, batch):
        data_train = DataLoader(dataset=data, batch_size=batch, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print("Training epoch: ", epoch)
            for dummy, batch in enumerate(data_train):
                x_train, y_train = batch['input'], batch['output']
                self.train_step(x_train, y_train, criterion)
            loss = criterion(self.model(x_train.float()), y_train)
            self.loss_vector.append(loss.item())

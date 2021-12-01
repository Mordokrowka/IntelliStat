from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseNeuralNetwork(ABC):
    def __init__(self):
        self.loss_vector: Optional[list] = None
        self.model = None

    @abstractmethod
    def train(self, data: Dataset, epochs: int, batch: Optional[int]):
        """Trains neural network

        :param data: due to the PyTorch requirements torch.utils.data.Dataset needs to be used as a data provider
        :param epochs: numbers of training epochs
        :param batch: size of batch
        """
        ...

    @abstractmethod
    def train_step(self, x, y, criterion):
        ...

    def test(self, test_data: np.ndarray) -> np.ndarray:
        """Tests trained neural network

        :param test_data: data to test neural network model
        :return: result of neural network model work
        """
        X_NN = torch.tensor(test_data)
        Y_NN = self.model(X_NN)
        return Y_NN.detach().numpy()

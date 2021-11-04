from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseNeuralNetwork(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, data: Dataset, epochs: int, batch: Optional[int]):
        ...

    @abstractmethod
    def train_step(self, x, y, criterion):
        ...

    def test(self, X_test) -> np.ndarray:
        X_NN = torch.tensor(X_test)
        Y_NN = self.model(X_NN)
        return Y_NN.detach().numpy()

import numpy as np
import math

from IntelliStat.datasets.base_dataset import BaseDataset


def Gauss(x, A, u, sigma):
    return A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))


class ShapeDataset(BaseDataset):
    def __init__(self, X, function: str, Y):

        super().__init__(X, Y)

        if function == "Gauss":
            for i in range(len(self.X)):
                for j in range(len(self.X[0])):
                    self.X[i][j] = Gauss(self.X[i][j], 1, self.Y[i][0], self.Y[i][1])
        if function == "Gauss+Gauss":
            for i in range(len(self.X)):
                for j in range(len(self.X[0])):
                    self.X[i][j] = Gauss(self.X[i][j], 1, self.Y[i][0], self.Y[i][1]) \
                                   + Gauss(self.X[i][j], 1, self.Y[i][2], self.Y[i][3])
        if function == "Gauss+Gauss+Exp":
            for i in range(len(self.X)):
                for j in range(len(self.X[0])):
                    self.X[i][j] = Gauss(self.X[i][j], 1, self.Y[i][0], self.Y[i][1]) \
                                   + Gauss(self.X[i][j], 1, self.Y[i][2], self.Y[i][3]) \
                                   + math.exp(-0.2 * self.X[i][j])

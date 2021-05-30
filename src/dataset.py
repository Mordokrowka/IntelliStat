import numpy as np

from torch.utils.data import Dataset

from utils import Gauss


# TODO default values for mu, sigma
# TODO check about reshape
# TODO FIX another shapelearnings
class FunctionDataset(Dataset):
    def __init__(self, X_data: np.ndarray):
        self.X = X_data.copy()
        self.Y = np.empty((X_data.shape[0], 0), dtype=np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return {'input': self.X[index],
                'output': self.Y[index]}


class GaussDataset(FunctionDataset):
    def __init__(self, X_data, mu=None, sigma=None, amplitude=None):
        super().__init__(X_data)

        if mu is not None:
            self.Y = np.append(self.Y, mu, 1)
        if sigma is not None:
            self.Y = np.append(self.Y, sigma, 1)
        if amplitude is not None:
            self.Y = np.append(self.Y, amplitude, 1)

        amplitude = amplitude if amplitude is not None else np.ones((X_data.shape[0], 1), dtype=np.float32)

        mu = mu if mu is not None else 0
        sigma = sigma if sigma is not None else 0

        self.X = Gauss(self.X, amplitude, mu, sigma)
        # if function == "Gauss+Gauss+Exp":
        #     for i in range(len(self.X)):
        #         for j in range(len(self.X[0])):
        #             self.X[i][j] = Gauss(self.X[i][j], 1, self.Y[i][0], self.Y[i][1]) \
        #                            + Gauss(self.X[i][j], 1, self.Y[i][2], self.Y[i][3]) \
        #                            + math.exp(-0.2 * self.X[i][j])

#
# class DoubleFunctionDataset(FunctionDataset):
#     def __init__(self, X_data, mu1=None, sigma1=None, amplitude1=None, mu2=None, sigma2=None, amplitude2=None):
#         super().__init__(X_data)
#
#         gauss1 = SingleFunctionDataset(X_data, mu1, sigma1, amplitude1)
#         gauss2 = SingleFunctionDataset(X_data, mu2, sigma2, amplitude2)
#
#         self.X = np.append(gauss1.X, gauss2.X, 1)
#         self.Y = np.append(gauss1.Y, gauss2.Y, 1)


class LinearDataset(FunctionDataset):
    def __init__(self, X_data, params):
        super().__init__(X_data)
        self.Y = self.X * params[0] + params[1]






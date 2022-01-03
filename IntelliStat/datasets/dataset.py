import numpy as np
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return {'input': self.X[index],
                'output': self.Y[index]}

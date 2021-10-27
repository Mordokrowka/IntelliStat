from typing import Optional, Sequence, Union

import numpy as np

from IntelliStat.datasets.base_dataset import BaseDataset


class FunctionDataset(BaseDataset):
    def __init__(self, X, function: str, params: Union[np.ndarray, Sequence]):

        Y = np.zeros(X.shape[0])
        if function == "linear":
            Y = [[self.X[point][0] * params[0] + params[1]] for point in range(X.shape[0])]
        Y = np.array(Y, dtype=np.float32)

        super().__init__(X, Y)

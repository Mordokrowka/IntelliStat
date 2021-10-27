from typing import Optional, Sequence, Union

import numpy as np

from IntelliStat.datasets.base_dataset import BaseDataset


class FunctionDataset(BaseDataset):
    def __init__(self, X, function: str = '', Y: Optional[Union[np.ndarray, Sequence]] = None):
        if Y is None:
            Y = np.zeros(len(X))
            if function == "linear":
                Y = [[self.X[point][0] * Y[0] + Y[1]] for point in range(len(X))]
            Y = np.array(Y, dtype=np.float32)

        super().__init__(X, Y)

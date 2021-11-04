from typing import Sequence, Union

import numpy as np

from .base_dataset import BaseDataset


class FunctionDataset(BaseDataset):
    def __init__(self, X, function: str, params: Union[np.ndarray, Sequence]):

        if function == "linear":
            Y = X * params[0] + params[1]

        super().__init__(X, Y)

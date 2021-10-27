from ..components.component_functions import Gauss, GG, GGE
from .base_dataset import BaseDataset


class ShapeDataset(BaseDataset):
    def __init__(self, X, function: str, Y):

        super().__init__(X, Y)
        if function == "Gauss":
            self.X = Gauss(self.X, 1, Y[:, 0].reshape(-1, 1), Y[:, 1].reshape(-1, 1))
        if function == "Gauss+Gauss":
            self.X = GG(self.X, 1, Y[:, 0].reshape(-1, 1), Y[:, 1].reshape(-1, 1),
                        1, Y[:, 2].reshape(-1, 1), Y[:, 3].reshape(-1, 1))
        if function == "Gauss+Gauss+Exp":
            self.X = GGE(self.X, 1, Y[:, 0].reshape(-1, 1), Y[:, 1].reshape(-1, 1),
                         1, Y[:, 2].reshape(-1, 1), Y[:, 3].reshape(-1, 1), 1, 0.2)

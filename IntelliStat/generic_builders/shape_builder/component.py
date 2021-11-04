from enum import Enum, EnumMeta, auto, unique
from typing import Callable

import numpy as np

from IntelliStat.generic_builders.shape_builder.component_functions import Exp, GE, GGE, multiG


class ComponentEnumMeta(EnumMeta):
    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError(f"`index` {index} is not an int")
        if index >= super().__len__():
            raise KeyError("Key error")

        return list(self)[index]


@unique
class Component(Enum, metaclass=ComponentEnumMeta):
    Gauss = 'Gauss', (1, 0), multiG, 1
    Gauss_Gauss = 'Gauss+Gauss', (2, 0), multiG, 2
    Gauss_Gauss_Gauss = 'Gauss+Gauss+Gauss', (3, 0), multiG, 3
    Gauss_Gauss_Exp = 'Gauss+Gauss+Exp', (2, 1), GGE,
    Gauss_Exp = 'Gauss+Exp', (1, 1), GE,
    Exp = 'Exp', (0, 1), Exp,
    G_4 = '4G', (4, 0), multiG, 4
    G_5 = '5G', (5, 0), multiG, 5
    G_6 = '6G', (6, 0), multiG, 6
    G_7 = '7G', (7, 0), multiG, 7

    def __init__(self, component_name: str, class_vector: tuple, data_generator: Callable,
                 n_gauss: int = None):

        self.component_name: str = component_name
        self.class_vector: np.ndarray = np.array(class_vector)
        self.data_generator: Callable = data_generator
        self.n_gauss: int = n_gauss

    def __new__(cls, *values):
        obj = object.__new__(cls)
        obj._value_ = auto()
        for other_value in values:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values

        return obj

    def __repr__(self):
        return '<%s.%s: %s>' % (
            self.__class__.__name__,
            self._name_,
            ', '.join([str(v) for v in self._all_values]),
        )

    # TODO to change -> one builder
    def generate_data(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if self.data_generator == multiG:
            return self.data_generator(x, n=self.n_gauss, **kwargs)
        return self.data_generator(x, **kwargs)


if __name__ == '__main__':
    print(repr(Component('Gauss')))
    print(repr(Component[0]))
    print(repr(Component.Gauss))

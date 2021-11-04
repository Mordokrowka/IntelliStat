from enum import Enum, EnumMeta, auto, unique
from typing import Callable

import numpy as np

from IntelliStat.generic_builders.shape_builder.component_functions import Exp, Gauss


class ComponentBuilderEnumMeta(EnumMeta):
    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError(f"`index` {index} is not an int")
        if index >= super().__len__():
            raise KeyError("Key error")

        return list(self)[index]


@unique
class ComponentBuilder(Enum, metaclass=ComponentBuilderEnumMeta):
    Gauss = 'Gauss', Gauss
    Exp = 'Exp', Exp,

    def __init__(self, component_name: str, data_generator: Callable):

        self.component_name: str = component_name

        self.data_generator: Callable = data_generator

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

    def generate_component(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.data_generator(x, **kwargs)


if __name__ == '__main__':
    print(repr(ComponentBuilder('Gauss')))
    print(repr(ComponentBuilder[0]))
    print(repr(ComponentBuilder.Gauss))

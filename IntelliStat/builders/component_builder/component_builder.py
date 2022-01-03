from enum import Enum, auto, unique
from typing import Callable

import numpy as np

from IntelliStat.builders.component_builder.components import Exp, Gauss
from IntelliStat.builders.utils.builder_enum_meta import BuilderEnumMeta


@unique
class ComponentBuilder(Enum, metaclass=BuilderEnumMeta):
    """Builds components

    Available components:
        - Gauss
        - Exp

    :param component_name: human name of component
    :param component_generator: function which will generate component
    """
    Gauss = 'Gauss', Gauss
    Exp = 'Exp', Exp

    def __init__(self, component_name: str, component_generator: Callable):
        self.component_name: str = component_name
        self.component_generator: Callable = component_generator

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
        return self.component_generator(x, **kwargs)

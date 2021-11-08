import random
from enum import Enum, EnumMeta, auto
from pathlib import Path
from typing import Optional

import numpy as np

from IntelliStat.generic_builders.utils import load_configuration
from IntelliStat.generic_builders import ComponentBuilder


class ShapeBuilderEnumMeta(EnumMeta):
    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError(f"`index` {index} is not an int")
        if index >= super().__len__():
            raise KeyError("Key error")

        return list(self)[index]


class ShapeBuilder(Enum, metaclass=ShapeBuilderEnumMeta):
    Gauss = 'Gauss', (1, 0), 'gauss.json'
    Gauss_Gauss = 'Gauss+Gauss', (2, 0), 'gauss_gauss.json'
    Gauss_Gauss_Gauss = 'Gauss+Gauss+Gauss', (3, 0), 'gauss_gauss_gauss.json'
    Gauss_Gauss_Exp = 'Gauss+Gauss+Exp', (2, 1), 'gauss_gauss_exp.json'
    Gauss_Exp = 'Gauss+Exp', (1, 1), 'gauss_exp.json'
    Exp = 'Exp', (0, 1), 'exp.json'
    G_4 = '4G', (4, 0), 'gauss_4.json'
    G_5 = '5G', (5, 0), 'gauss_5.json'
    G_6 = '6G', (6, 0), 'gauss_6.json'
    G_7 = '7G', (7, 0), 'gauss_7.json'

    def __new__(cls, *values):
        obj = object.__new__(cls)
        obj._value_ = auto()
        for other_value in values:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values

        return obj

    def __init__(self, component_name: str, class_vector: tuple, config_file: Path):
        self.class_vector = class_vector
        self.shapes_folder = Path(__file__).parent / 'shapes/'
        self.component_name = component_name
        self.config_file = self.shapes_folder / config_file
        self.config_schema_file = Path(__file__).parent / 'schema/shape_schema.json'

    def build_shape(self, x: np.ndarray) -> np.ndarray:
        config = load_configuration(config_file=self.config_file, config_schema_file=self.config_schema_file)
        shape: np.ndarray = np.zeros(x.shape, dtype=np.float32)

        for component in config.components:
            for _ in range(component.amount):
                component_params = {}
                for param in component.params:
                    component_params[param.name] = param.value + random.uniform(*param.range)
                shape += ComponentBuilder(component.name).generate_component(x, **component_params)
        return shape


if __name__ == '__main__':
    builder = ShapeBuilder
    test_x = np.array([[0, 1], [1, 2]])
    print(builder.Gauss_Gauss_Exp.build_shape(x=test_x))

import random
from collections import defaultdict
from enum import Enum, auto, unique
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from IntelliStat.utils import load_configuration
from IntelliStat.builders import ComponentBuilder
from IntelliStat.builders.utils.builder_enum_meta import BuilderEnumMeta


@unique
class ShapeBuilder(Enum, metaclass=BuilderEnumMeta):
    """Builds specified shapes

    Available shapes:
        - Gauss
        - Gauss+Gauss
        - Gauss+Gauss+Gauss
        - Gauss+Gauss+Exp
        - Gauss+Exp
        - Exp
        - 4 x Gauss
        - 5 x Gauss
        - 6 x Gauss
        - 7 x Gauss

    :param shape_name: human name of shape
    :param components_vector: what are the components of shape
    :param config_file: file based on which shape will be build
    """
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

    def __init__(self, shape_name: str, components_vector: tuple, config_file: Union[Path, str]):
        self.class_vector = components_vector
        self.shape_name: str = shape_name
        base_dir = Path(__file__).parent
        self.config_file: Path = base_dir / 'shapes/' / config_file
        self.config_schema_file: Path = base_dir / 'schema/shape_schema.json'

    def build_shape(self, x: np.ndarray, components_params=None) -> Tuple[np.ndarray, dict]:
        config = load_configuration(config_file_path=self.config_file, config_schema_file_path=self.config_schema_file)
        shape: np.ndarray = np.zeros(x.shape, dtype=np.float32)
        if not components_params:
            components_params = []
        for component_idx, component in enumerate(config.components, start=1):
            component_builder = ComponentBuilder(component.name)
            for i in range(component.amount):
                try:
                    params = components_params[component_idx * i]
                except IndexError:
                    params = {}
                    for param in component.params:
                        param_value = param.value
                        if param.range:
                            param_value += random.uniform(*param.range)
                        params[param.name] = param_value
                    components_params.append(params)
                shape += component_builder.generate_component(x, **params)

        shape_params = defaultdict(float)
        for params in components_params:
            for key, value in params.items():
                shape_params[key] += value
        return shape, shape_params

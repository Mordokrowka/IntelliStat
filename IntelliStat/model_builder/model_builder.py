import json
from functools import lru_cache, singledispatch
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

from jsonschema import validate

from IntelliStat.neural_networks import available_neural_networks
from IntelliStat.neural_networks.neural_network import BaseNeuralNetwork


@singledispatch
def wrap_namespace(ob):
    return ob


@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{k: wrap_namespace(v) for k, v in ob.items()})


@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(v) for v in ob]


class ModelBuilder:
    def __init__(self, ):
        # Constant
        self.config_schema_file = Path(__file__).parent / 'resources/config_schema.json'

    @lru_cache(maxsize=1)
    def load_config_schema(self, config_schema_file: Optional[Path] = None):
        with config_schema_file.open() as fp:
            return json.load(fp)

    @lru_cache(maxsize=1)
    def load_configuration(self, config_file: Path, config_schema_file: Optional[Path] = None):
        config_schema_file = config_schema_file if config_schema_file else self.config_schema_file
        with config_file.open() as fp:
            config_data = json.load(fp)
            validate(config_data, schema=self.load_config_schema(config_schema_file))
            return wrap_namespace(config_data)

    def build_model(self, config_file: Path, config_schema_file: Optional[Path] = None) -> BaseNeuralNetwork:
        model_config = self.load_configuration(config_file=config_file, config_schema_file=config_schema_file)
        if not self.validate_neural_network(model_config.neural_network.name):
            raise ValueError(f"Wrong name for neural network model. "
                             f"Available neural models - {available_neural_networks.keys()}")
        neural_network_class = available_neural_networks.get(model_config.neural_network.name)

        return neural_network_class(learning_rate=model_config.neural_network.learning_rate,
                                    **vars(model_config.neural_network.layers))

    @staticmethod
    def validate_neural_network(neural_network_name: str):
        return neural_network_name in available_neural_networks


if __name__ == '__main__':
    builder = ModelBuilder()
    config_schema = Path(__file__).parent / 'resources/config_schema.json'
    config_file = Path(__file__).parent / 'resources/config.json'
    print(builder.build_model(config_file=config_file, config_schema_file=config_schema))

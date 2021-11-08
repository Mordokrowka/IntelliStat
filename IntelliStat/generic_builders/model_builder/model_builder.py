from pathlib import Path

from IntelliStat.generic_builders.utils import load_configuration
from IntelliStat.neural_networks import available_neural_networks
from IntelliStat.neural_networks import BaseNeuralNetwork


def build_model(config_file: Path, config_schema_file: Path) -> BaseNeuralNetwork:
    config = load_configuration(config_file=config_file,
                                config_schema_file=config_schema_file)
    if not validate_neural_network(config.neural_network.name):
        raise ValueError(f"Wrong name for neural network model. "
                         f"Available neural models - {available_neural_networks.keys()}")
    neural_network_class = available_neural_networks.get(config.neural_network.name)

    return neural_network_class(learning_rate=config.neural_network.learning_rate,
                                **vars(config.neural_network.layers))


def validate_neural_network(neural_network_name: str):
    return neural_network_name in available_neural_networks

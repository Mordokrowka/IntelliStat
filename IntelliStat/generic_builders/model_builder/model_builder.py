from pathlib import Path

from IntelliStat.generic_builders.utils import load_configuration
from IntelliStat.neural_networks import available_neural_networks
from IntelliStat.neural_networks import BaseNeuralNetwork


def build_model(config_file: Path, config_schema_file: Path) -> BaseNeuralNetwork:
    """ Build model of neural network based on config file

    :param config_file: path to model config file
    :param config_schema_file: path to json schema file
    :return: Instance of neural network class
    """
    config = load_configuration(config_file=config_file,
                                config_schema_file=config_schema_file)

    neural_network_class = available_neural_networks.get(config.neural_network.name)

    # Check if this type of neural network is implemented
    if not neural_network_class:
        raise ValueError(f"Wrong name for neural network model. "
                         f"Available neural models - {available_neural_networks.keys()}")

    # Initialize neural network object
    neural_network_model = neural_network_class(learning_rate=config.neural_network.learning_rate,
                                                **vars(config.neural_network.layers))
    return neural_network_model

from pathlib import Path

from IntelliStat.neural_networks import BaseNeuralNetwork
from IntelliStat.neural_networks import available_neural_networks
from IntelliStat.utils import load_configuration


def build_model(config_file_path: Path, config_schema_file_path: Path) -> BaseNeuralNetwork:
    """ Build model of neural network based on config file

    :param config_file_path: path to model config file
    :param config_schema_file_path: path to json schema file
    :return: Instance of neural network class
    """
    config = load_configuration(config_file_path=config_file_path,
                                config_schema_file_path=config_schema_file_path)

    # Check if this type of neural network is implemented
    if config.neural_network.name not in available_neural_networks:
        error_message = f"Wrong name for neural network model. "
        f"Available neural models - {available_neural_networks.keys()}"
        raise ValueError(error_message)

    neural_network_class = available_neural_networks.get(config.neural_network.name)

    # Initialize neural network object
    learning_rate = config.neural_network.learning_rate
    neural_network_model = neural_network_class(learning_rate=learning_rate,
                                                **vars(config.neural_network.layers))
    return neural_network_model

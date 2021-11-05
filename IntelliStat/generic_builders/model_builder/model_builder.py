from pathlib import Path
from typing import Optional

from IntelliStat.generic_builders import BaseBuilder
from IntelliStat.neural_networks import available_neural_networks
from IntelliStat.neural_networks import BaseNeuralNetwork


class ModelBuilder(BaseBuilder):
    def __init__(self):
        # Constant
        super().__init__()

    def build_model(self, config_file: Path, config_schema_file: Optional[Path] = None) -> BaseNeuralNetwork:
        config = self.load_configuration(config_file=config_file, config_schema_file=config_schema_file)
        if not self.validate_neural_network(config.neural_network.name):
            raise ValueError(f"Wrong name for neural network model. "
                             f"Available neural models - {available_neural_networks.keys()}")
        neural_network_class = available_neural_networks.get(config.neural_network.name)

        return neural_network_class(learning_rate=config.neural_network.learning_rate,
                                    **vars(config.neural_network.layers))

    @staticmethod
    def validate_neural_network(neural_network_name: str):
        return neural_network_name in available_neural_networks

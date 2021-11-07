from .base_neural_network import BaseNeuralNetwork
from .enn import ENN
from .enn_classifier import ENN_Classifier

available_neural_networks = {
    "ENN": ENN,
    "ENN_classifier": ENN_Classifier
}
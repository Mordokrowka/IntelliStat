from pathlib import Path
from random import random

import numpy as np
from matplotlib import pyplot as plt

from IntelliStat.datasets.dataset import Dataset
from IntelliStat.generic_builders import build_model
from IntelliStat.generic_builders.utils import load_configuration


def main():
    #  Config and validation file
    config_schema = Path(__file__).parent / 'resources/config_schema.json'
    config_file = Path(__file__).parent / 'resources/config.json'

    # Build neural network model
    EvolutionalNN = build_model(config_file=config_file, config_schema_file=config_schema)

    # Load Configuration
    configuration = load_configuration(config_file=config_file, config_schema_file=config_schema)
    epoch: int = configuration.epoch
    batch_size: int = configuration.batch_size

    # Create X data
    X_data = [[n_x + 3 * (random() - 0.5)] for n_x in range(epoch)]
    X_data = np.array(X_data, dtype=np.float32)

    # Create Y data
    Y_data = [[(2 * x_point[0] + 2 + 10 * (random() - 0.5))] for x_point in X_data]
    Y_data = np.array(Y_data, dtype=np.float32)

    # Create linear shape
    Y = X_data * 2 + 2

    # Create Dataset
    dataset = Dataset(X_data, Y)

    # Train neural network model
    EvolutionalNN.train(dataset, epoch, batch_size)

    # After training, checking performance
    X_tensor = [[n_x + 3 * (random() - 0.5)] for n_x in range(epoch)]
    X_tensor = np.array(X_tensor, dtype=np.float32)
    Y_NN = EvolutionalNN.test(X_tensor)

    # Visualization
    fig, ax = plt.subplots()
    ax.plot(X_data, Y_data, 'ko', label="Data points")
    # print("Regression:")
    # print("A: ",A,", mean: ", u,", sigma: ", sigma)
    # X = np.linspace(0,50,501)
    # Y = [ gauss(x_point, A, u, sigma) for x_point in X]
    ax.plot(X_data, Y_NN, 'blue', label="Evolutional NN")
    leg = ax.legend(loc='upper left', prop={'size': 7})

    # A, u, sigma = round(A, 8), round(u, 8), round(sigma, 8)
    # a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    # ax.text(0.45, 1.06, "A: " + str(A) + ", mean: " + str(u) + ", sigma: " + str(sigma),
    #    horizontalalignment='center', verticalalignment='center',
    #    transform=ax.transAxes, color = 'blue')
    plt.show()


if __name__ == "__main__":
    main()

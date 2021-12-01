import math
from pathlib import Path
from pprint import pprint
from random import random
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from IntelliStat.datasets import Dataset
from IntelliStat.generic_builders import ShapeBuilder, build_model
from IntelliStat.generic_builders.utils import load_configuration
from IntelliStat.neural_networks import BaseNeuralNetwork


def multi_shape_classifier():
    # Config and validation file
    config_schema = Path(__file__).parent / 'resources/config_schema.json'
    config_file = Path(__file__).parent / 'resources/config.json'

    # Build neural network model
    EvolutionalNN: BaseNeuralNetwork = build_model(config_file=config_file, config_schema_file=config_schema)

    # Load Configuration
    configuration = load_configuration(config_file=config_file, config_schema_file=config_schema)
    epoch: int = configuration.epoch
    samples: int = configuration.samples
    batch_size: int = configuration.batch_size
    shapes: int = configuration.shapes

    # Create X data
    X_data: List[List[float]] = [[X / 4 for X in range(40)] for _ in range(shapes * samples)]
    X_data: np.ndarray = np.array(X_data, dtype=np.float32)

    # Initialize Y data
    def build_linear(x):
        a = 0.05
        b = 0.05 * random()
        return a * x + b * random(), {"a": a, "b": b}

    Y_data: np.ndarray = np.zeros((X_data.shape[0], shapes), dtype=np.float32)
    shapes_builders = [
        ShapeBuilder.Gauss.build_shape,
        ShapeBuilder.Exp.build_shape,
        build_linear,
    ]

    shape_areas_calculator = [
        lambda A, sigma, u: A * math.sqrt(math.pi / u),
        lambda B, b: (B - B * math.e ** (-b * 10) / b),
        lambda a, b: 10 * (a * 10 + 2 * b) / 2,
    ]

    # Fill Y data using ShapeBuilder
    for shape in range(shapes):
        for i in range(samples):
            shapes_area = np.zeros(shapes)
            x_data = np.zeros(X_data[i + shape * samples].shape)
            for idx, (shape_builder, calculate_shape_area) in enumerate(zip(shapes_builders, shape_areas_calculator)):
                shape_data, shape_params = shape_builder(X_data[i + shape * samples])
                shapes_area[idx] = calculate_shape_area(**shape_params)
                x_data += shape_data
            X_data[i + shape * samples] = x_data
            Y_data[i + shape * samples] = shapes_area / np.sum(shapes_area)

    # Split data to test/train datasets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=configuration.test_dataset_size)

    # Define dataset
    dataset = Dataset(X_train, Y_train)

    # Train neural network model
    EvolutionalNN.train(dataset, epoch, batch_size)

    # Test trained model
    Y_NN = EvolutionalNN.test(X_test)
    pprint(Y_NN)
    pprint(Y_test)

    accuracy: int = 0
    for i in range(len(Y_NN)):
        # if np.array_equal(np.round(Y_NN[i], 1), np.round(Y_test[i], 1)):
        # accuracy += 1
        # print(np.mean(np.abs(Y_NN[i] - Y_test[i]), axis=0), np.mean(np.abs(Y_NN[i] - Y_test[i]) / Y_test[i], axis=0))
        if np.mean(np.abs(Y_NN[i] - Y_test[i]) / Y_test[i], axis=0) < 0.1:
            accuracy += 1

    accuracy = accuracy / Y_NN.shape[0]
    print("Reached accuracy: ", accuracy)

    # # Visualization
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(8, 8)
    # n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 0], 100, alpha=0.5, range=[0, 1], label='Gauss')
    # n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 1], 100, alpha=0.5, range=[0, 1], label='Gauss+Gauss')
    # n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 2], 100, alpha=0.5, range=[0, 1], label='Gauss+Gauss+Gauss')
    # n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 3], 100, alpha=0.5, range=[0, 1], label='Gauss+Gauss+Exp')
    # n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 4], 100, alpha=0.5, range=[0, 1], label='Gauss+Exp')
    # n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 5], 100, alpha=0.5, range=[0, 1], label='Exp')
    # ax[0, 0].set_xlabel('Class A (Gauss+Gauss+Exp)')
    # ax[0, 0].set_ylabel('Number of counts')
    # ax[0, 0].legend(loc='upper right')
    # ax[0, 0].set_title('Class affiliation probability')
    #
    ax[1, 0].plot(range(0, epoch), EvolutionalNN.loss_vector, 'k-', label="Loss function")
    ax[1, 0].set_xlabel('Epoch number')
    ax[1, 0].set_ylabel('Loss function')
    ax[1, 0].set_yscale('log')

    X_plot = [X / 40 for X in range(400)]
    X_plot = np.array(X_plot, dtype=np.float32)

    x_data = np.zeros(X_plot.shape)
    for shape_builder in shapes_builders:
        x_data += shape_builder(X_plot)[0]
    X_plot = x_data
    ax[0, 1].plot(np.linspace(0, 10, 400, endpoint=False), X_plot, 'r-')

    # ax[0, 1].plot(np.linspace(0, 10, 40, endpoint=False), X_test[0], 'g-', label="Test")
    ax[0, 1].set_xlabel('X argument')
    ax[0, 1].set_ylabel('Y')
    # ax[0, 1].legend(loc='upper right')
    ax[0, 1].set_title('Exemplary training G + E + Linear + Double E')

    # n, bins, patches = ax[0, 0].hist(np.mean(np.abs(Y_NN[:1] - Y_test[:1]) / Y_test[:1], axis=0), 100, alpha=0.5, color = 'blue', label='St. dev.')
    # ax[0, 0].set_tite('Average relative error')
    # ax[0, 1].set_xlabel('Relative error')
    #
    # ax[0, 0].set_ylabel('Number of counts')

    # ax[0,1].plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[2], 'r-', label="Gauss no. 2")
    # ax[0,1].plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[3], 'b-', label="Gauss no. 3")

    # ax[0,0].plot(Y_NN[1][0], Gauss(Y_NN[1][0], 1, Y_NN[1][0], Y_NN[1][1]), 'k*', label="Predicted mean")
    # ax[0,0].plot(Y_NN[2][0], Gauss(Y_NN[2][0], 1, Y_NN[2][0], Y_NN[2][1]), 'r*', label="Predicted mean")
    # ax[0,0].plot(Y_NN[3][0], Gauss(Y_NN[3][0], 1, Y_NN[3][0], Y_NN[3][1]), 'b*', label="Predicted mean")
    # ax[0,0].set_xlabel('Argument X')
    # ax[0,0].set_ylabel('Value Y')

    # leg = ax[0,0].legend(loc = 'upper left', prop={'size':7})

    # A, u, sigma = round(A, 8), round(u, 8), round(sigma, 8)
    # a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    # ax[0,0].text(0.45, 1.12, " Mean: " + str(round(Y_data[1][0],3)) + ", NN prediction: " + str(round(Y_NN[1][0],3)),
    #    horizontalalignment='center', verticalalignment='center',
    #    transform=ax[0,0].transAxes, color = 'k')
    # ax[0,0].text(0.45, 1.07, " Mean: " + str(round(Y_data[2][0], 3)) + ", NN prediction: " + str(round(Y_NN[2][0], 3)),
    #        horizontalalignment='center', verticalalignment='center',
    #        transform=ax[0,0].transAxes, color='r')
    # ax[0,0].text(0.45, 1.02, " Mean: " + str(round(Y_data[3][0], 3)) + ", NN prediction: " + str(round(Y_NN[3][0], 3)),
    #        horizontalalignment='center', verticalalignment='center',
    #        transform=ax[0,0].transAxes, color='b')

    # n, bins, patches = ax[0, 1].hist(Y_data[:,0] - Y_NN[:,0], 100, alpha=0.5, color = 'red', label='Mean')
    # ax[0, 1].set_xlabel('Real - NN prediction')
    # ax[0, 1].set_ylabel('Number of counts')

    # n, bins, patches = ax[1, 1].hist(Y_data[:,1] - Y_NN[:,1], 100, alpha=0.5, color = 'blue', label='St. dev.')
    # ax[1, 1].set_xlabel('Real - NN prediction')
    # ax[1, 1].set_ylabel('Number of counts')
    # axs[0, 1].axvline(x=np.mean(Mass_B_data), color='r', linestyle='dashed')
    # ax[0, 1].legend(loc='upper right')
    # ax[1, 1].legend(loc='upper right')

    plt.show()


if __name__ == "__main__":
    multi_shape_classifier()

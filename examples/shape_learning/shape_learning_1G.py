from random import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from IntelliStat.neural_networks import ENN
from IntelliStat.datasets.shape_dataset import ShapeDataset
from IntelliStat.generic_builders.component_builder.component_functions import Gauss


def main():
    EvolutionalNN = ENN(20, 20, 10, 5, 2, learning_rate=0.001)
    epoch = 500
    X_data = [[X / 2 for X in range(20)] for _ in range(500)]
    Y_data = [[4 * random() + 3, 0.5 + random()] for _ in X_data]
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = ShapeDataset(X_data, "Gauss", Y_data)
    EvolutionalNN.train(Dataset, epoch, 20)

    # After training, checking performance
    test_data = [[4 * random() + 3, 0.5 + random()] for _ in X_data]
    test_data = np.array(test_data, dtype=np.float32)
    test_data = ShapeDataset(X_data, "Gauss", test_data).X
    X_NN = torch.tensor(test_data)
    Y_NN = EvolutionalNN.model(X_NN)
    Y_NN = Y_NN.detach().numpy()

    vis_len = 100
    F_X = np.linspace(0, 10, vis_len, endpoint=False)
    In_data = Gauss(F_X, 1, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(8, 8)
    ax[0, 0].plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[1], 'k-', label="Gauss no. 1")
    ax[0, 0].plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[2], 'r-', label="Gauss no. 2")
    ax[0, 0].plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[3], 'b-', label="Gauss no. 3")

    ax[0, 0].plot(Y_NN[1][0], Gauss(Y_NN[1][0], 1, Y_NN[1][0], Y_NN[1][1]), 'k*', label="Predicted mean")
    ax[0, 0].plot(Y_NN[2][0], Gauss(Y_NN[2][0], 1, Y_NN[2][0], Y_NN[2][1]), 'r*', label="Predicted mean")
    ax[0, 0].plot(Y_NN[3][0], Gauss(Y_NN[3][0], 1, Y_NN[3][0], Y_NN[3][1]), 'b*', label="Predicted mean")
    ax[0, 0].set_xlabel('Argument X')
    ax[0, 0].set_ylabel('Value Y')

    leg = ax[0, 0].legend(loc='upper left', prop={'size': 7})

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

    n, bins, patches = ax[0, 1].hist(Y_data[:, 0] - Y_NN[:, 0], 100, alpha=0.5, color='red', label='Mean')
    ax[0, 1].set_xlabel('Real - NN prediction')
    ax[0, 1].set_ylabel('Number of counts')

    n, bins, patches = ax[1, 1].hist(Y_data[:, 1] - Y_NN[:, 1], 100, alpha=0.5, color='blue', label='St. dev.')
    ax[1, 1].set_xlabel('Real - NN prediction')
    ax[1, 1].set_ylabel('Number of counts')
    # axs[0, 1].axvline(x=np.mean(Mass_B_data), color='r', linestyle='dashed')
    ax[0, 1].legend(loc='upper right')
    ax[1, 1].legend(loc='upper right')

    ax[1, 0].plot(range(0, epoch), EvolutionalNN.loss_vector, 'k-', label="Loss function")
    ax[1, 0].set_xlabel('Epoch number')
    ax[1, 0].set_ylabel('Loss function')
    ax[1, 0].set_yscale('log')

    plt.show()


if __name__ == "__main__":
    main()

from random import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from IntelliStat.neural_networks.ENN import ENN
from IntelliStat.utils.datasets.shape_dataset import ShapeDataset
from IntelliStat.utils.components.component_functions import Gauss


def main():
    EvolutionalNN = ENN(40, 20, 20, 10, 4)

    X_data = [[X / 2 for X in range(40)] for _ in range(1000)]
    Y_data = [[random() + 4, 0.5, random() + 6, 0.5] for _ in X_data]
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = ShapeDataset(X_data, "Gauss+Gauss", Y_data)
    EvolutionalNN.train(Dataset, 500, 20)

    test_data = [[random() + 4, 0.5, random() + 6, 0.5] for _ in X_data]
    test_data = np.array(test_data, dtype=np.float32)
    test_data = ShapeDataset(X_data, "Gauss+Gauss", test_data).X
    X_NN = torch.tensor(test_data)
    Y_NN = EvolutionalNN.model(X_NN)
    Y_NN = Y_NN.detach().numpy()

    for i in range(Y_data.shape[0]):
        print("Mean, real : ", Y_data[i][0], Y_data[i][2], " , trained: ", Y_NN[i][0], Y_NN[i][2])
        print("Std, real : ", Y_data[i][1], Y_data[i][3], " , trained: ", Y_NN[i][1], Y_NN[i][3])

    vis_len = 100

    F_X = np.linspace(0, 10, vis_len, endpoint=False)

    In_data_g1 = Gauss(F_X, 1, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))
    In_data_g2 = Gauss(F_X, 1, Y_data[:, 2].reshape(-1, 1), Y_data[:, 3].reshape(-1, 1))
    In_data = In_data_g1 + In_data_g2


    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data_g1[1], 'g-.', label="Partial Gauss 1")
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data_g2[1], 'k-.', label="Partial Gauss 2")
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[1], 'r-', linewidth=2, label="Total")
    # ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[2], 'r-', label="Gauss no. 2")
    # ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[3], 'b-', label="Gauss no. 3")

    ax.plot([Y_NN[1][0], Y_NN[1][2]], [Gauss(Y_NN[1][0], 1, Y_NN[1][0], Y_NN[1][1])
                                       + Gauss(Y_NN[1][0], 1, Y_NN[1][2], Y_NN[1][3]),
                                       Gauss(Y_NN[1][2], 1, Y_NN[1][0], Y_NN[1][1])
                                       + Gauss(Y_NN[1][2], 1, Y_NN[1][2], Y_NN[1][3])],
            'k*', linewidth=2, label="Predicted mean")

    # ax.plot(Y_NN[2][0], Gauss(Y_NN[2][0], 1, Y_NN[2][0], Y_NN[2][1]), 'r*', label="Predicted mean")
    # ax.plot(Y_NN[3][0], Gauss(Y_NN[3][0], 1, Y_NN[3][0], Y_NN[3][1]), 'b*', label="Predicted mean")
    # print("Regression:")
    # print("Red, real mean: ", Y_data[1][0] ,", NN output: ", Y_NN[1][0])
    # X = np.linspace(0,50,501)
    # Y = [ gauss(x_point, A, u, sigma) for x_point in X]
    # ax.plot(X_data, Y_NN, 'blue', label = "Evolutional NN")
    leg = ax.legend(loc='upper left', prop={'size': 7})

    # A, u, sigma = round(A, 8), round(u, 8), round(sigma, 8)
    # a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    ax.text(0.45, 1.06, " Mean: " + str(round(Y_data[1][0], 3)) + ", " + str(round(Y_data[1][2], 3))
            + ", NN prediction: " + str(round(Y_NN[1][0], 3)) + ", " + str(round(Y_NN[1][2], 3)),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='r')

    plt.show()


if __name__ == "__main__":
    main()

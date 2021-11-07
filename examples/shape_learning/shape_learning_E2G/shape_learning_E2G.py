import math
from pathlib import Path
from random import random

import numpy as np
from matplotlib import pyplot as plt

from IntelliStat.datasets.dataset import Dataset
from IntelliStat.generic_builders import ModelBuilder, ShapeBuilder
from IntelliStat.generic_builders.component_builder.components import Gauss


def main():
    builder = ModelBuilder()
    config_schema = Path(__file__).parent / 'resources/config_schema.json'
    config_file = Path(__file__).parent / 'resources/config.json'

    EvolutionalNN = builder.build_model(config_file=config_file, config_schema_file=config_schema)

    configuration = builder.load_configuration(config_file=config_file, config_schema_file=config_schema)
    epoch: int = configuration.epoch
    batch_size = configuration.batch_size

    X_data = [[X / 2 for X in range(80)] for _ in range(epoch)]
    Y_data = [[random() + 4, 0.32 + random() / 2, random() + 6, 0.32 + random() / 2] for X in X_data]
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    shape = ShapeBuilder.Gauss_Gauss_Exp.build_shape(X_data)
    dataset = Dataset(shape, Y_data)
    EvolutionalNN.train(dataset, epoch, batch_size)

    Y_NN = EvolutionalNN.test(X_data)

    for i in range(len(Y_data)):
        print("Mean, real : ", Y_data[i][0], Y_data[i][2], " , trained: ", Y_NN[i][0], Y_NN[i][2])
        print("Std, real : ", Y_data[i][1], Y_data[i][3], " , trained: ", Y_NN[i][1], Y_NN[i][3])

    vis_len = 100
    In_data = [[0 for _ in range(vis_len)] for _ in range(len(X_data))]
    In_data_g1 = [[0 for _ in range(vis_len)] for _ in range(len(X_data))]
    In_data_g2 = [[0 for _ in range(vis_len)] for _ in range(len(X_data))]
    In_data_e = [[0 for _ in range(vis_len)] for _ in range(len(X_data))]
    F_X = np.linspace(0, 10, vis_len, endpoint=False)
    for i in range(len(X_data)):
        for j in range(vis_len):
            In_data[i][j] = Gauss(F_X[j], 1, Y_data[i][0], Y_data[i][1]) \
                            + Gauss(F_X[j], 1, Y_data[i][2], Y_data[i][3]) \
                            + math.exp(-0.2 * F_X[j])
            In_data_g1[i][j] = Gauss(F_X[j], 1, Y_data[i][0], Y_data[i][1])
            In_data_g2[i][j] = Gauss(F_X[j], 1, Y_data[i][2], Y_data[i][3])
            In_data_e[i][j] = math.exp(-0.2 * F_X[j])

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data_g1[1], 'g-.', label="Partial Gauss 1")
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data_g2[1], 'k-.', label="Partial Gauss 2")
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data_e[1], 'b-.', label="Exp background")
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[1], 'r-', linewidth=2, label="Total")
    # ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[2], 'r-', label="Gauss no. 2")
    # ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[3], 'b-', label="Gauss no. 3")

    ax.plot([Y_NN[1][0], Y_NN[1][2]], [Gauss(Y_NN[1][0], 1, Y_NN[1][0], Y_NN[1][1])
                                       + Gauss(Y_NN[1][0], 1, Y_NN[1][2], Y_NN[1][3])
                                       + math.exp(-0.2 * Y_NN[1][0]),
                                       Gauss(Y_NN[1][2], 1, Y_NN[1][0], Y_NN[1][1])
                                       + Gauss(Y_NN[1][2], 1, Y_NN[1][2], Y_NN[1][3])
                                       + math.exp(-0.2 * Y_NN[1][2])],
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
    ax.text(0.45, 1.1, " Mean: " + str(round(Y_data[1][0], 3)) + ", " + str(round(Y_data[1][2], 3))
            + ", NN prediction: " + str(round(Y_NN[1][0], 3)) + ", " + str(round(Y_NN[1][2], 3)),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='k')
    ax.text(0.45, 1.05, " Std: " + str(round(Y_data[1][1], 3)) + ", " + str(round(Y_data[1][3], 3))
            + ", NN prediction: " + str(round(Y_NN[1][1], 3)) + ", " + str(round(Y_NN[1][3], 3)),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='k')

    plt.show()


if __name__ == "__main__":
    main()

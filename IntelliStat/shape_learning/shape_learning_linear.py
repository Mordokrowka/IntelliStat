from random import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from IntelliStat.utils.datasets import FunctionDataset
from IntelliStat.neural_networks import ENN


# TODO FIX
def main():
    EvolutionalNN = ENN(1, 2, 4, 2, 1, learning_rate=0.001)

    X_data = [[n_x + 3 * (random() - 0.5)] for n_x in range(50)]
    Y_data = [[(2 * x_point[0] + 2 + 10 * (random() - 0.5))] for x_point in X_data]

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = FunctionDataset(X_data, "linear", [2, 2])
    EvolutionalNN.train(Dataset, 2000, 5)

    X_tensor = [[n_x + 3 * (random() - 0.5)] for n_x in range(50)]
    X_data = np.array(X_data, dtype=np.float32)
    X_tensor = torch.tensor(X_tensor)
    Y_NN = EvolutionalNN.model(X_tensor)
    Y_NN = Y_NN.detach().numpy()

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

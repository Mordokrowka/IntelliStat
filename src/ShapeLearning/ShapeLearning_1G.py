import numpy as np
import torch

from random import random

from src.dataset import GaussDataset
from src.ENN import ENN
from src.utils import Gauss
from src.visualization import visualize


def test_with_different_sigma():
    EvolutionalNN = ENN(20, 20, 10, 5, 2)
    epoch = 500

    X_data = [[X / 2 for X in range(20)] for _ in range(epoch)]
    Y_data = [[4 * random() + 3, 0.5 + 10* random()] for _ in X_data]

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = GaussDataset(X_data, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))

    EvolutionalNN.train(Dataset, epoch, 20)

    sigmas = [0.5, 3, 7, 9.5]
    for sigma in sigmas:

        Y_data = [[4 * random() + 3, sigma + random(), 1] for _ in X_data]
        Y_data = np.array(Y_data, dtype=np.float32)
        test_data = np.zeros(X_data.shape, dtype=np.float32)
        for i in range(X_data.shape[0]):
            test_data[i] = Gauss(X_data[i], Y_data[i][2], Y_data[i][0], Y_data[i][1])

        X_NN = torch.tensor(test_data)
        Y_NN = EvolutionalNN.model(X_NN)
        Y_NN = Y_NN.detach().numpy()
        Y_NN = np.append(Y_NN, np.ones([len(Y_NN), 1]), 1)
        visualize(epoch, Y_data, Y_NN, EvolutionalNN.loss_vector, sigma)


def test_with_different_A_v1():
    EvolutionalNN = ENN(20, 20, 10, 5, 2)
    epoch = 500

    X_data = [[X / 2 for X in range(20)] for _ in range(epoch)]
    Y_data = [[4 * random() + 3, 0.5 + random()] for _ in X_data]

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = GaussDataset(X_data, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))
    EvolutionalNN.train(Dataset, epoch, 20)

    amplitudes = [1, 3, 5, 10]
    for A in amplitudes:
        Y_data = [[4 * random() + 3, 0.5 + random(), A] for _ in X_data]
        Y_data = np.array(Y_data, dtype=np.float32)
        test_data = np.zeros(X_data.shape, dtype=np.float32)
        for i in range(X_data.shape[0]):
            test_data[i] = Gauss(X_data[i], Y_data[i][2], Y_data[i][0], Y_data[i][1])

        X_NN = torch.tensor(test_data)
        Y_NN = EvolutionalNN.model(X_NN)
        Y_NN = Y_NN.detach().numpy()
        Y_NN = np.append(Y_NN, np.full([len(Y_NN), 1], A), 1)
        visualize(epoch, Y_data, Y_NN, EvolutionalNN.loss_vector, amplitude=A)


def test_with_different_A_v2():
    EvolutionalNN = ENN(20, 20, 10, 5, 3)
    epoch = 1000

    X_data = [[X / 2 for X in range(20)] for _ in range(2*epoch)]
    Y_data = [[4 * random() + 3, 0.5 + random(), 1 + 10 * random()] for _ in X_data]

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = GaussDataset(X_data, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1), Y_data[:, 2].reshape(-1, 1))
    EvolutionalNN.train(Dataset, epoch, 20)

    amplitudes = [1, 3, 5, 10]
    for A in amplitudes:
        Y_data = [[4 * random() + 3, 0.5 + random(), A + random()] for _ in X_data]
        Y_data = np.array(Y_data, dtype=np.float32)
        test_data = np.zeros(X_data.shape, dtype=np.float32)
        for i in range(X_data.shape[0]):
            test_data[i] = Gauss(X_data[i], Y_data[i][2], Y_data[i][0], Y_data[i][1])

        X_NN = torch.tensor(test_data)
        Y_NN = EvolutionalNN.model(X_NN)
        Y_NN = Y_NN.detach().numpy()
        visualize(2*epoch, Y_data, Y_NN, EvolutionalNN.loss_vector, amplitude=A)


def main():
    # test_with_different_sigma()
    # test_with_different_A_v1()
    test_with_different_A_v2()


if __name__ == "__main__":
    main()

import numpy as np
import torch

from random import random

from sklearn.preprocessing import StandardScaler
from src.dataset import GaussDataset
from src.ENN import ENN
from src.utils import Gauss
from src.visualization import visualize, track_mean_evolution


def test_with_different_sigma():
    EvolutionalNN = ENN(20, 20, 10, 5, 2)
    epoch = 500

    X_data = [[X / 2 for X in range(20)] for _ in range(epoch)]
    Y_data = [[4 * random() + 3, 0.5 + random()] for _ in X_data]

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = GaussDataset(X_data, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))

    EvolutionalNN.train(Dataset, epoch, 20)

    sigmas = [0.5, 3, 7, 9.5]
    for sigma in sigmas:
        Y_data = [[4 * random() + 3, sigma + random(), 1] for _ in X_data]
        Y_data = np.array(Y_data, dtype=np.float32)

        test_data = Gauss(X_data, Y_data[:, 2].reshape(-1, 1), Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))

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

        test_data = Gauss(X_data, Y_data[:, 2].reshape(-1, 1), Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))

        X_NN = torch.tensor(test_data)
        Y_NN = EvolutionalNN.model(X_NN)
        Y_NN = Y_NN.detach().numpy()
        Y_NN = np.append(Y_NN, np.full([len(Y_NN), 1], A), 1)
        visualize(epoch, Y_data, Y_NN, EvolutionalNN.loss_vector, amplitude=A)


def test_with_different_A_v2():
    EvolutionalNN = ENN(20, 20, 10, 5, 3)
    epoch = 500

    X_data = [[X / 2 for X in range(20)] for _ in range(epoch)]
    Y_data = [[4 * random() + 3, 0.5 + random(), 1 + 10 * random()] for _ in X_data]

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = GaussDataset(X_data, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1),
                           Y_data[:, 2].reshape(-1, 1))
    EvolutionalNN.train(Dataset, epoch, 20)

    amplitudes = [1, 3, 5, 10]
    for A in amplitudes:
        Y_data = [[4 * random() + 3, 0.5 + random(), A + random()] for _ in X_data]
        Y_data = np.array(Y_data, dtype=np.float32)

        test_data = Gauss(X_data, Y_data[:, 2].reshape(-1, 1), Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))

        X_NN = torch.tensor(test_data)
        Y_NN = EvolutionalNN.model(X_NN)
        Y_NN = Y_NN.detach().numpy()
        visualize(epoch, Y_data, Y_NN, EvolutionalNN.loss_vector, amplitude=A)


def test_with_noise():
    EvolutionalNN = ENN(20, 20, 10, 5, 2)
    epoch = 500

    X_data = [[X / 2 for X in range(20)] for _ in range(epoch)]
    Y_data = [[4 * random() + 3, 0.5 + random(), 1] for _ in X_data]

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = GaussDataset(X_data, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))

    EvolutionalNN.train(Dataset, epoch, 20)

    Y_data = [[4 * random() + 3, 0.5 + random(), 1] for _ in X_data]
    Y_data = np.array(Y_data, dtype=np.float32)

    test_data = Gauss(X_data, 1, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))
    percentage = np.random.choice(list(range(95, 105)), size=Dataset.X.shape)
    test_data *= percentage / 100
    X_NN = torch.tensor(test_data)
    Y_NN = EvolutionalNN.model(X_NN)
    Y_NN = Y_NN.detach().numpy()
    Y_NN = np.append(Y_NN, np.ones([len(Y_NN), 1]), 1)
    visualize(epoch, Y_data, Y_NN, EvolutionalNN.loss_vector)


def add_residues():
    EvolutionalNN = ENN(120, 80, 40, 20, 3)
    epoch = 500

    X_data = [[X / 2 for X in range(120)] for _ in range(4 * epoch)]
    Y_data = [[4 * random() + 3, 0.5 + random(), 1 + random()] for _ in X_data]

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = GaussDataset(X_data, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1),
                           Y_data[:, 2].reshape(-1, 1))

    epochs = np.linspace(10, epoch, dtype=int)

    Y_data = [[4 * random() + 3, 0.5 + random(), 1 + random()] for _ in X_data]
    Y_data = np.array(Y_data, dtype=np.float32)

    test_data = Gauss(X_data, Y_data[:, 2].reshape(-1, 1), Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))

    scaler = StandardScaler()
    scaler.fit(test_data)
    test_data = scaler.transform(test_data)
    residuals = np.zeros((len(epochs), 3))

    for index, e in enumerate(epochs):
        EvolutionalNN.train(Dataset, 10, 20)
        X_NN = torch.tensor(test_data)
        Y_NN = EvolutionalNN.model(X_NN)
        Y_NN = Y_NN.detach().numpy()

        residuals[index] = np.mean(np.abs(Y_NN - Y_data) / Y_data * 100, axis=0)

    visualize(4 * epoch, Y_data, Y_NN, EvolutionalNN.loss_vector, residuals=residuals, epochs=epochs)


def mean_evolution():
    EvolutionalNN = ENN(120, 80, 40, 20, 3)
    epoch = 500

    X_data = [[X / 2 for X in range(120)] for _ in range(4 * epoch)]
    Y_data = [[4 * random() + 3, 0.5 + random(), 1 + random()] for _ in X_data]

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    Dataset = GaussDataset(X_data, Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1),
                           Y_data[:, 2].reshape(-1, 1))

    Y_data = [[4 * random() + 3, 0.5 + random(), 1 + random()] for _ in X_data]
    Y_data = np.array(Y_data, dtype=np.float32)

    test_data = Gauss(X_data, Y_data[:, 2].reshape(-1, 1), Y_data[:, 0].reshape(-1, 1), Y_data[:, 1].reshape(-1, 1))

    scaler = StandardScaler()
    scaler.fit(test_data)
    test_data = scaler.transform(test_data)
    means_NN = []
    for e in [100] * 5:
        EvolutionalNN.train(Dataset, e, 20)
        X_NN = torch.tensor(test_data)
        Y_NN = EvolutionalNN.model(X_NN)
        Y_NN = Y_NN.detach().numpy()
        means_NN.append(Y_NN[:, 0])

    track_mean_evolution(Y_data[:, 0], means_NN, step=100)
    # visualize(4*epoch, Y_data, Y_NN, EvolutionalNN.loss_vector)


def main():
    # test_with_different_sigma()
    # test_with_different_A_v1()
    # test_with_different_A_v2()
    # test_with_noise()
    # add_residues()
    mean_evolution()


if __name__ == "__main__":
    main()

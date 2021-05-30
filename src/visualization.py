import numpy as np
from matplotlib import pyplot as plt
from utils import Gauss


def visualize(epoch, Y_data, Y_NN, loss_vector, sigma=1, amplitude=1):
    vis_len = 100

    In_data = np.zeros((epoch, vis_len))

    F_X = np.linspace(0, 10, vis_len, endpoint=False)
    for i in range(Y_data.shape[0]):
        In_data[i] = Gauss(F_X, Y_data[i][2], Y_data[i][0], Y_data[i][1])

    fig, ax = plt.subplots(2, 3)
    fig.suptitle(f"Sigma={sigma}, Amplitude={amplitude}", fontsize=16)
    fig.set_size_inches(12, 12)

    plot_gauss(ax[0, 0], F_X, In_data, Y_NN)

    plot_mean_histogram(ax[0, 1], Y_data[:, 0] - Y_NN[:, 0])

    plot_st_dev_histogram(ax[1, 1], Y_data[:, 1] - Y_NN[:, 1])

    plot_amplitude_histogram(ax[1, 2], Y_data[:, 2] - Y_NN[:, 2])

    plot_loss_function(ax[1, 0], loss_vector)

    plt.show()


def plot_loss_function(ax, loss_vector):
    ax.plot(loss_vector, 'k-', label="Loss function")
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Loss function')
    ax.set_yscale('log')


def plot_gauss(ax, F_X, gauss_data, Y_NN):
    ax.plot(F_X, gauss_data[1], 'k-', label="Gauss no. 2")
    ax.plot(F_X, gauss_data[2], 'r-', label="Gauss no. 2")
    ax.plot(F_X, gauss_data[3], 'b-', label="Gauss no. 3")

    ax.plot(Y_NN[1][0], Gauss(Y_NN[1][0], Y_NN[1][2], Y_NN[1][0], Y_NN[1][1]), 'k*', label="Predicted mean")
    ax.plot(Y_NN[2][0], Gauss(Y_NN[2][0], Y_NN[2][2], Y_NN[2][0], Y_NN[2][1]), 'r*', label="Predicted mean")
    ax.plot(Y_NN[3][0], Gauss(Y_NN[3][0], Y_NN[3][2], Y_NN[3][0], Y_NN[3][1]), 'b*', label="Predicted mean")

    ax.set_xlabel('Argument X')
    ax.set_ylabel('Value Y')

    ax.legend(loc='upper left', prop={'size': 7})


def plot_mean_histogram(ax, x):
    n, bins, patches = ax.hist(x, 100, alpha=0.5, color='red', label='Mean')
    ax.set_xlabel('Real - NN prediction')
    ax.set_ylabel('Number of counts')

    ax.legend(loc='upper right')


def plot_st_dev_histogram(ax, x):
    n, bins, patches = ax.hist(x, 100, alpha=0.5, color='blue', label='St. dev.')

    ax.set_xlabel('Real - NN prediction')
    ax.set_ylabel('Number of counts')

    ax.legend(loc='upper right')


def plot_amplitude_histogram(ax, x):
    n, bins, patches = ax.hist(x, 100, alpha=0.5, color='green', label='amplitude')

    ax.set_xlabel('Real - NN prediction')
    ax.set_ylabel('Number of counts')

    ax.legend(loc='upper right')


def for_future():
    ...
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
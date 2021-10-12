import numpy as np
from matplotlib import pyplot as plt
from matplotlib import table
from utils import Gauss
from scipy.stats import skew


def visualize(epoch, Y_data, Y_NN, loss_vector, sigma=1, amplitude=1, residuals=None, epochs=None):
    vis_len = 100

    In_data = np.zeros((epoch, vis_len))

    F_X = np.linspace(0, 10, vis_len, endpoint=False)
    for i in range(Y_data.shape[0]):
        In_data[i] = Gauss(F_X, Y_data[i][2], Y_data[i][0], Y_data[i][1])

    fig, ax = plt.subplots(3, 3)

    fig.suptitle(f"Sigma={sigma}, Amplitude={amplitude}", fontsize=16)
    fig.set_size_inches(16, 16)

    plot_gauss(ax[0, 0], F_X, In_data, Y_NN)
    plot_loss_function(ax[0, 1], loss_vector)
    if residuals is not None:
        plot_residuals(ax[0, 2], epochs, residuals)

    colors = ['red', 'blue', 'green']
    labels = ['Mean', 'St. dev.', 'Amplitude']

    for i in range(Y_data.shape[1]):
        plot_variable_distribution(ax[2, i], Y_data[:, i], colors[i], labels[i])
        plot_histogram(ax[1, i], (Y_NN[:, i] - Y_data[:, i]) / Y_data[:, i], colors[i], labels[i])
    fig.tight_layout()
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


def plot_histogram(ax, data, color, label):
    n, bins, patches = ax.hist(data, bins=50, alpha=0.5, color=color, label=label)

    ax.set_xlabel('Real - NN prediction')
    ax.set_ylabel('Number of counts')

    ax.legend(loc='upper right')


def plot_variable_distribution(ax, variable, color, label):
    ax.plot(variable, color=color, label=label)
    ax.set_xlabel('Argument X')
    ax.set_ylabel('Value Y')

    ax.legend(loc='upper left', prop={'size': 7})


def plot_residuals(ax, epochs, residuals):
    ax.plot(epochs, residuals[:, 0], 'r-', label="Mean")
    ax.plot(epochs, residuals[:, 1], 'b-', label="St. dev")
    ax.plot(epochs, residuals[:, 2], 'g-', label="Amplitude")
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Residual mean')
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


def track_mean_evolution(mean_Y, means_NN, step):
    fig, ax = plt.subplots(2, 3)
    fig.suptitle(f"Evolution of mean", fontsize=16)
    fig.set_size_inches(16, 8)
    row = 0
    for idx, mean_NN in enumerate(means_NN):
        if idx > 0 and idx % 3 == 0:
            row += 1

        plot_histogram(ax[row, idx % 3], (mean_NN - mean_Y) / mean_Y, 'red', f'Epoch: {(idx + 1) * step}')

    # plt.subplots_adjust(bottom=0.3)

    ax[1, 2].axis("off")
    # info_table = table.table(cellText=[[np.mean(mean), skew(mean)] for mean in means_NN],
    #                rowLabels=[step * (i + 1) for i in range(len(means_NN))],
    #                colLabels=['Mean', 'Skew'], loc='center', fontsize=50)
    info_table = ax[1, 2].table(cellText=[[round(np.mean(mean), 6), round(skew(mean), 6)] for mean in means_NN],
                                rowLabels=[step * (i + 1) for i in range(len(means_NN))],
                                colLabels=['Mean', 'Skew'], loc='center', bbox=[0, 0, 1, 1])
    info_table.auto_set_font_size(False)
    info_table.set_fontsize(15)
    cellDict = info_table.get_celld()
    # for i in range(0, 2):
    #     cellDict[(0, i)].set_height(.15)
    #     for j in range(0, 6):
    #         cellDict[(j, i)].set_height(.15)
    # bbox = [0, -0.5, 1, 0.5]
    # table.auto_set_font_size(False)
    # table.set_fontsize(20)
    # table.scale(1.5, 1.5)
    plt.show()

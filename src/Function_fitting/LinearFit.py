import numpy as np

from matplotlib import pyplot as plt
from random import random

from src.utils import apply_boundary


def linear_regression(X_data, Y_data):

    sumX = np.sum(X_data)
    sumX2 = np.sum(X_data * X_data)
    sumY = np.sum(Y_data)
    sumXY = np.sum(X_data * Y_data)

    a = ((X_data.shape[0] * sumXY) - (sumX * sumY)) / ((X_data.shape[0] * sumX2) - (sumX * sumX))
    b = (sumY - a * sumX) / len(X_data)

    return [a, b]


def linear_genetic_optimization(X_data, Y_data):
    N = 50

    params = 2

    vec, vec2 = np.empty((N * params,)), np.empty((N * params,))

    for i in range(N):
        vec[i] = 1 + (random() - 0.5)
        vec[i + N] = 10 * random()

    cost = calculate_costs(X_data, Y_data, vec, params)

    for _ in range(400):
        for j in range(N):
            e = 2 * random() - 1
            z1 = round((N - 1) * random())
            z2 = round((N - 1) * random())
            vec2[j] = vec[j] + e * (vec[z1] - vec[z2])
            vec2[j + N] = vec[j + N] + e * (vec[z1 + N] - vec[z2 + N])

            vec2[j] = apply_boundary(0, 3, vec2[j])
            vec2[j + N] = apply_boundary(0, 10, vec2[j + N])

        cost2 = calculate_costs(X_data, Y_data, vec2, params)

        for j in range(N):
            if cost2[j] < cost[j]:
                vec[j] = vec2[j]
                vec[j + N] = vec2[j + N]
                cost[j] = cost2[j]

    cost2 = calculate_costs(X_data, Y_data, vec, params)

    best_solution = np.argmin(cost2)

    return vec[best_solution], vec[best_solution + N]


def calculate_costs(X_data, Y_data, agents, params):
    N = agents.shape[0] // params
    calculated_costs = np.empty((N,))
    for i in range(N):
        cost = np.sum(np.power(X_data * agents[i] + agents[i + N] - Y_data, 2))
        calculated_costs[i] = cost

    return calculated_costs


def main():
    X_data = [not_biased_x_point + 2 * (random() - 0.5) for not_biased_x_point in range(50)]
    Y_data = [(2 * x_point + 2 + 10 * (random() - 0.5)) for x_point in X_data]
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)
    fig, ax = plt.subplots()
    ax.plot(X_data, Y_data, 'ro', label="Data points")

    a, b = linear_regression(X_data, Y_data)
    print(a, b)
    X = list(range(50))
    Y = [a * x_point + b for x_point in X]
    ax.plot(X, Y, 'green', label="Linear regression")

    a_opt, b_opt = linear_genetic_optimization(X_data, Y_data)
    print(a_opt, b_opt)
    X_opt = list(range(50))
    Y_opt = [a_opt * x_point + b_opt for x_point in X_opt]
    ax.plot(X_opt, Y_opt, 'blue', label="Differential optimization")

    ax.legend()

    a, b = round(a, 8), round(b, 8)
    a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    ax.text(0.6, 0.2, "y = " + str(a) + "x + " + str(b), horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, color='green')
    ax.text(0.6, 0.1, "y = " + str(a_opt) + "x + " + str(b_opt), horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, color='blue')

    plt.show()


if __name__ == "__main__":
    main()

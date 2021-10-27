from random import random
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt


def linear_regression(X_data: np.ndarray, Y_data: np.ndarray) -> Tuple[float, float]:
    sumX = np.sum(X_data)
    sumX2 = np.sum(X_data * X_data)
    sumY = np.sum(Y_data)
    sumXY = np.sum(X_data * Y_data)

    a = ((len(X_data) * sumXY) - (sumX * sumY)) / ((len(X_data) * sumX2) - (sumX * sumX))
    b = (sumY - a * sumX) / len(X_data)

    return a, b


def linear_genetic_optimization(X_data, Y_data) -> Tuple[float, float]:
    N = 50
    params = 2

    vec, vec2 = list(range(N * params)), list(range(N * params))

    # agents initialization
    for i in range(N):
        vec[i] = 1 + (random() - 0.5)
        vec[i + N] = 10 * random()

    cost = calculate_cost(X_data, Y_data, vec)

    for loop in range(400):
        for j in range(N):
            e = 2 * random() - 1
            z1 = round((N - 1) * random())
            z2 = round((N - 1) * random())
            vec2[j] = vec[j] + e * (vec[z1] - vec[z2])
            vec2[j + N] = vec[j + N] + e * (vec[z1 + N] - vec[z2 + N])

            vec2[j] = apply_boundary(0, 3, vec2[j])
            vec2[j + N] = apply_boundary(0, 10, vec2[j + N])

        cost2 = calculate_cost(X_data, Y_data, vec2)

        for j in range(N):
            if cost2[j] < cost[j]:
                vec[j] = vec2[j]
                vec[j + N] = vec2[j + N]
                cost[j] = cost2[j]

    cost2 = calculate_cost(X_data, Y_data, vec)

    best_solution = cost2.index(min(cost2))

    return vec[best_solution], vec[best_solution + N]


def calculate_cost(X_data, Y_data, agents) -> list:
    cost: list = []
    N = int(len(agents) / 2)
    for i in range(N):
        calculated_cost = np.sum((X_data * agents[i] + agents[i + N] - Y_data) ** 2)
        cost.append(calculated_cost)

    return cost


def apply_boundary(min, max, value):
    if value < min:
        result = min + 0.01 * (max - min) * random()
    elif value > max:
        result = max - 0.01 * (max - min) * random()
    else:
        result = value

    return result


def main():
    X_data: List[float] = [not_biased_x_point + 2 * (random() - 0.5) for not_biased_x_point in range(50)]
    Y_data: List[float] = [(2 * x_point + 2 + 10 * (random() - 0.5)) for x_point in X_data]
    X_data: np.ndarray = np.array(X_data, dtype=np.float32)
    Y_data: np.ndarray = np.array(Y_data, dtype=np.float32)

    fig, ax = plt.subplots()
    ax.plot(X_data, Y_data, 'ro', label="Data points")

    a, b = linear_regression(X_data, Y_data)
    print(a, b)
    Y = [a * x_point + b for x_point in range(50)]
    ax.plot(Y, 'green', label="Linear regression")

    a_opt, b_opt = linear_genetic_optimization(X_data, Y_data)
    print(a_opt, b_opt)
    Y_opt = [a_opt * x_point + b_opt for x_point in range(50)]
    ax.plot(Y_opt, 'blue', label="Differential optimization")

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

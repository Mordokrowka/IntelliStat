from random import random
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from IntelliStat.components.component_functions import Gauss


def gaussian_regression(X_data: np.ndarray, Y_data: np.ndarray) -> Tuple[float, float, float]:
    # Hongwei G., A simple algorithm for fitting a gaussian function,
    # IEEE Signal Processing Magazine 28(5) 2011

    Y_dat = np.log(Y_data)

    sumX = np.sum(X_data)
    sumX2 = np.sum(X_data ** 2)
    sumX3 = np.sum(X_data ** 3)
    sumX4 = np.sum(X_data ** 4)
    sumlogY = np.sum(Y_dat)
    sumXlogY = np.sum(X_data * Y_dat)
    sumX2logY = np.sum((X_data ** 2) * Y_dat)

    M = np.array([[len(X_data), sumX, sumX2], [sumX, sumX2, sumX3], [sumX2, sumX3, sumX4]])
    X = np.array([sumlogY, sumXlogY, sumX2logY])
    pol_coeffs = np.linalg.inv(M).dot(X)

    A = np.exp(pol_coeffs[0] - (pol_coeffs[1] ** 2) / (4 * pol_coeffs[2]))
    u = - pol_coeffs[1] / (2 * pol_coeffs[2])
    pol_coeffs[2] = - abs(pol_coeffs[2])
    sigma = np.sqrt(- 1 / (2 * pol_coeffs[2]))

    return A.mean(), u.mean(), sigma.mean()


def gauss_genetic_optimization(X_data: np.ndarray, Y_data: np.ndarray) -> Tuple[float, float, float]:
    N = 50
    params = 3

    vec, vec2 = list(range(N * params)), list(range(N * params))

    # agents initialization
    for i in range(N):
        vec[i] = 100 + 100 * (random() - 0.5)
        vec[i + N] = 50 * random()
        vec[i + 2 * N] = 10 * random()

    cost: List[float] = calculate_cost(X_data, Y_data, vec)

    for _ in range(400):
        for j in range(N):
            e = 2 * random() - 1
            z1 = round((N - 1) * random())
            z2 = round((N - 1) * random())
            vec2[j] = vec[j] + e * (vec[z1] - vec[z2])
            vec2[j + N] = vec[j + N] + e * (vec[z1 + N] - vec[z2 + N])
            vec2[j + 2 * N] = vec[j + 2 * N] + e * (vec[z1 + 2 * N] - vec[z2 + 2 * N])

            vec2[j] = apply_boundary(0, 5000, vec2[j])
            vec2[j + N] = apply_boundary(0, 50, vec2[j + N])
            vec2[j + 2 * N] = apply_boundary(0, 25, vec2[j + 2 * N])

        cost2: List[float] = calculate_cost(X_data, Y_data, vec2)

        for j in range(N):
            if cost2[j] < cost[j]:
                vec[j] = vec2[j]
                vec[j + N] = vec2[j + N]
                vec[j + 2 * N] = vec2[j + 2 * N]
                cost[j] = cost2[j]

    cost2: List[float] = calculate_cost(X_data, Y_data, vec)

    best_solution: int = cost2.index(min(cost2))

    return vec[best_solution], vec[best_solution + N], vec[best_solution + 2 * N]


def calculate_cost(X_data: np.ndarray, Y_data: np.ndarray, agents) -> List[float]:
    cost: list = []
    N: int = len(agents) // 3
    for i in range(N):
        calculated_cost = np.sum((Gauss(X_data, agents[i], agents[i + N], agents[i + 2 * N]) - Y_data) ** 2)
        cost.append(calculated_cost)

    return cost


def apply_boundary(minimum: float, maximum: float, value: float) -> float:
    if value < minimum:
        result = minimum + 0.01 * (maximum - minimum) * random()
    elif value > maximum:
        result = maximum - 0.01 * (maximum - minimum) * random()
    else:
        result = value

    return result


def gaussian_method_of_moments(X_data: np.ndarray, Y_data: np.ndarray) -> tuple:

    sumY = np.sum(Y_data) * (max(X_data) - min(X_data)) / len(Y_data)

    u = np.sum(X_data * Y_data / sumY)

    sigma = np.sqrt(np.sum((((X_data - u) ** 2) * Y_data) / sumY))

    A = np.sum(sumY * (1 / (sigma * np.sqrt(2 * np.pi))))

    return A, u, sigma


def main():
    X_data: List[float] = [biased_x_point + 1 * (random() - 0.5) for biased_x_point in range(50)]
    X_data: np.ndarray = np.array(X_data, dtype=np.float32)

    Y_data: np.ndarray = np.abs(Gauss(X_data, 100, 25, 5) + 0 * (random() - 0.5))

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    print("Regression:")
    ax.plot(X_data, Y_data, 'ko', label="Data points")

    A, u, sigma = gaussian_regression(X_data, Y_data)
    print("A: ", A, ", mean: ", u, ", sigma: ", sigma)

    X = np.linspace(0, 50, 501)
    Y = [Gauss(x_point, A, u, sigma) for x_point in X]
    ax.plot(X, Y, 'blue', label="Gaussian regression")
    ax.legend(loc='upper left', prop={'size': 7})

    A, u, sigma = round(A, 8), round(u, 8), round(sigma, 8)
    ax.text(0.45, 1.06, "A: " + str(A) + ", mean: " + str(u) + ", sigma: " + str(sigma),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='blue')

    print("Method of moments:")
    ax2.plot(X_data, Y_data, 'ko', label="Data points")

    A_m, u_m, sigma_m = gaussian_method_of_moments(X_data, Y_data)
    print("A: ", A_m, ", mean: ", u_m, ", sigma: ", sigma_m)

    X_m = np.linspace(0, 50, 501)
    Y_m = [Gauss(x_point, A_m, u_m, sigma_m) for x_point in X_m]

    ax2.plot(X_m, Y_m, 'green', label="Method of moments")
    ax2.legend(loc='upper left', prop={'size': 7})
    A_m, u_m, sigma_m = round(A_m, 8), round(u_m, 8), round(sigma_m, 8)
    ax2.text(1.65, 1.06, "A: " + str(A_m) + ", mean: " + str(u_m) + ", sigma: " + str(sigma_m),
             horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes, color='green')

    print("Differential optimization:")
    ax3.plot(X_data, Y_data, 'ko', label="Data points")

    A_opt, u_opt, sigma_opt = gauss_genetic_optimization(X_data, Y_data)
    print("A: ", A_opt, ", mean: ", u_opt, ", sigma: ", sigma_opt)

    X_opt = np.linspace(0, 50, 501)
    Y_opt = [Gauss(x_point, A_opt, u_opt, sigma_opt) for x_point in X_opt]

    ax3.plot(X_opt, Y_opt, 'red', label="Diff. optimization")
    ax3.legend(loc='upper left', prop={'size': 7})
    A_opt, u_opt, sigma_opt = round(A_opt, 8), round(u_opt, 8), round(sigma_opt, 8)
    ax3.text(2.85, 1.06, "A: " + str(A_opt) + ", mean: " + str(u_opt) + ", sigma: " + str(sigma_opt),
             horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes, color='red')

    plt.show()


if __name__ == "__main__":
    main()

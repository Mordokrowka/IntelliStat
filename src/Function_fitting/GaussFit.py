import numpy as np

from random import random
from matplotlib import pyplot as plt

from src.utils import Gauss, apply_boundary


def gaussian_regression(X_data, Y_data):
    # Hongwei G., A simple algorithm for fitting a gaussian function,
    # IEEE Signal Processing Magazine 28(5) 2011

    Y_dat = np.log(Y_data)
    sumX = np.sum(X_data)
    sumX2 = np.sum(np.power(X_data, 2))
    sumX3 = np.sum(np.power(X_data, 3))
    sumX4 = np.sum(np.power(X_data, 4))
    sumlogY = np.sum(Y_dat)
    sumXlogY = np.sum(X_data * Y_dat)
    sumX2logY = np.sum(np.power(X_data, 2) * Y_dat)

    M = np.array([[X_data.shape[0], sumX, sumX2], [sumX, sumX2, sumX3], [sumX2, sumX3, sumX4]])
    X = np.array([sumlogY, sumXlogY, sumX2logY])
    pol_coeffs = np.linalg.inv(M).dot(X)

    A = np.exp(pol_coeffs[0] - np.power(pol_coeffs[1], 2) / (4 * pol_coeffs[2]))
    u = - pol_coeffs[1] / (2 * pol_coeffs[2])
    pol_coeffs[2] = - abs(pol_coeffs[2])
    sigma = np.sqrt(- 1 / (2 * pol_coeffs[2]))

    return [A, u, sigma]


def gauss_genetic_optimization(X_data, Y_data):
    N = 50
    params = 3

    vec, vec2 = np.empty((N * params,)), np.empty((N * params,))

    # agents initialization
    for i in range(N):
        vec[i] = 100 + 100 * (random() - 0.5)
        vec[i + N] = 50 * random()
        vec[i + 2 * N] = 10 * random()

    costs = calculate_costs(X_data, Y_data, vec, params)

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

        costs2 = calculate_costs(X_data, Y_data, vec2, params)

        for j in range(N):
            if costs2[j] < costs[j]:
                vec[j] = vec2[j]
                vec[j + N] = vec2[j + N]
                vec[j + 2 * N] = vec2[j + 2 * N]
                costs[j] = costs2[j]

    cost2 = calculate_costs(X_data, Y_data, vec, params)
    best_solution = np.argmin(cost2)

    return vec[best_solution], vec[best_solution + N], vec[best_solution + 2 * N]


def calculate_costs(X_data, Y_data, agents, params):
    N = agents.shape[0] // params
    calculated_costs = np.empty((N,))

    for i in range(N):
        cost = np.sum(np.power(Gauss(X_data, agents[i], agents[i + N], agents[i + 2 * N]) - Y_data, 2))
        calculated_costs[i] = cost

    return calculated_costs


def gaussian_method_of_moments(X_data, Y_data):
    sumY = np.sum(Y_data)

    sumY = sumY * (np.max(X_data) - np.min(X_data)) / len(Y_data)
    u = np.sum(X_data * Y_data) / sumY

    sigma = np.sum((np.power(X_data - u, 2) * Y_data)) / sumY
    sigma = np.sqrt(sigma)
    A = sumY * (1 / (sigma * np.sqrt(2 * np.pi)))

    return [A, u, sigma]


def main():
    X_data = [biased_x_point + 1 * (random() - 0.5) for biased_x_point in range(50)]
    Y_data = [abs(Gauss(x_point, 100, 25, 5) + 0 * (random() - 0.5)) for x_point in X_data]
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax.plot(X_data, Y_data, 'ko', label="Data points")
    A, u, sigma = gaussian_regression(X_data, Y_data)
    print("Regression:")
    print("A: ", A, ", mean: ", u, ", sigma: ", sigma)
    X = np.linspace(0, 50, 501)
    Y = [Gauss(x_point, A, u, sigma) for x_point in X]
    ax.plot(X, Y, 'blue', label="Gaussian regression")
    ax.legend(loc='upper left', prop={'size': 7})

    A, u, sigma = round(A, 8), round(u, 8), round(sigma, 8)
    # a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    ax.text(0.45, 1.06, "A: " + str(A) + ", mean: " + str(u) + ", sigma: " + str(sigma),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='blue')

    ax2.plot(X_data, Y_data, 'ko', label="Data points")
    A_m, u_m, sigma_m = gaussian_method_of_moments(X_data, Y_data)
    print("Method of moments:")
    print("A: ", A_m, ", mean: ", u_m, ", sigma: ", sigma_m)
    X_m = np.linspace(0, 50, 501)
    Y_m = [Gauss(x_point, A_m, u_m, sigma_m) for x_point in X_m]
    ax2.plot(X_m, Y_m, 'green', label="Method of moments")
    leg = ax2.legend(loc='upper left', prop={'size': 7})
    A_m, u_m, sigma_m = round(A_m, 8), round(u_m, 8), round(sigma_m, 8)
    # a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    ax2.text(1.65, 1.06, "A: " + str(A_m) + ", mean: " + str(u_m) + ", sigma: " + str(sigma_m),
             horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes, color='green')

    ax3.plot(X_data, Y_data, 'ko', label="Data points")
    A_opt, u_opt, sigma_opt = gauss_genetic_optimization(X_data, Y_data)
    print("Differential optimization:")
    print("A: ", A_opt, ", mean: ", u_opt, ", sigma: ", sigma_opt)
    X_opt = np.linspace(0, 50, 501)
    Y_opt = [Gauss(x_point, A_opt, u_opt, sigma_opt) for x_point in X_opt]
    ax3.plot(X_opt, Y_opt, 'red', label="Diff. optimization")
    leg = ax3.legend(loc='upper left', prop={'size': 7})
    A_opt, u_opt, sigma_opt = round(A_opt, 8), round(u_opt, 8), round(sigma_opt, 8)
    # a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    ax3.text(2.85, 1.06, "A: " + str(A_opt) + ", mean: " + str(u_opt) + ", sigma: " + str(sigma_opt),
             horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes, color='red')
    plt.show()


if __name__ == "__main__":
    main()

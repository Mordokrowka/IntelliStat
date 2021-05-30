import numpy as np


def Gauss(x, A, u, sigma):
    return A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))


def apply_boundary(min_value, max_value, value):
    if value < min_value:
        result = min_value + 0.01 * (max_value - min_value) * np.random.random()
    elif value > max_value:
        result = max_value - 0.01 * (max_value - min_value) * np.random.random()
    else:
        result = value

    return result

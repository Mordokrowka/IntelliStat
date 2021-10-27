from random import random

import numpy as np


def Gauss(x: np.ndarray, A: float, u, sigma) -> np.ndarray:
    return A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))


def Exp(x: np.ndarray, B: float = 0.7 + 0.6 * random(), b: float = 0.5 + random()) -> np.ndarray:
    return B * np.exp(-b * x)


def GG(x: np.ndarray, A1: float = 0.7 + 0.6 * random(), u1: float = 1 + 8 * random(),
       sigma1: float = 0.1 + 0.4 * random(), A2: float = 0.7 + 0.6 * random(), u2: float = 1 + 8 * random(),
       sigma2: float = 0.1 + 0.4 * random()) -> np.ndarray:
    return Gauss(x, A1, u1, sigma1) + Gauss(x, A2, u2, sigma2)


def GGG(x: np.ndarray, A1: float = 0.7 + 0.6 * random(), u1: float = 1 + 8 * random(),
        sigma1: float = 0.1 + 0.4 * random(), A2: float = 0.7 + 0.6 * random(), u2: float = 1 + 8 * random(),
        sigma2: float = 0.1 + 0.4 * random(), A3: float = 0.7 + 0.6 * random(), u3: float = 1 + 8 * random(),
        sigma3: float = 0.1 + 0.4 * random()) -> np.ndarray:
    return Gauss(x, A1, u1, sigma1) + Gauss(x, A2, u2, sigma2) + Gauss(x, A3, u3, sigma3)


def GGE(x: np.ndarray, A1: float = 0.7 + 0.6 * random(), u1: float = 1 + 8 * random(),
        sigma1: float = 0.1 + 0.4 * random(), A2: float = 0.7 + 0.6 * random(),
        u2: float = 1 + 8 * random(), sigma2: float = 0.1 + 0.4 * random(), B1: float = 0.7 + 0.6 * random(),
        b1: float = 0.5 + random()) -> np.ndarray:
    return Gauss(x, A1, u1, sigma1) + Gauss(x, A2, u2, sigma2) + Exp(x, B1, b1)


def GE(x: np.ndarray, A1: float = 0.7 + 0.6 * random(), u1: float = 1 + 8 * random(),
       sigma1: float = 0.1 + 0.4 * random(), B1: float = 0.7 + 0.6 * random(),
       b1: float = 0.5 + random()) -> np.ndarray:
    return Gauss(x, A1, u1, sigma1) + Exp(x, B1, b1)


def multiG(x: np.ndarray, n: int = 0) -> np.ndarray:
    multi_g: np.ndarray = np.zeros(x.shape)
    for it in range(n):
        u = 1 + 8 * random()
        A = 0.7 + 0.6 * random()
        sigma = 0.2 + 0.5 * random()
        multi_g += Gauss(x, A, u, sigma)
    return multi_g

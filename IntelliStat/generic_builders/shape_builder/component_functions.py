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


def GGE(x: np.ndarray, A1: float = 0.7 + 0.6 * random(), u1: float = 1 + 8 * random(),
        sigma1: float = 0.1 + 0.4 * random(), A2: float = 0.7 + 0.6 * random(),
        u2: float = 1 + 8 * random(), sigma2: float = 0.1 + 0.4 * random(), B1: float = 0.7 + 0.6 * random(),
        b1: float = 0.5 + random()) -> np.ndarray:
    return Gauss(x, A1, u1, sigma1) + Gauss(x, A2, u2, sigma2) + Exp(x, B1, b1)

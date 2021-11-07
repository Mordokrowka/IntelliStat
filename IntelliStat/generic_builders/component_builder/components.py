from random import random

import numpy as np


def Gauss(x: np.ndarray, A: float, u, sigma) -> np.ndarray:
    return A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))


def Exp(x: np.ndarray, B: float = 0.7 + 0.6 * random(), b: float = 0.5 + random()) -> np.ndarray:
    return B * np.exp(-b * x)

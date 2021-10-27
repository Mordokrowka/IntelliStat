from random import random

import numpy as np


def Gauss(x, A: float, u, sigma):
    return A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))


def Exp(x, B: float = 0.7 + 0.6 * random(), b: float = 0.5 + random()):
    return B * np.exp(-b * x)


def GG(x, A1: float = 0.7 + 0.6 * random(), u1: float = 1 + 8 * random(), sigma1: float = 0.1 + 0.4 * random(),
       A2: float = 0.7 + 0.6 * random(), u2: float = 1 + 8 * random(), sigma2: float = 0.1 + 0.4 * random()):
    return Gauss(x, A1, u1, sigma1) + Gauss(x, A2, u2, sigma2)


def GGG(x, A1: float = 0.7 + 0.6 * random(), u1: float = 1 + 8 * random(), sigma1: float = 0.1 + 0.4 * random(),
        A2: float = 0.7 + 0.6 * random(), u2: float = 1 + 8 * random(), sigma2: float = 0.1 + 0.4 * random(),
        A3: float = 0.7 + 0.6 * random(), u3: float = 1 + 8 * random(), sigma3: float = 0.1 + 0.4 * random()):
    return Gauss(x, A1, u1, sigma1) + Gauss(x, A2, u2, sigma2) + Gauss(x, A3, u3, sigma3)


def GGE(x, A1: float = 0.7 + 0.6 * random(), u1: float = 1 + 8 * random(), sigma1: float = 0.1 + 0.4 * random(),
        A2: float = 0.7 + 0.6 * random(), u2: float = 1 + 8 * random(), sigma2: float = 0.1 + 0.4 * random(),
        B1: float = 0.7 + 0.6 * random(), b1: float = 0.5 + random()):
    print(x.shape)
    return Gauss(x, A1, u1, sigma1) + Gauss(x, A2, u2, sigma2) + Exp(x, B1, b1)


def GE(x, A1: float = 0.7 + 0.6 * random(), u1: float = 1 + 8 * random(), sigma1: float = 0.1 + 0.4 * random(),
       B1: float = 0.7 + 0.6 * random(), b1: float = 0.5 + random()):
    return Gauss(x, A1, u1, sigma1) + Exp(x, B1, b1)


# TODO change `sum` name
def multiG(x, n):
    sum = 0
    for it in range(n):
        u = 1 + 8 * random()
        A = 0.7 + 0.6 * random()
        sigma = 0.2 + 0.5 * random()
        sum += Gauss(x, A, u, sigma)
    x = sum
    return x

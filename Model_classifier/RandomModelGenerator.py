from torch.utils.data import Dataset
import numpy as np
import math
from random import random

def class_option(option):
    return {
        0 : 'Gauss',
        1 : 'Gauss+Gauss',
        2 : 'Gauss+Exp',
        3 : 'Exp',
    }[option]

def Gauss(x, A, u, sigma ):
    for i in range(len(x)):
        x[i] = A * np.exp(-np.power(x[i] - u, 2) / (2 * np.power(sigma, 2)))
    return x
def Exp(x, A, b):
    for i in range(len(x)):
        x[i] = A * math.exp(-b * x[i])
    return x

def data_generator(X, option_model):
    return {
        'Gauss' : Gauss(X, 1, 3 + 4*random(), 0.2 + random()),
        'Gauss+Gauss' : Gauss(X, 1, 3 + random(), 0.2 + random()) + Gauss(X, 1, 6 + random(), 0.2 + random()),
        'Gauss+Exp' : Gauss(X, 1, 3 + 4*random(), 0.2 + random() ) + Exp(X, 1, 0.5 + random() ),
        'Exp' : Exp(X, 1, 0.5 + random() ),
    }[option_model]

def generate_data(X_data, option):

    # Gauss, class 0
    option_model = class_option(option)
    X_data = data_generator(X_data, option_model)

    return X_data


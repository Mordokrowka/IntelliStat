from torch.utils.data import Dataset
import numpy as np
import math
from random import random

def Gauss( x, A, u, sigma ):
    return  A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))

def generate_data(X_data, Y_data, rate):

    X_data = np.array(X_data, dtype = np.float32)
    Y_data = np.array(Y_data, dtype = np.longlong)

    # Gauss, class 0
    for i in range(500):
        u = 4.5 + random()
        std = 0.5 + 0.5*random()
        for j in range(20):
            X_data[i][j] = Gauss(X_data[i][j], 1, u, std )
        Y_data[i] = 0
    #Gauss + Exp, class 2
    for i in range(500,1000):
        u = 4.5 + random()
        std = 0.5 + 0.5*random()
        b = 0.4 + 0.2 * random()
        for j in range(20):
            X_data[i][j] = Gauss(X_data[i][j], 1, u, std ) + math.exp(-b * X_data[i][j])
        Y_data[i] = 2
    #Exp, class 3
    for i in range(1000,1500):
        b = 0.4 + 0.2 * random()
        for j in range(20):
            X_data[i][j] = math.exp(-b * X_data[i][j])
        Y_data[i] = 2
    #Gauss + Gauss, clas 1
    for i in range(1500,2000):
        u1 = 3.5 + random()
        std1 = 0.5 + 0.5*random()
        u2 = 6.5 + random()
        std2 = 0.5 + 0.5*random()
        for j in range(20):
            X_data[i][j] = Gauss(X_data[i][j], 1, u1, std1 ) + Gauss(X_data[i][j], 1, u2, std2 )
        Y_data[i] = 1
    return X_data, Y_data

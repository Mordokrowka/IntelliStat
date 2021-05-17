import numpy as np
from random import random
import torch
from matplotlib import pyplot as plt

from ShapeCreator import ShapeCreator
from ENN import ENN

def Gauss( x, A, u, sigma ):
    return  A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))


def main():
    EvolutionalNN = ENN(20, 20, 10, 5, 2)

    X_data = [[X/2 for X in range(20)] for it in range(300)]
    #X_data = np.array(X_data, dtype=np.float32)
    #Y_tensor = torch.tensor(Y_tensor)
    Y_data = [ [2*random() + 4, 1] for X in X_data]
    X_data = np.array(X_data, dtype = np.float32)
    Y_data = np.array(Y_data, dtype = np.float32)
    #print(X_data)
    #print(Y_data)
    #X_tensor = X_data #[[x] for x in X_data]

    Dataset = ShapeCreator(X_data, "Gauss", Y_data)
    EvolutionalNN.train(Dataset, 400, 20 )

    #In_data = X_data
    #print(X_data)
    #for i in range(len(X_data)):
    #    for j in range(len(X_data[0])):
    #        In_data[i][j] = Gauss(X_data[i][j], 1, Y_data[i][0], Y_data[i][1])
        #print(In_data[i])
        #print(Y_data[i])
    X_NN = torch.tensor(X_data)
    Y_NN = EvolutionalNN.model(X_NN)
    Y_NN = Y_NN.detach().numpy()
    #print(X_NN)
    for i in range(len(Y_data)):
        print("Mean, real : ", Y_data[i][0], " , trained: ", Y_NN[i][0])
        print("Std, real : ", Y_data[i][1], " , trained: ", Y_NN[i][1])

    vis_len = 100
    In_data = [[0 for t2 in range(vis_len)] for t1 in range(len(X_data))]
    F_X = np.linspace(0, 10, vis_len, endpoint=False)
    for i in range(len(X_data)):
        for j in range(vis_len):
            In_data[i][j] = Gauss(F_X[j], 1, Y_data[i][0], Y_data[i][1])

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[1], 'k-', label="Gauss no. 1")
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[2], 'r-', label="Gauss no. 2")
    ax.plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[3], 'b-', label="Gauss no. 3")

    ax.plot(Y_NN[1][0], Gauss(Y_NN[1][0], 1, Y_NN[1][0], Y_NN[1][1]), 'k*', label="Predicted mean")
    ax.plot(Y_NN[2][0], Gauss(Y_NN[2][0], 1, Y_NN[2][0], Y_NN[2][1]), 'r*', label="Predicted mean")
    ax.plot(Y_NN[3][0], Gauss(Y_NN[3][0], 1, Y_NN[3][0], Y_NN[3][1]), 'b*', label="Predicted mean")
    #print("Regression:")
    #print("Red, real mean: ", Y_data[1][0] ,", NN output: ", Y_NN[1][0])
    #X = np.linspace(0,50,501)
    #Y = [ gauss(x_point, A, u, sigma) for x_point in X]
    #ax.plot(X_data, Y_NN, 'blue', label = "Evolutional NN")
    leg = ax.legend(loc = 'upper left', prop={'size':7})

    #A, u, sigma = round(A, 8), round(u, 8), round(sigma, 8)
    #a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    ax.text(0.45, 1.12, " Mean: " + str(round(Y_data[1][0],3)) + ", NN prediction: " + str(round(Y_NN[1][0],3)),
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes, color = 'k')
    ax.text(0.45, 1.07, " Mean: " + str(round(Y_data[2][0], 3)) + ", NN prediction: " + str(round(Y_NN[2][0], 3)),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='r')
    ax.text(0.45, 1.02, " Mean: " + str(round(Y_data[3][0], 3)) + ", NN prediction: " + str(round(Y_NN[3][0], 3)),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='b')
    plt.show()

if __name__ == "__main__" :
    main()

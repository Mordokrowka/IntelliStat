import numpy as np
from random import random
import torch
from torch.optim import Adam

from matplotlib import pyplot as plt

class ENN():
    def __init__ (self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1,5),
            torch.nn.ReLU(),
            torch.nn.Linear(5,1)
        )
        self.optimizer = Adam(self.model.parameters(), lr = 0.001)

    def train(self, x, y, epoch):
        self.model.zero_grad()
        output = self.model(x)
        loss = torch.nn.MSELoss(output, y)
        loss.backward()
        self.optimizer.step()

def main():
    EvolutionalNN = ENN()
    for param_tensor in EvolutionalNN.model.state_dict():
        print(param_tensor, "\t", EvolutionalNN.model.state_dict()[param_tensor][0])
    #for param in EvolutionalNN.model.parameters():
    #    print(param)
    X_data = [not_biased_x_point + 3 * (random() - 0.5) for not_biased_x_point in range(50)]
    Y_data = [ (2 * x_point  + 2 + 10 * (random() - 0.5) ) for x_point in X_data]
    X_tensor = [[x] for x in X_data]
    X_tensor = torch.tensor(X_tensor)
    Y_NN = EvolutionalNN.model(X_tensor)
    Y_NN = Y_NN.detach().numpy()

    fig, ax = plt.subplots()
    ax.plot(X_data, Y_data, 'ko', label = "Data points")
    #print("Regression:")
    #print("A: ",A,", mean: ", u,", sigma: ", sigma)
    #X = np.linspace(0,50,501)
    #Y = [ gauss(x_point, A, u, sigma) for x_point in X]
    ax.plot(X_data, Y_NN, 'blue', label = "Evolutional NN")
    leg = ax.legend(loc = 'upper left', prop={'size':7})

    #A, u, sigma = round(A, 8), round(u, 8), round(sigma, 8)
    #a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    #ax.text(0.45, 1.06, "A: " + str(A) + ", mean: " + str(u) + ", sigma: " + str(sigma),
    #    horizontalalignment='center', verticalalignment='center',
    #    transform=ax.transAxes, color = 'blue')
    plt.show()

if __name__ == "__main__" :
    main()

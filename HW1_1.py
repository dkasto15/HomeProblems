import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from random import *


def main():
    ''' Task 1 '''
    # # # a) # # #

    # # Ensemble average # #
    N = 100000 # nbr of realizations
    t = 1
    x_ens = 0
    for i in range(N):
        w = uniform(2, 6) # Uniformly randomized number between 2 and 6
        if i == 0:
            w1 = w
        x_ens += 2 * np.sin(w * t) # Stoch. proc. val. at arb. time point
    x_ens = 1 / float(N) * x_ens
    print('1a)', x_ens)
    print('1a), analytical', 1 / (2 * float(t)) * (np.cos(2 * t) - np.cos(6 * t)))

    # # Time average # #
    M = 100000 # nbr of timesteps
    # w = uniform(2, 6) # Uniformly randomized number between 2 and 6
    t = np.linspace(0,100000,M) # vector with discrete timesteps
    x_t = 1 / float(M) * np.sum(2 * np.sin(w1 * t))
    print('1a)', x_t)

    # # ^Ergodic means timeAv=ensemAv, this is not the case here. # #

    # # # b) # # #

    w0 = 1
    # # Ensemble average # #
    N = 100000 # nbr of realizations
    t = 1
    x_ens = 0
    for i in range(N):
        theta = uniform(0, 2 * np.pi) # Uniformly randomized number between 0 and 2*pi
        if i == 0:
            theta1 = theta
        x_ens += 2 * np.sin(w0 * t + theta) # Stoch. proc. val. at arb. time point
    x_ens = 1 / float(N) * x_ens
    print('1b)', x_ens)

    # # Time average # #
    M = 100000 # nbr of timesteps
    # w = uniform(2, 6) # Uniformly randomized number between 2 and 6
    t = np.linspace(0,100000,M) # vector with discrete timesteps
    x_t = 1 / float(M) * np.sum(2 * np.sin(w0 * t + theta1))
    print('1b)', x_t)

    # # ^Ergodic means timeAv=ensemAv, this is the case here. # #

if __name__ == '__main__':
    main()

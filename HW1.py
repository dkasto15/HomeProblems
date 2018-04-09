import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from random import *


def main():

    ''' Variables and constants '''

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

    # # Time average # #
    M = 100000 # nbr of timesteps
    # w = uniform(2, 6) # Uniformly randomized number between 2 and 6
    t = np.linspace(0,1,M) # vector with discrete timesteps
    x_t = 1 / float(M) * np.sum(2 * np.sin(w1 * t))
    print('1a)', x_t)

    # # ^Ergodic means timeAv=ensemAv, this is not the case here. # #

    # # # b) # # #

    w0 = 1
    # # Ensemble average # #
    N = 10000 # nbr of realizations
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
    M = 10000 # nbr of timesteps
    # w = uniform(2, 6) # Uniformly randomized number between 2 and 6
    t = np.linspace(0,1,M) # vector with discrete timesteps
    x_t = 1 / float(M) * np.sum(2 * np.sin(w0 * t + theta1))
    print('1b)', x_t)

    # # ^Ergodic means timeAv=ensemAv, this is not the case here. # #

    ''' Task 2 '''
    # # # 1.a) # # #
    W = [[1/4.0, 0, 1/8.0], [1/2.0, 1/2.0, 3/4.0], [1/4.0, 1/2.0, 1/8.0]]

    # # # 1.b) # # #
    distr_0 = [500, 300, 200]
    distr_1 = np.dot(W, distr_0)
    print('2.1b)', distr_1)

    # # # 1.c) # # #
    # W = np.array([[0, 0.25, 0.25], [1/2.0, 1/2.0, 0.25], [0.5, 0.25, 0.5]]) # From test exself.
    eig_W_r, eig_vec_r = eig(W) # right eigenvalues/-vectors
    max_ind = np.argmax(eig_W_r)  # index of eigenvalue with value 1
    P_st = eig_vec_r[:, max_ind]
    norm = np.sum(P_st)
    P_st = P_st / norm
    print(P_st, np.sum(P_st))
    print(np.dot(W, P_st))


    # # # 1.d) # # #
    eig_W_l, eig_vec_l = eig(W, left=True, right=False) # left eigenvalues/-vectors
    print(eig_W_r)
    for i in range(len(W)):
        if i != max_ind:
            print(eig_W_r[i])
            a = np.array(eig_vec_r[:, i].reshape(3,1)) # enable mat mult
            b = np.array(eig_vec_l[:, i].reshape(1,3))  # enable mat mult
            B = np.dot(a, b)
            print(B)



    eig_vec_r = []

    ''' Plotting '''
    # fig_1 = plt.figure()
    # ax_potential = fig_1.add_subplot(111)
    # label1 = 'Calculated distribution. \nEnergy: ' \
    #     + str(np.round(np.real(energy_min), 6)) + ' atomic units'
    # label2 = 'Theoretical distribution. \nEnergy: ' \
    #     + str(-0.5) + ' atomic units'
    # ax_potential.plot(r, 4 * np.pi * r**2 * abs(phi)**2, label=label1)
    # ax_potential.plot(r, 4 * np.pi * r**2 * abs(phi_s_H)**2, '--', label=label2)
    # ax_potential.set_xlabel('Radial coordinate [atomic units]')
    # ax_potential.set_ylabel('Probability distribution function')
    # ax_potential.set_title('Probability distribution function for simulated ground state hydrogen\n' +
    #                        'compared with analytical solution')
    # ax_potential.legend(loc=1)
    #
    # plt.savefig('HA3/RadProb.eps')
    # plt.savefig('HA3/RadProb.png')
    # plt.show()

    ''' Write data to file '''
    # with open("calculation_outputs.txt", "w") as textfile:
    #     textfile.write("Minimum energy: " + str(np.real(energy_min)) + "\n")
    #     textfile.write("Normalization: " + str(np.real(trapz(phi_min**2, r))) + "\n")

    ''' Functions '''
    # def V_LJ(r12):
    # ''' Function that calculates the Lennard-Jones (LJ) potential between two atoms'''
    # global epsilon
    # global sigma
    # return 4*epsilon*((sigma/r12)**12 - (sigma/r12)**6)

if __name__ == '__main__':
    main()

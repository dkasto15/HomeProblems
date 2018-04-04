import numpy as np
import matplotlib.pyplot as plt
from random import *


def main():

    ''' Variables and constants '''

    ''' Task 1 '''
    # # # a) # # #


    # # Ensemble average # #
    N = 1000 # nbr of realizations
    t = 1
    x_ens = 0
    for i in range(N):
        w = uniform(2, 6) # Uniformly randomized number between 2 and 6
        x_ens += 2 * np.sin(w*t) # Stoch. proc. val. at arb. time point
    x_ens = 1 / float(N) * x_ens
    print(x_ens)

    # # Time average # #
    M = 1000 # nbr of timesteps
    w = uniform(2, 6) # Uniformly randomized number between 2 and 6
    t = np.linspace(0,1,N) # vector with discrete timesteps
    x_t = 1 / float(M) * np.sum(2 * np.sin(w*t))
    print(x_t)



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

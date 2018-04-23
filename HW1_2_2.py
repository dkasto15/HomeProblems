import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
import matplotlib.pyplot as plt

def main():
    ''' 2.2.1b) Calculate the stationary state for the matrix G'''
    G = np.array([[-1, 1/4.0, 0, 0],
                  [1, -1, 1/4.0, 0],
                  [0, 3/4.0, -1, 1],
                  [0, 0, 3/4.0, -1]])
    eig_G, eig_vec_G = eig(G)
    ind_st = np.argmin(abs(eig_G))
    lambda_st = eig_G[ind_st]
    P_st = eig_vec_G[:, ind_st]
    P_st = P_st / sum(P_st) # normalization
    print('2.2.1b) P_st: ', P_st)

    ''' 2.2.2b) Calculate the stationary state for the matrix G'''
    G = np.array([[-3, 4, 1],
                  [1, -5, 1],
                  [2, 1, -2]])
    eig_G, eig_vec_G = eig(G, right=True)
    ind_st = np.argmin(abs(eig_G)) # Get the 0
    lambda_st = eig_G[ind_st]
    P_st = eig_vec_G[:, ind_st]
    P_st = P_st / sum(P_st) # normalization
    print('2.2.1b) P_st: ', P_st)

    ''' 2.2.2c) Time evolution of P '''
    P_0 = np.array([1, 0, 0]) # init. prob.
    t_0 = 0 # init. time
    t_f = 1.75 # final time
    dt = 0.01 # time step
    t_vec = np.arange(t_0, t_f, dt)

    P = np.zeros([len(t_vec), 3])
    for t in range(len(t_vec)):
        P[t, :] = np.dot(expm(t_vec[t]*G), P_0)

    print('P_st simulated: ', P[-1, :])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t_vec, P[:, 0], label='P_1_st: ' + str(round(P[-1, 0], 3)), color='blue')
    ax.plot(t_vec, P[:, 1], label='P_2_st: ' + str(round(P[-1, 1], 3)), color='red')
    ax.plot(t_vec, P[:, 2], label='P_3_st: ' + str(round(P[-1, 2], 3)), color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.set_xlim([t_vec[0], t_vec[-1]])
    ax.set_ylim([0, 1])
    ax.minorticks_on()
    ax.grid(True, which='major')
    ax.grid(True, which='minor', linestyle='--')
    ax.legend()
    fig.savefig('timeEv.png', bbox='tight')

    plt.show()

if __name__ == '__main__':
    main()

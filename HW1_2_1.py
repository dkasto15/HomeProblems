import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from random import *

def main():
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
    print('2.1c)', P_st, np.sum(P_st))
    print('2.1c)', np.dot(W, P_st))


    # # # 1.d) # # #
    eig_W_l, eig_vec_l = eig(W, left=True, right=False) # left eigenvalues/-vectors
    for i in range(len(W)):
        if i != max_ind:
            print('2.1d: lambda', eig_W_r[i])
            a = np.array(eig_vec_r[:, i].reshape(3,1)) # enable mat mult
            b = np.array(eig_vec_l[:, i].reshape(1,3))  # enable mat mult
            B = np.dot(a, b)
            print('2.1d: B', B)

if __name__ == '__main__':
    main()

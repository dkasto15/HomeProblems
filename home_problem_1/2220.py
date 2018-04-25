import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
import matplotlib.pyplot as plt
G = np.array([[-3, 4, 1],
              [1, -5, 1],
              [2, 1, -2]])

''' a) Calculate the stationary state of G'''

(lmbd, P) = eig(G, right=True)

n_st = np.argmin(abs(lmbd))
lmbd_st = lmbd[n_st]
P_st = P[:, n_st]
P_st = P_st/sum(P_st)
print('P_st analytic: ', P_st)

''' b) Time evolution of P '''

P_0 = np.array([1, 0, 0])
t_0 = 0
t_f = 2.5
dt = 0.01
t = np.arange(t_0, t_f, dt)

P = np.zeros([len(t), 3])
for i in range(len(t)):
    P[i, :] = np.dot(expm(t[i]*G), P_0)

print('P_st numeric: ', P[-1, :])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, P[:, 0], label='P_1', color='blue')
ax.plot(t, P[:, 1], label='P_2', color='red')
ax.plot(t, P[:, 2], label='P_3', color='green')
ax.set_xlabel('Time')
ax.set_ylabel('Probability')
ax.set_xlim([t[0], t[-1]])
ax.set_ylim([0, 1])
ax.minorticks_on()
ax.grid(True, which='major')
ax.grid(True, which='minor', linestyle='--')
ax.legend()
fig.savefig('markov.png', bbox='tight')

plt.show()

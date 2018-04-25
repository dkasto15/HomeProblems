import numpy as np
import matplotlib.pyplot as plt
from random import uniform


def main():
    ''' a) '''

    # # # Ensemble average # #
    N_ens = 10000 # nbr of realizations

    t_ens = np.linspace(0.1, 5, 200)
    x_ens = np.zeros(np.size(t_ens))
    for j in range(len(t_ens)):
        for i in range(N_ens):
            w = uniform(2, 6)
            x_ens[j] += 2 * np.sin(w * t_ens[j])
        x_ens[j] = 1 / float(N_ens) * x_ens[j]


    # # Time average # #
    N_t = 100000 # nbr of timesteps
    w_vec = np.linspace(2, 6, 100)
    x_t_bar = np.zeros(np.shape(w_vec))

    t = np.linspace(0, 100000, N_t) # vector with discrete timesteps

    for i in range(len(w_vec)):
        x_t_bar[i] = 1 / float(N_t) * np.sum(2 * np.sin(w_vec[i] * t))


    fig_a = plt.figure()
    ax_a_ens = fig_a.add_subplot(211)
    ax_a_ens.plot(t_ens, x_ens, color='blue', label='Simulated')
    ax_a_ens.plot(t_ens, (np.cos(2*t_ens)-np.cos(6*t_ens))/(2*t_ens), color='red', label='Analytic')
    ax_a_ens.set_xlabel('Time')
    ax_a_ens.set_ylabel('Ensemble average')
    ax_a_ens.set_xlim([0, t_ens[-1]])
    ax_a_ens.minorticks_on()
    ax_a_ens.grid(True, which='major')
    ax_a_ens.grid(True, which='minor', linestyle='--')
    ax_a_ens.legend()

    ax_a_time = fig_a.add_subplot(212)
    ax_a_time.plot(w_vec, x_t_bar, color='blue')
    ax_a_time.set_xlabel('Starting point in phase space')
    ax_a_time.set_ylabel('Time average')
    ax_a_time.set_xlim([w_vec[0], w_vec[-1]])
    ax_a_time.set_ylim([-1, 1])
    ax_a_time.minorticks_on()
    ax_a_time.grid(True, which='major')
    ax_a_time.grid(True, which='minor', linestyle='--')
    fig_a.subplots_adjust(hspace=0.3)
    fig_a.savefig('ensemble_average_a.png', bbox='tight')
    plt.show()
    # # # b) # # #
    w_0 = 1
    for j in range(len(t_ens)):
        for i in range(N_ens):
            theta = uniform(0, 2 * np.pi)  # Uniformly randomized number between 0 and 2*pi
            x_ens[j] += 2 * np.sin(w_0 * t_ens[j] + theta)  # Stoch. proc. val. at arb. time point
        x_ens[j] = 1 / float(N_ens) * x_ens[j]

    # # Time average # #
    theta_vec = np.linspace(0, 2 * np.pi, 100)
    for i in range(len(theta_vec)):
        x_t_bar[i] = 1 / float(N_t) * np.sum(2 * np.sin(w_0 * t + theta_vec[i]))

    fig_b = plt.figure()
    ax_b_ens = fig_b.add_subplot(211)
    ax_b_ens.plot(t_ens, x_ens, color='blue')
    ax_b_ens.set_xlabel('Time')
    ax_b_ens.set_ylabel('Ensemble average')
    ax_b_ens.set_xlim([0, t_ens[-1]])
    ax_b_ens.minorticks_on()
    ax_b_ens.set_ylim([-1, 1])
    ax_b_ens.grid(True, which='major')
    ax_b_ens.grid(True, which='minor', linestyle='--')
    ax_b_ens.legend()

    ax_b_time = fig_b.add_subplot(212)
    ax_b_time.plot(w_vec, x_t_bar, color='blue')
    ax_b_time.set_xlabel('Starting point in phase space')
    ax_b_time.set_ylabel('Time average')
    ax_b_time.set_xlim([w_vec[0], w_vec[-1]])
    ax_b_time.set_ylim([-1, 1])
    ax_b_time.minorticks_on()
    ax_b_time.grid(True, which='major')
    ax_b_time.grid(True, which='minor', linestyle='--')
    fig_b.subplots_adjust(hspace=0.3)
    fig_b.savefig('ensemble_average_b.png', bbox='tight')


if __name__ == '__main__':
    main()

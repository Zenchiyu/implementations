# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:09:07 2023

@author: steph

Ising model: model of magnetic material. Lattice of spins.

Use Markov Chain Monte Carlo to sample from the pdf of steady states.
"""
import numpy as np
import matplotlib.pyplot as plt
from mcmc import mcmc

def flip_spin(s, spin):
    new_s = s.copy()
    new_s[spin] = -new_s[spin]
    return new_s


if __name__ == "__main__":
    n = 20
    num_spins = n**2
    s = 2*np.random.randint(2, size=(num_spins, ))-1
    
    # Metropolis with uniform proposal distribution selecting one random neighbor
    # neighbors = lambda x: [flip_spin(s, spin) for spin in range(num_spins)]
    sample_candidate = lambda x: flip_spin(s, np.random.randint(num_spins))
    proposal_distr = lambda x_prime, x: 1/num_spins #if (np.count_nonzero(x_prime-x) == 1 and
                                                    #    np.linalg.norm(x_prime-x,ord=1)) else 0
    
    
    J = 1/4
    k_B_times_T = 0.1
    
    # sum_i s_i*sum_jneighbor s_j = sum_i, jneighbor s_i*s_j
    # left, right, up, down (don't always have a left (right.. up.. down..) closest neighbor)
    s_i_times_s_j = lambda s: s*(np.where(np.arange(num_spins)%n > 0, np.roll(s,1), 0)+
                                      np.where(np.arange(num_spins)%n < n-1, np.roll(s,-1), 0)+
                                      np.where(np.arange(num_spins) > n-1, np.roll(s,n), 0)+
                                      np.where(np.arange(num_spins) < num_spins-1-(n-1), np.roll(s,-n), 0))

    energy = lambda s, J: -J*np.sum(s_i_times_s_j(s))
    f_unnorm = lambda x: np.exp(-1/k_B_times_T*energy(x, J))
    
    num_steps = 500
    samples = mcmc(f_unnorm,
                    s,
                    algorithm="glauber",
                    sample_candidate=sample_candidate,
                    proposal_distr=proposal_distr,
                    tmax=num_steps*num_spins)
    
    fig, axs = plt.subplots(1, 5)
    axs[0].imshow(samples[0].reshape(n, n))
    axs[1].imshow(samples[20*num_spins].reshape(n, n))
    axs[2].imshow(samples[50*num_spins].reshape(n, n))
    axs[3].imshow(samples[-1].reshape(n, n))
    axs[4].imshow(s.reshape(n, n))

    
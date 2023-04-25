# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:31:58 2023

@author: steph
"""
import numpy as np
import matplotlib.pyplot as plt

from mcmc import mcmc
from matplotlib.colors import LogNorm

if __name__ == "__main__":
    xs = np.arange(200)/199
    XX, YY = np.meshgrid(xs, xs)
    # Pdf
    Cov = np.linalg.inv(np.array([[200, 40], [40, 200]]))  # negative correlation
    Cov_inv = np.array([[200, 40], [40, 20]])
    mu = np.array([[0.5], [0.5]])
    f_unnorm = lambda x: np.exp(-(Cov_inv[0, 0]*(x[0]-mu[0])**2 + 2*Cov_inv[0, 1]*(x[0]-mu[0])*(x[1]-mu[1]) + Cov_inv[1, 1]*(x[1]-mu[1])**2))
    # Same as:
    # f_unnorm = lambda x: np.exp(-(200*(x[0]-0.5)**2 + 2*40*(x[0]-0.5)*(x[1]-0.5) + 20*(x[1]-0.5)**2))
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 4),
                            sharey=True)
    cs = axs[0].contourf(XX, YY, f_unnorm([XX, YY]), extend="neither")
    axs[0].set_ylim([0, 1])
    axs[0].set_aspect("equal")

    # Metropolis with isotropic gaussian proposal distribution with std 0.1
    samples = mcmc(f_unnorm,
                    np.random.randint(200, size=(2, 1))/200,
                    tmax=200000)
    # # Metropolis with uniform proposal distribution selecting one random neighbor
    # # Out of 4 (in a 200 x 200 grid)
    # delta_x = delta_y = xs[1]-xs[0]
    # neighbors = lambda x: [np.array([[x[0] + delta_x], [x[1] + 0]]),
    #                         np.array([[x[0] + 0], [x[1] + delta_y]]),
    #                         np.array([[x[0] - delta_x], [x[1] + 0]]),
    #                         np.array([[x[0] + 0], [x[1] - delta_y]])]
    # isin = lambda vec, list_vec: np.any(list(map(lambda el: np.all(np.isclose(vec, el)),
    #                                           list_vec)))
    
    # sample_candidate = lambda x: neighbors(x)[np.random.randint(4)]
    # proposal_distr = lambda x_prime, x: 1/4 if isin(x_prime, neighbors(x)) else 0
    
    # samples = mcmc(f_unnorm,
    #                 np.random.randint(200, size=(2, 1))/200,
    #                 algorithm="metropolis-hastings",
    #                 sample_candidate=sample_candidate,
    #                 proposal_distr=proposal_distr)
    
    x = np.array(samples)[:, 0, :].flatten()
    y = np.array(samples)[:, 1, :].flatten()
    
    # Define the bin edges
    xbins = np.linspace(0, 1, 70)
    ybins = np.linspace(0, 1, 70)

    axs[1].hist2d(x, y,
                  bins=[xbins, ybins],
                  density=True,
                  cmin=0)
    axs[1].set_title("Metropolis:\n"+\
                        r"$g(x_{new}|x)$ gaussian: $\mu=x$, $\sigma=0.1$")
    axs[1].set_aspect("equal")
    plt.savefig("./images/mcmc_metropolis_2d.png")
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:27:44 2023

@author: steph

Markov Chain Monte Carlo to sample a normal distribution with some mean and std
using Metropolis Hasting algorithm (by default Metropolis algorithm with gaussian
                                    proposal distribution).


Note that the burn-in we chose was arbitrarily tmax//10 steps..
https://stats.stackexchange.com/questions/65614/in-mcmc-how-is-burn-in-time-chosen

One also needs to choose some proposal distribution such that don't reject
too often.
"""
import numpy as np
import matplotlib.pyplot as plt


def mcmc(f,
         x0,
         algorithm=None,
         tmax=100000,
         **kwargs):
    # Initial point
    x = x0
    samples = []
    
    if algorithm is None:
        # Metropolis with Isotropic gaussian proposal distribution
        # with std 0.1 by default..
        sample_candidate = lambda x: np.random.normal(x, 0.1)
        P_accept = lambda x_new, x: min(1, f(x_new)/f(x))
        
    elif algorithm == "metropolis-hastings":
        # Sampling from the proposal distribution g(x_new | x)
        sample_candidate = lambda x: kwargs["sample_candidate"](x)
        g = kwargs["proposal_distr"]
        P_accept = lambda x_new, x: min(1, f(x_new)*g(x, x_new)/(f(x)*g(x_new, x)))
    elif algorithm == "glauber":
        if "sample_candidate" in kwargs:
            sample_candidate = lambda x: kwargs["sample_candidate"](x)
        else:
            # Let's use the same proposal distribution as the default
            sample_candidate = lambda x: np.random.normal(x, 0.1)
        P_accept = lambda x_new, x: f(x_new)/(f(x)+f(x_new))
        
    for t in range(tmax):
        x_new = sample_candidate(x)
        # Metropolis Hastings rule, accept the selected 
        if np.random.uniform() < P_accept(x_new, x):
            x = x_new
        # ignore the first samples because not following distribution
        # we would want samples coming from the equilibrium distribution
        if t > tmax//10:
            samples.append(x)
    return samples

if __name__ == "__main__":
    xs = np.linspace(-4, 4, 100)
    # Pdf
    f_general = lambda x, mu, std: 1/np.sqrt(2*np.pi*std**2)*np.exp(-1/2*(x-mu)**2/std**2)
    f = lambda x: f_general(x, 0, 1)
    # "Unnormalized" pdf, just to show that Metropolis-Hastings works with it
    f_unnorm = lambda x: f(x)*10
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 6),sharex = True)
    
    # Metropolis with isotropic gaussian proposal distribution with std 0.1
    samples = mcmc(f_unnorm,
                    8*np.random.uniform()-4)
    axs[0, 0].plot(xs, f(xs), label=r"$\rho(x)$")
    axs[0, 0].hist(samples, bins=50, density=True,
             label="hist")
    axs[0, 0].legend()
    axs[0, 0].set_title("Metropolis:\n"+\
                        r"$g(x_{new}|x)$ gaussian: $\mu=x$, $\sigma=0.1$")
    
    # Metropolis with uniform proposal distribution g(x_new|x) with support
    # in [x-0.5, x+0.5]
    samples = mcmc(f_unnorm,
                    8*np.random.uniform()-4,
                    algorithm="metropolis-hastings",
                    sample_candidate=lambda x: np.random.uniform(x-0.5, x+0.5),
                    proposal_distr=lambda x_prime, x: int((x_prime <= x + 0.5) and (x_prime >= x - 0.5)))
    axs[0, 1].plot(xs, f(xs), label=r"$\rho(x)$")
    axs[0, 1].hist(samples, bins=50, density=True,
             label="hist")
    axs[0, 1].legend()
    axs[0, 1].set_title("Metropolis:\n"+\
                        r"$g(x_{new}|x)$ uniform on $[x-0.5, x+0.5]$")


    # Metropolis-Hastings
    samples = mcmc(f_unnorm,
                    8*np.random.uniform()-4,
                    algorithm="metropolis-hastings",
                    sample_candidate=lambda x: np.random.uniform(x-0.5, x+0.3),
                    proposal_distr=lambda x_prime, x: 10/8*int((x_prime >= x - 0.5) and (x_prime <= x + 0.3)))
    
    axs[1, 0].plot(xs, f(xs), label=r"$\rho(x)$")
    axs[1, 0].hist(samples, bins=50, density=True,
             label="hist")
    axs[1, 0].legend()
    axs[1, 0].set_title("Metropolis-Hastings:\n"+\
                        r"$g(x_{new}|x)$ uniform on $[x-0.5, x+0.3]$")
    
    
    samples = mcmc(f_unnorm,
                    8*np.random.uniform()-4,
                    algorithm="glauber")
    axs[1, 1].plot(xs, f(xs), label=r"$\rho(x)$")
    axs[1, 1].hist(samples, bins=50, density=True,
             label="hist")
    axs[1, 1].legend()
    axs[1, 1].set_title("Glauber:\n"+\
                        r"$g(x_{new}|x)$ gaussian: $\mu=x$, $\sigma=0.1$")
    
    plt.suptitle("Markov Chain Monte Carlo with $100$k steps with a burn-in of $10$k steps")
    plt.tight_layout()
    plt.savefig("./images/mcmc.png")
    
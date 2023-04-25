#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:58:37 2023

@author: zenchi

Iterative maps for representing graphically the evolution of
a discrete time dynamical system that can have 
a fixed point, limit cycle or chaos:

    - Linear map to see the fixed points:
        either attractive (stable), or repulsive (unstable)
    - Logistic map to see:
        - Limit cycle
        - Chaos (impredictibility and sensitive to initial cond.)
"""
import numpy as np
import matplotlib.pyplot as plt


def iterative_map(s_0, tmax, F, **kwargs):
    xs = np.linspace(0, 1, 100)
    # Plot the function F:
    plt.plot(xs, F(xs, **kwargs))
    # Plotting y=x
    plt.plot(xs, xs, "k", linestyle='--')
    
    s_t = s_0
    states = [s_0]
    plt.plot(s_0, 0, 'ro', label="start")
    
    plt.plot([s_0, s_0], [0, F(s_0, **kwargs)],
             "r")
    
    for i in range(tmax):
        s_t = F(s_t, **kwargs)
        states.append(s_t)
        
        # Plotting the point
        plt.plot(s_t, F(s_t, **kwargs), "r.")
        
        # Horizontal, go to y=x
        plt.plot([states[i], s_t], [s_t, s_t], "r")
        # Vertical
        plt.plot([s_t, s_t], [s_t, F(s_t, **kwargs)], "r")
    plt.axis("equal")
    
    
def iterate(s_0, tmax, F, **kwargs):
    s_t = s_0
    states = [s_0]
    bifurcations_xs = []
    
    for i in range(tmax):
        s_t = F(s_t, **kwargs)
        
        if np.any(np.isclose(s_t, states)) and not(np.any(np.isclose(s_t, bifurcations_xs, rtol=1e-4))):
            bifurcations_xs.append(s_t)
        
        states.append(s_t)
        
    return bifurcations_xs
    
def F_linear(s_t, **kwargs):
    return kwargs["gamma"]*s_t

def F_logistic(s_t, **kwargs):
    return kwargs["gamma"]*s_t*(1-s_t)

if __name__ == "__main__":
    tmax = 10
    s_0 = 0.45
    fig, axs = plt.subplots(2, 2)
    for i, gamma in enumerate([-0.5, 0.5, -1.1, 1.1]):
        plt.sca(axs[i//2, i%2])
        iterative_map(s_0, tmax, F_linear, gamma=gamma)
        plt.title(fr"$\gamma$={gamma}")
    plt.suptitle(r"Linear map $s_{t+1} = \gamma s_t$")
    plt.tight_layout()
    plt.savefig("./images/linear_maps.png")
    
    s_0 = 0.3
    fig, axs = plt.subplots(2, 2)
    for i, gamma in enumerate([3.1, 3.4, 3.5, 3.9]):
        iterative_map(s_0, tmax, F_logistic, gamma=gamma)
        plt.sca(axs[i//2, i%2])
        plt.title(fr"$\gamma$={gamma}")
    plt.suptitle(r"Logistic map $s_{t+1} = \gamma \cdot s_t \cdot (1 - s_t)$")
    plt.tight_layout()
    plt.savefig("./images/logistic_maps.png")
        
    # Bifurcation diagram
    # (according to my own definition of bifurcation)
    gammas = np.linspace(2.4, 4, 100)
    bs = [iterate(s_0, 1000, F_logistic, gamma=gamma)
          for gamma in gammas]
    plt.figure()
    for gamma, b in zip(gammas, bs):
        for x in b:
            plt.plot(gamma, x, '.')
    plt.title("My bifurcation diagram")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$x$")
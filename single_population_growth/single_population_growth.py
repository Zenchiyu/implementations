#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:32:23 2023

@author: zenchi

Two models, one with maximum capacity (modelled by
the logistic differential equation).

We plot both the analytical solution the numerical one
that use either the Euler method or the Runge Kutta (order 2)
method
"""
import numpy as np
import matplotlib.pyplot as plt


def P_simple(t, P0, r):
    # Population at time t for a given initial populatin P0
    # and the variation of P is proportional to the P (by
    # a coefficient r)
    return P0*np.exp(r*t)

def P_max_capacity(t, P0, r, M):
    # Population at time t for a given initial population P0
    # where P satisfies the logistic differential equation
    # meaning that the variation of P:
    # dP/dt = r*(1-P/M)*P where M is the maximum capacity
    return M/(1+(M-P0)/P0*np.exp(-r*t))

def euler(ts, P0, slope_func, delta_t=0.5):
    tmax = ts.max()
    ts_euler = [0]
    Ps_euler = [P0]
    t = delta_t
    logs = {"slopes": []}
    
    while t <= tmax:
        ts_euler.append(t)
        P = Ps_euler[-1]
        new_P = P + delta_t*slope_func(P)
        
        Ps_euler.append(new_P)
        logs["slopes"].append(slope_func(P))
        
        t += delta_t
        
    return ts_euler, Ps_euler, logs

def runge_kutta(ts, P0, slope_func, delta_t=0.5):
    tmax = ts.max()
    ts_rgk = [0]
    Ps_rgk = [P0]
    t = delta_t
    logs = {"slopes": []}
    
    while t <= tmax:
        ts_rgk.append(t)
        P = Ps_rgk[-1]
        P_tmp = P + delta_t*slope_func(P)  # just like in Euler
        new_P = P + delta_t/2*(slope_func(P) + slope_func(P_tmp))
        
        Ps_rgk.append(new_P)
        logs["slopes"].append(slope_func(P))
        
        t += delta_t
        
    return ts_rgk, Ps_rgk, logs

if __name__ == "__main__":
    ### Population model without maximum capacity
    M = 0.65*1.5
    r = 2
    ts = np.linspace(0, 4, 50)
    
    plt.figure(figsize=(8, 6))
    plt.plot(ts, P_simple(ts, 1.5*0.08, r),
             "r",
             label=fr"$P_0=${0.12}")
    plt.plot(ts, P_simple(ts, 1.5*0.8, r),
             "b",
             label=fr"$P_0=${1.2}")
    plt.xlim([0, 4])
    plt.legend()
    
    delta_t = 0.5  # for euler and runge kutta
    plt.title("Population Growth w/o maximum capacity\n"+\
              r"$\dot{P} = r \cdot P$"+\
                "\n"+\
              fr"$r=${r}, $\Delta t=${delta_t}")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$P(t)$")
    
    slope_func = lambda P: r*P
    # Euler method !!
    ts_euler, Ps_euler, logs = euler(ts, 0.12,
                                     slope_func,
                                     delta_t=delta_t)
    plt.plot(ts_euler, Ps_euler,
             color="r",
             marker=".",
             linestyle='-',
             label=fr"$P_0=${0.12}, Euler")
    plt.legend()
    
    ts_euler, Ps_euler, logs = euler(ts, 1.2,
                                     slope_func,
                                     delta_t=delta_t)
    plt.plot(ts_euler, Ps_euler,
             color="b",
             marker=".",
             linestyle='-',
             label=fr"$P_0=${1.2}, Euler")
    plt.legend()
    
    # Runge kutta of order 2 !!
    ts_rgk, Ps_rgk, logs = runge_kutta(ts, 0.12,
                                       slope_func,
                                       delta_t=delta_t)
    plt.plot(ts_rgk, Ps_rgk,
             color="r",
             marker='x',
             linestyle="--",
             label=fr"$P_0=${0.12}, Runge Kutta order 2")
    plt.legend()
    
    ts_rgk, Ps_rgk, logs = runge_kutta(ts, 1.2,
                                       slope_func,
                                       delta_t=delta_t)
    plt.plot(ts_rgk, Ps_rgk,
             color="b",
             marker="x",
             linestyle="--",
             label=fr"$P_0=${1.2}, Runge Kutta order 2")
    plt.legend()
    plt.savefig("./images/no_max_capacity.png")
    
    
    
    ### Population model using Logistic differential equation
    M = 0.65*1.5
    r = 2
    ts = np.linspace(0, 4, 50)
    
    plt.figure(figsize=(8, 6))
    plt.plot(ts, P_max_capacity(ts, 1.5*0.08, r, M),
             label=fr"$P_0=${0.12}")
    plt.plot(ts, P_max_capacity(ts, 1.5*0.8, r, M),
             label=fr"$P_0=${1.2}")
    plt.xlim([0, 4])
    plt.ylim([0, 1.5])
    plt.legend()
    
    delta_t = 0.5  # for euler and runge kutta
    plt.title("Population Growth with maximum capacity\n"+\
              r"$\dot{P} = r \cdot \left(1-\frac{P}{M}\right) \cdot P)$"+\
                "\n"+\
              fr"$M\approx${M:.3f}, $r=${r}, $\Delta t=${delta_t}")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$P(t)$")
    
    slope_func = lambda P: (r*(1-P/M)*P)
    # Euler method !!
    ts_euler, Ps_euler, logs = euler(ts, 0.12,
                                     slope_func,
                                     delta_t=delta_t)
    plt.plot(ts_euler, Ps_euler, '.-',
             label=fr"$P_0=${0.12}, Euler")
    plt.legend()
    
    ts_euler, Ps_euler, logs = euler(ts, 1.2,
                                     slope_func,
                                     delta_t=delta_t)
    plt.plot(ts_euler, Ps_euler, '.-',
             label=fr"$P_0=${1.2}, Euler")
    plt.legend()
    
    # Runge kutta of order 2 !!
    ts_rgk, Ps_rgk, logs = runge_kutta(ts, 0.12,
                                       slope_func,
                                       delta_t=delta_t)
    plt.plot(ts_rgk, Ps_rgk, 'x-',
             label=fr"$P_0=${0.12}, Runge Kutta order 2")
    plt.legend()
    
    ts_rgk, Ps_rgk, logs = runge_kutta(ts, 1.2,
                                       slope_func,
                                       delta_t=delta_t)
    plt.plot(ts_rgk, Ps_rgk, 'x-',
             label=fr"$P_0=${1.2}, Runge Kutta order 2")
    plt.legend()
    plt.savefig("./images/max_capacity.png")
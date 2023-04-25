# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:30:34 2023

@author: steph

Importance sampling and sampling importance resampling
"""
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    xs = np.linspace(-5, 10, 100)
    # Pdf (Laplacian distribution shifted a little bit)
    f_general = lambda x, mu, b: 1/(2*b)*np.exp(-np.abs(x-mu)/b)
    mu_laplace = 1.5
    b_laplace = 1
    f = lambda x: f_general(x, mu_laplace, b_laplace)
    
    # We want to compute the expected value of h(X) (not entropy) where h is:
    h = lambda x: 0.1*np.sin(x-1-(mu_laplace-2))
    
    # Let's say we can sample from a proposal distribution g (gaussian):
    g_general = lambda x, mu, std: 1/np.sqrt(2*np.pi*std**2)*np.exp(-1/2*(x-mu)**2/std**2)
    g = lambda x: g_general(x, 0, 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(xs, f(xs), "--",
             label=fr"$f=\rho$=Laplace({mu_laplace}, {b_laplace})")
    plt.plot(xs, g(xs),
             label=r"$g=\mathcal{N}$(0, 1)")
    plt.plot(xs, h(xs), "r",
             label=r"$h(x)$")
    plt.plot(xs, f(xs)*h(xs), "g", alpha=0.5,
             label=r"$\rho(x)\cdot h(x)$")
    plt.fill_between(xs, f(xs)*h(xs), y2=0, alpha=0.2)
   
    # Let's get samples according to g !!
    n_samples = 200
    samples = np.random.normal(0, 1, size=n_samples)
    # Compute sample mean based on the samples
    importance_weights = f(samples)/g(samples)
    sample_mean = (importance_weights*h(samples)).mean()
    print(sample_mean)
    
    
    plt.plot(samples, np.zeros_like(samples), "bx", alpha=0.5,
             label=r"$x_i$")
    plt.hlines(0, xs.min(), xs.max(), "k")
    plt.text(1.4, -0.05,
             r"$\mathbb{E}[h]=\int_{-\infty}^\infty \rho(x) h(x) dx \approx$"+\
             f"{np.trapz(f(xs)*h(xs), xs):.4f}", color="g")
        
    plt.text(1.4, -0.1,
             r"$\mathbb{E}[h] \approx \frac{1}{L} \sum_i \frac{\rho(x_i)}{g(x_i)} h(x_i)\approx$"+\
             f"{sample_mean:.4f}", color="b")
        
    plt.legend()
    plt.title(r"Importance sampling to estimate $\mathbb{E}[h]=\int_{-\infty}^\infty \rho(x) h(x) dx \approx$"+\
              f"{np.trapz(f(xs)*h(xs), xs):.4f}")
        
    ## Sampling importance resampling, let's use more samples
    n_samples = 10000
    samples = np.random.normal(0, 1, size=n_samples)
    # Compute sample mean based on the samples
    importance_weights = f(samples)/g(samples)
    sample_mean = (importance_weights*h(samples)).mean()
    print(sample_mean)
    
    resamples = [np.random.choice(samples, p=importance_weights/importance_weights.sum()) for _ in range(n_samples)]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, f(xs), "--",
             label=fr"$f=\rho$=Laplace({mu_laplace}, {b_laplace})")
    plt.plot(xs, g(xs),
             label=r"$g=\mathcal{N}$(0, 1)")
    
    # Samples according to g !!
    plt.plot(samples, -0.05*np.ones_like(samples), "bx", alpha=0.2,
             label=r"$x_i$")
    plt.plot(resamples, -0.025*np.ones_like(samples), "ro", alpha=0.2,
             fillstyle="none",
             label=r"$z_i$")
    plt.hist(samples, bins=60, density=True, alpha=0.5,
             label=r"histogram of $x_i$'s")
    plt.hist(resamples, bins=60, density=True, alpha=0.5,
             label=r"histogram of 'resamples' $z_i$'s")
    plt.hlines(0, xs.min(), xs.max(), "k")

        
    plt.legend()
    plt.title("Sampling importance resampling:\n"+\
              r"obtaining $z_i$'s from $x_i$'s and normalized importance weights")
    
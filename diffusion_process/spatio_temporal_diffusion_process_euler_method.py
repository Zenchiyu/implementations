# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:44:28 2023

@author: steph
"""
import numpy as np
import matplotlib.pyplot as plt


def apply_boundary_cond(s):
    s[:, 1] = 1
    s[-1, :] = 1

    s[:, -1] = 0
    s[0, :] = 0
    
if __name__ == "__main__":
    # Eulerian approach for the spatial component
    delta_x = 1
    delta_t = 0.1
    D = np.sqrt(delta_x**2/4)  # diffusion coefficient
    tmax = 10001
    
    s_0 = 0.2*np.ones((100, 100))
    apply_boundary_cond(s_0)
    s = s_0
    
    t = 0
    num_s = 0
    while t < tmax:
        s_up = np.roll(s, -1, axis=0)
        s_down = np.roll(s, 1, axis=0)
        s_left = np.roll(s, -1, axis=1)
        s_right = np.roll(s, 1, axis=1)
        
        pdv_2_x = 1/delta_x**2*(s_right - 2*s + s_left)
        pdv_2_y = 1/delta_x**2*(s_down - 2*s + s_up)
        new_s = s + delta_t*D*(pdv_2_x + pdv_2_y)
        apply_boundary_cond(new_s)
        s = new_s
        
        if num_s % 500 == 0:
            plt.imsave(f"./images/diffusion_t{t}.png", s)
        num_s += 1
        t += delta_t
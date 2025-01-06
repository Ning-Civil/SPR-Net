# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:08:19 2024

@author: cning
"""

import numpy as np

def hyst_envelope(disp_original, force_original, width_int, coeff_dy):
    # Calculate ultimate displacement and maximum force
    du = np.ceil(max(abs(disp_original)))
    fmax = np.ceil(max(abs(force_original)))
    dy = du * coeff_dy
    
    def compute_envelope(disp_original, force_original, int_points):
        env_disp = [0]
        env_force = [0]
        
        for i in range(len(int_points) - 1):
            mask = (disp_original - int_points[i])*(disp_original-int_points[i + 1])<0
            if np.any(mask):
                idx = np.argmax(force_original[mask])
                env_disp.append(disp_original[mask][idx])
                env_force.append(force_original[mask][idx])
        
        return env_disp, env_force
    
    # Positive envelope
    int_p_elastic = np.arange(0, dy + width_int, width_int)
    int_p_plastic = np.arange(dy, du + width_int, width_int)
    d_env_p_elastic, F_env_p_elastic = compute_envelope(disp_original, force_original, int_p_elastic)
    d_env_p_plastic, F_env_p_plastic = compute_envelope(disp_original, force_original, int_p_plastic)
    
    d_env_p = d_env_p_elastic + d_env_p_plastic[1:]  # combine elastic and plastic branches
    F_env_p = F_env_p_elastic + F_env_p_plastic[1:]
    
    # Negative envelope
    int_n_elastic = np.arange(0, -dy - width_int, -width_int)
    int_n_plastic = np.arange(-dy, -du - width_int, -width_int)
    d_env_n_elastic, F_env_n_elastic = compute_envelope(disp_original, -force_original, int_n_elastic)
    d_env_n_plastic, F_env_n_plastic = compute_envelope(disp_original, -force_original, int_n_plastic)
    
    d_env_n = d_env_n_elastic + d_env_n_plastic[1:]  # combine elastic and plastic branches
    F_env_n = F_env_n_elastic + F_env_n_plastic[1:]

    return d_env_p, F_env_p, d_env_n, F_env_n, du, fmax

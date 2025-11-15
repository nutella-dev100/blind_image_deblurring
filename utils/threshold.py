import numpy as np
from . import gradient

#thresholds
def threshold_dark_channel(D, w_k, xi):
    #returns p
    #pass xi from config.XI
    #D should be float 0-1 or 0-255
    threshold = w_k / xi
    p = np.where(D >= threshold, D, 0)
    return p

def threshold_gradient(grad, theta, lam):
    #returns g
    gh, gv = grad
    mag_sq = gradient.gradient_mag_sq(grad)
    threshold = theta / lam

    mask = (mag_sq >= threshold)
    g_h = gh * mask     #keep the value wherever condition is true
    g_v = gv * mask
    return g_h, g_v
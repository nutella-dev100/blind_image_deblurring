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

def threshold_gradient(g, theta, lam):
    gh, gv = g

    mag = np.sqrt(gh*gh + gv*gv)
    T = theta / (lam + 1e-8)

    mask = mag > T

    # KEEP original magnitude, only zero-out weak gradients
    gh_t = gh * mask
    gv_t = gv * mask

    return gh_t, gv_t

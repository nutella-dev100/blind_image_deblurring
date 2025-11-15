import numpy as np
from numpy.fft import fft2, ifft2
from utils import gradient

EPS = 1e-8


#fft based solvers.
def update_l(l, k, b, g, p, params):
    """
    Solve for l using FFT equation (18)
    """
    lam = params.LAMBDA
    xi = params.XI

    H, W = b.shape
    shape = (H, W)

    #ffts
    Fb = fft2(b)
    Fp = fft2(p)
    Fk = fft2(k, shape)

    wx = np.tile(np.fft.fftfreq(W), (H,1)) * (2*np.pi)
    wy = np.tile(np.fft.fftfreq(H), (W,1)).T * (2*np.pi)

    Dh = np.exp(-1j * wx) - 1
    Dv = np.exp(-1j * wy) - 1

    g_h, g_v = g
    Fgh = fft2(g_h)
    Fgv = fft2(g_v)

    Fg = np.conj(Dh) * Fgh + np.conj(Dv) * Fgv
    numerator = np.conj(Fk) * Fb + xi * Fp + lam * Fg
    denominator = (np.conj(Fk)*Fk) + lam*(np.conj(Dh)*Dh + np.conj(Dv)*Dv) + xi

    Fl = numerator / (denominator + EPS)
    l_new = np.real(ifft2(Fl))
    return l_new

def update_kernel(l, b, gamma):
    """
    Solve for k using FFT equation (22)
    """
    H, W = b.shape
    shape = (H, W)

    #finding gradients

    #horizontal
    grad_l_h = gradient.gradient_h(l)
    grad_b_h = gradient.gradient_h(b)

    #vertical
    grad_l_v = gradient.gradient_v(l)
    grad_b_v = gradient.gradient_v(b)

    #ffts
    Flh = fft2(grad_l_h)
    Flv = fft2(grad_l_v)
    Fbh = fft2(grad_b_h)
    Fbv = fft2(grad_b_v)

    numerator = np.conj(Flh) * Fbh + np.conj(Flv) * Fbv
    denominator = np.conj(Flh) * Flh + np.conj(Flv) * Flv + gamma

    Fk = numerator / (denominator + EPS) # Add EPS for numerical stability
    k = np.real(ifft2(Fk))
    return k


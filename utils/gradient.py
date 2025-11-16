import numpy as np

def gradient_h(I):
    return np.pad(I[:, 1:] - I[:, :-1], ((0,0),(0,1)))

def gradient_v(I):
    return np.pad(I[1:, :] - I[:-1, :], ((0,1),(0,0)))


def compute_gradients(img):
    #return tuple(grad_v, grad_h)
    gh = gradient_h(img)
    gv = gradient_v(img)
    return gh, gv

def gradient_mag_sq(grad):
    gh, gv = grad
    return gh**2 + gv**2
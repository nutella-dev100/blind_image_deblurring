import numpy as np

def gradient_h(img):
    return np.roll(img, -1, axis=1) - img

def gradient_v(img):
    return np.roll(img, -1, axis=0) - img

def compute_gradients(img):
    #return tuple(grad_v, grad_h)
    gh = gradient_h(img)
    gv = gradient_v(img)
    return gh, gv

def gradient_mag_sq(grad):
    gh, gv = grad
    return gh**2 + gv**2
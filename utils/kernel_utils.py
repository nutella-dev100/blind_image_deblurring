import numpy as np
from numpy.fft import fftshift
import cv2

#kernel utils
def normalise_kernel(k):
    s = k.sum()
    if s > 1e-8:
        return k / s
    return k

def threshold_small_values(k, thresh=1e-3):
    k = np.where(k < thresh, 0, k)
    return k

def clamp_kernel(k):
    k = np.clip(k, 0, None)
    return k

def center_kernel(k):
    return fftshift(k)

def crop_kernel(k):
    """
    Crop kernel to its minimal bounding box with nonzero entries.
    """
    # Boolean mask of nonzero
    nz = np.nonzero(k)

    if len(nz[0]) == 0:
        # fallback: keep original
        return k

    y_min, y_max = nz[0].min(), nz[0].max()
    x_min, x_max = nz[1].min(), nz[1].max()

    cropped = k[y_min:y_max+1, x_min:x_max+1]
    return cropped

def resize_kernel(k, target_shape):
    """
    Resize kernel to match the target shape (e.g., next pyramid level).
    """
    resized = cv2.resize(k, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0, None)
    return normalise_kernel(resized)

#one function as a wrapper
def clean_kernel(k):
    k = clamp_kernel(k)
    k = threshold_small_values(k)
    k = center_kernel(k)
    k = crop_kernel(k)
    k = normalise_kernel(k)
    return k
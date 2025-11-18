import numpy as np
from numpy.fft import fftshift, ifftshift
import cv2

#kernel utils
def normalise_kernel(k):
    s = k.sum()
    if s > 1e-8:
        return k / s
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
    # clamp negative values
    k = np.clip(k, 0, None)

    # === relative threshold (FIX) ===
    thr = max(1e-8, 1e-3 * k.max())
    k[k < thr] = 0.0

    # center kernel for convenience
    k = fftshift(k)

    # crop to bounding box
    k = crop_kernel(k)

    # normalize
    s = k.sum()
    if s > 1e-12:
        k = k / s
    else:
        # fallback to delta
        k = np.zeros_like(k)
        k[k.shape[0]//2, k.shape[1]//2] = 1.0

    return k


def pad_and_ifftshift_kernel(k_small, image_shape):
    """
    Place a small kernel at top-left then ifftshift so that its center
    is at (0,0) in frequency domain (correct for fft2).
    """
    H, W = image_shape
    padded = np.zeros((H, W), dtype=np.float32)
    kh, kw = k_small.shape
    padded[:kh, :kw] = k_small
    # Move center to top-left -> ifftshift will move center to origin for FFT
    return ifftshift(padded)

def postprocess_kernel_spatial(k_full):
    """
    After inverse FFT: shift center to middle, clamp negatives, threshold small,
    crop to minimal bbox, normalize and return kernel (small).
    """
    # move the center of kernel to middle
    from numpy.fft import fftshift
    k = fftshift(k_full.real)

    # clamp negatives to zero
    k = np.clip(k, 0, None)

    # threshold tiny values
    thr = max(1e-8, 1e-3 * k.max())
    k[k < thr] = 0.0


    # crop to bounding box
    nz = np.nonzero(k)
    if len(nz[0]) == 0:
        return k  # all zeros, fallback
    y0, y1 = nz[0].min(), nz[0].max()
    x0, x1 = nz[1].min(), nz[1].max()
    k_cropped = k[y0:y1+1, x0:x1+1]

    # normalize
    s = k_cropped.sum()
    if s > 1e-12:
        k_cropped /= s

    return k_cropped

def pad_kernel_centered(k, out_shape):
    H2, W2 = out_shape
    kh, kw = k.shape

    # Create empty padded kernel
    kpad = np.zeros((H2, W2), dtype=np.float32)

    # Compute center placement
    cx = (H2 - kh) // 2
    cy = (W2 - kw) // 2

    # Place kernel centered
    kpad[cx:cx+kh, cy:cy+kw] = k

    # Shift so that convolution kernel origin is at (0,0)
    kpad = np.fft.ifftshift(kpad)

    return kpad


def extract_kernel_center(k_full, expected_size=None):
    """
    After ifft2, k_full is image-sized. fftshift brings the kernel peak
    to the center. Then crop either:
    - the expected target size (kh, kw), OR
    - auto bbox if expected_size is None.
    """
    k = np.real(fftshift(k_full))   # center kernel

    # clip negatives
    k = np.clip(k, 0, None)
    # === relative threshold (FIX) ===
    thr = max(1e-8, 1e-3 * k.max())
    k[k < thr] = 0.0
    H, W = k.shape

    if expected_size is not None:
        kh, kw = expected_size
        cy, cx = H // 2, W // 2
        y0 = cy - kh // 2
        x0 = cx - kw // 2
        cropped = k[y0:y0 + kh, x0:x0 + kw]
    else:
        # auto-crop bounding box of non-zero values
        nz = np.nonzero(k)
        if len(nz[0]) == 0:
            # fallback to 1×1 delta
            return np.array([[1.0]], dtype=np.float32)
        y0, y1 = nz[0].min(), nz[0].max()
        x0, x1 = nz[1].min(), nz[1].max()
        cropped = k[y0:y1 + 1, x0:x1 + 1]

    # final normalization
    s = cropped.sum()
    if s <= 1e-12:
        # fallback delta
        out = np.zeros_like(cropped, dtype=np.float32)
        out[out.shape[0] // 2, out.shape[1] // 2] = 1.0
        return out

    cropped = cropped / s
    return cropped
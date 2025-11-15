import numpy as np
import cv2

#pyramid fns
def gaussian_pyramid(img, num_levels):
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")
    
    pyr = [img.copy()]
    for _ in range(1, num_levels):
        prev = pyr[-1]
        if prev.shape[0] < 2 or prev.shape[1] < 2:
            # cannot downsample further
            break
        down = cv2.pyrDown(prev)
        pyr.append(down)

    # We want coarse -> fine for the algorithm loop
    pyr_coarse_to_fine = pyr[::-1]
    return pyr_coarse_to_fine

def upsample_kernel(k, target_shape):
    """
    Upsample kernel `k` (2D) to the target image shape (H, W). Returns normalized kernel.
    Typically used to pad/resize a small kernel to the image FFT size or to the next pyramid level.
    target_shape: (H, W)
    """
    target_h, target_w = target_shape
    # Avoid zero-sized target
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_shape must be positive")

    # Resize kernel to target size using linear interpolation
    k_resized = cv2.resize(k, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Enforce non-negativity and normalize
    k_resized = np.clip(k_resized, 0, None)
    s = k_resized.sum()
    if s > 1e-12:
        k_resized = k_resized / s
    return k_resized

def upsample_l(l, target_shape):
    target_h, target_w = target_shape
    up = cv2.resize(l, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return up

def kernel_to_fft_size(k_small, image_shape):
    """
    Helper: embed small kernel into an image-sized array centered at (0,0) FFT convention.
    This pads a small kernel into the top-left corner (which corresponds to circular shift).
    If you prefer the kernel centered, use np.fft.ifftshift before fft2.
    """
    H, W = image_shape
    out = np.zeros((H, W), dtype=k_small.dtype)
    kh, kw = k_small.shape
    # place small kernel at top-left corner
    out[:kh, :kw] = k_small
    return out
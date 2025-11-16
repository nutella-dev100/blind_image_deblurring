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

def upsample_kernel(k, target_hw):
    """
    Resize kernel to given (H, W). Normalize afterwards.
    """
    th, tw = target_hw

    # cv2.resize uses (width, height)
    resized = cv2.resize(k, (tw, th), interpolation=cv2.INTER_LINEAR)

    resized = np.clip(resized, 0, None)
    s = resized.sum()
    if s > 1e-12:
        resized /= s
    else:
        # fallback delta
        ky, kx = th // 2, tw // 2
        resized = np.zeros((th, tw), dtype=np.float32)
        resized[ky, kx] = 1.0

    return resized

def upsample_small_kernel(k_small, scale_factor=2, max_size=None):
    """
    Upsample the small kernel by integer scale_factor (default 2).
    Keeps kernel small (not image-size). Returns normalized kernel.
    max_size: optional (h_max, w_max) to cap kernel growth.
    """
    kh, kw = k_small.shape
    new_kh = int(kh * scale_factor)
    new_kw = int(kw * scale_factor)

    if max_size is not None:
        mh, mw = max_size
        new_kh = min(new_kh, mh)
        new_kw = min(new_kw, mw)

    # if either dimension becomes <=0, fallback to same
    new_kh = max(1, new_kh)
    new_kw = max(1, new_kw)

    # cv2.resize expects (width, height)
    import cv2
    resized = cv2.resize(k_small, (new_kw, new_kh), interpolation=cv2.INTER_LINEAR)

    resized = np.clip(resized, 0, None)
    s = resized.sum()
    if s > 1e-12:
        resized = resized / s
    else:
        # fallback to centered delta
        resized = np.zeros((new_kh, new_kw), dtype=np.float32)
        resized[new_kh//2, new_kw//2] = 1.0

    return resized



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
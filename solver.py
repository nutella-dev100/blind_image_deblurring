import numpy as np
from numpy.fft import fft2, ifft2
from utils.kernel_utils import pad_kernel_centered, extract_kernel_center

EPS = 1e-8

def _crop_center(img, out_h, out_w):
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    y0 = cy - out_h // 2
    x0 = cx - out_w // 2
    return img[y0:y0 + out_h, x0:x0 + out_w]

def update_l(l, k, b, g, p, params):
    """
    Frequency-domain solve for latent image l.
    """
    lam = params.LAMBDA
    xi  = params.XI

    # Handle dimensions for FFT
    if b.ndim == 3:
        H, W, C = b.shape
        fft_axes = (0, 1)
    else:
        H, W = b.shape
        fft_axes = (0, 1)
    
    kh, kw = k.shape
    H2 = H + kh - 1
    W2 = W + kw - 1

    # Helper to pad
    def pad_img(img, h2, w2):
        if img.ndim == 3:
            out = np.zeros((h2, w2, img.shape[2]), dtype=np.float32)
            out[:img.shape[0], :img.shape[1], :] = img
        else:
            out = np.zeros((h2, w2), dtype=np.float32)
            out[:img.shape[0], :img.shape[1]] = img
        return out

    # ---- Pad image & priors ----
    bpad = pad_img(b, H2, W2)
    ppad = pad_img(p, H2, W2)
    
    Fb = fft2(bpad, s=(H2, W2), axes=fft_axes)
    Fp = fft2(ppad, s=(H2, W2), axes=fft_axes)

    # ---- Kernel FFT ----
    kpad = pad_kernel_centered(k, (H2, W2))
    Fk = fft2(kpad, s=(H2, W2)) 
    if b.ndim == 3: Fk = Fk[:, :, np.newaxis]

    # ---- Derivative Filters (Backward-ish) ----
    Dh_sp = np.zeros((H2, W2), dtype=np.float32)
    Dh_sp[0, 0] = -1.0
    if W2 > 1: Dh_sp[0, 1] = 1.0

    Dv_sp = np.zeros((H2, W2), dtype=np.float32)
    Dv_sp[0, 0] = -1.0
    if H2 > 1: Dv_sp[1, 0] = 1.0

    FDh = fft2(Dh_sp, s=(H2, W2))
    FDv = fft2(Dv_sp, s=(H2, W2))
    
    if b.ndim == 3:
        FDh = FDh[:, :, np.newaxis]
        FDv = FDv[:, :, np.newaxis]

    # ---- Gradient FFTs (With Alignment Fix) ----
    g_h, g_v = g
    
    # [FIX] RESTORED ROLL
    # Align Forward Diff (gradient.py) with Solver Diff ([-1, 1])
    # This prevents the image from drifting diagonally.
    g_h_shift = np.roll(g_h, -1, axis=1)
    g_v_shift = np.roll(g_v, -1, axis=0)

    gh_pad = pad_img(g_h_shift, H2, W2)
    gv_pad = pad_img(g_v_shift, H2, W2)
    
    Fgh = fft2(gh_pad, s=(H2, W2), axes=fft_axes)
    Fgv = fft2(gv_pad, s=(H2, W2), axes=fft_axes)

    # ---- Solve ----
    # fidelity to gradients
    Fg = np.conj(FDh) * Fgh + np.conj(FDv) * Fgv

    numerator = (np.conj(Fk) * Fb) + lam * Fg + xi * Fp
    denominator = (np.abs(Fk) ** 2) + lam * (np.abs(FDh) ** 2 + np.abs(FDv) ** 2) + xi
    denominator = np.maximum(denominator, 1e-2)

    Fl = numerator / denominator
    l_full = np.real(ifft2(Fl, axes=fft_axes))

    l_new = _crop_center(l_full, H, W)
    l_new = np.clip(l_new, 0.0, 1.0)

    return l_new

def update_kernel(l, b, gamma, image_shape, prev_k=None):
    H, W = image_shape[:2]

    # Compute gradients locally for consistency
    def get_grad(img):
        gh = np.zeros_like(img)
        gv = np.zeros_like(img)
        gh[:, :-1] = img[:, 1:] - img[:, :-1]
        gv[:-1, :] = img[1:, :] - img[:-1, :]
        return gh, gv

    grad_l_h, grad_l_v = get_grad(l)
    grad_b_h, grad_b_v = get_grad(b)

    # FFT (Axes 0, 1)
    fft_axes = (0, 1)
    Flh = fft2(grad_l_h, axes=fft_axes)
    Flv = fft2(grad_l_v, axes=fft_axes)
    Fbh = fft2(grad_b_h, axes=fft_axes)
    Fbv = fft2(grad_b_v, axes=fft_axes)

    # Numerator & Denominator
    num_ch = np.conj(Flh) * Fbh + np.conj(Flv) * Fbv
    denom_ch = (np.conj(Flh) * Flh) + (np.conj(Flv) * Flv)

    # Sum across RGB channels
    if l.ndim == 3:
        numerator = np.sum(num_ch, axis=2)
        denominator = np.sum(denom_ch, axis=2)
    else:
        numerator = num_ch
        denominator = denom_ch
    
    denominator += gamma
    Fk = numerator / (denominator + EPS)

    k_full = np.real(ifft2(Fk, axes=fft_axes))

    # Extract
    if prev_k is not None:
        expected = prev_k.shape
    else:
        expected = None

    k_small = extract_kernel_center(k_full, expected_size=expected)
    
    # Safety
    if k_small.sum() <= 1e-12:
        if prev_k is not None and prev_k.sum() > 1e-12:
            k_small = prev_k.copy()
        else:
            k_small = np.zeros((3,3), dtype=np.float32)
            k_small[1,1] = 1.0
            
    return k_small
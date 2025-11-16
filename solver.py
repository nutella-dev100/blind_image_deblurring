import numpy as np
from numpy.fft import fft2, ifft2
from utils import gradient
from utils.kernel_utils import pad_and_ifftshift_kernel, pad_kernel_centered, extract_kernel_center

EPS = 1e-8


#fft based solvers.
def _crop_center(img, out_h, out_w):
    H, W = img.shape
    cy, cx = H // 2, W // 2
    y0 = cy - out_h // 2
    x0 = cx - out_w // 2
    return img[y0:y0 + out_h, x0:x0 + out_w]

def update_l(l, k, b, g, p, params):
    """
    Frequency-domain solve for latent image l (corrected, padding-safe).
    """
    lam = params.LAMBDA
    xi  = params.XI

    H, W = b.shape
    kh, kw = k.shape

    # ---- safe padded convolution size ----
    H2 = H + kh - 1
    W2 = W + kw - 1

    # ---- explicit zero-padding ----
    bpad = np.zeros((H2, W2), dtype=np.float32)
    bpad[:H, :W] = b
    Fb = fft2(bpad)

    ppad = np.zeros((H2, W2), dtype=np.float32)
    ppad[:H, :W] = p
    Fp = fft2(ppad)

    # ---- pad kernel in top-left alignment ----
    # replace pad_kernel_centered(...) with this block
    kpad = np.zeros((H2, W2), dtype=np.float32)
    kpad[:kh, :kw] = k   # place kernel with origin at top-left (linear conv convention)
    Fk = fft2(kpad)


    # ---- spatial forward differences [-1, 1] ----
    Dh_sp = np.zeros((H2, W2), dtype=np.float32)
    Dh_sp[0,0] = -1.0
    if W2 > 1:
        Dh_sp[0,1] = 1.0

    Dv_sp = np.zeros((H2, W2), dtype=np.float32)
    Dv_sp[0,0] = -1.0
    if H2 > 1:
        Dv_sp[1,0] = 1.0

    FDh = fft2(Dh_sp)
    FDv = fft2(Dv_sp)

    # ---- gradient FFTs (also padded explicitly) ----
    g_h, g_v = g

    # shift gradients so they align with Dh/Dv
    g_h_shift = np.roll(g_h, -1, axis=1)
    g_v_shift = np.roll(g_v, -1, axis=0)

    gh_pad = np.zeros((H2, W2), dtype=np.float32)
    gh_pad[:H, :W] = g_h_shift
    Fgh = fft2(gh_pad)

    gv_pad = np.zeros((H2, W2), dtype=np.float32)
    gv_pad[:H, :W] = g_v_shift
    Fgv = fft2(gv_pad)


    # ---- gradient fidelity ----
    Fg = np.conj(FDh) * Fgh + np.conj(FDv) * Fgv

    # ---- numerator / denominator ----
    numerator = np.conj(Fk) * Fb + xi * Fp + lam * Fg
    denominator = (np.abs(Fk)**2) + lam*(np.abs(FDh)**2 + np.abs(FDv)**2) + xi
    denominator = np.maximum(denominator, 1e-2)

    # ---- debug print ----
    print("DBG FFT stats: |Fk| max:", np.max(np.abs(Fk)),
          " |FDh| max:", np.max(np.abs(FDh)),
          " |FDv| max:", np.max(np.abs(FDv)))
    print("DBG numerator: mean abs", np.mean(np.abs(numerator)),
          " max abs", np.max(np.abs(numerator)))
    print("DBG denominator: mean", np.mean(np.real(denominator)),
          " max", np.max(np.real(denominator)))

    # ---- solve ----
    Fl = numerator / (denominator + EPS)
    l_full = np.real(ifft2(Fl))

    # ---- crop TOP-LEFT, not center ----
    l_new = l_full[:H, :W]

    # ---- clip ----
    l_new = np.clip(l_new, 0.0, 1.0)

    print("max location in kpad:", np.unravel_index(np.argmax(kpad), kpad.shape))
    print("Fk phase at (0,0):", Fk[0,0])
    return l_new



def update_kernel(l, b, gamma, image_shape, prev_k=None):
    """
    Implements Eq. (22) — solve for kernel k in frequency domain.
    l, b : spatial images (latent, blurred)
    gamma : scalar regularizer
    image_shape : (H, W)
    prev_k : previous kernel (to estimate expected crop size)
    """

    H, W = image_shape

    # ==== 1) Compute gradients (forward differences) ====
    grad_l_h = np.zeros_like(l)
    grad_l_v = np.zeros_like(l)
    grad_b_h = np.zeros_like(b)
    grad_b_v = np.zeros_like(b)

    grad_l_h[:, :-1] = l[:, 1:] - l[:, :-1]
    grad_l_v[:-1, :] = l[1:, :] - l[:-1, :]
    grad_b_h[:, :-1] = b[:, 1:] - b[:, :-1]
    grad_b_v[:-1, :] = b[1:, :] - b[:-1, :]

    # ==== 2) FFT of gradients ====
    Flh = fft2(grad_l_h)
    Flv = fft2(grad_l_v)
    Fbh = fft2(grad_b_h)
    Fbv = fft2(grad_b_v)

    # ==== 3) numerator & denominator ====
    numerator = np.conj(Flh) * Fbh + np.conj(Flv) * Fbv
    denominator = (np.conj(Flh) * Flh) + (np.conj(Flv) * Flv) + gamma

    Fk = numerator / (denominator + EPS)

    # ==== 4) inverse FFT -> full-size kernel ====
    k_full = ifft2(Fk)

    # ==== 5) Extract small kernel from center ====
    if prev_k is not None:
        expected = prev_k.shape
    else:
        expected = None

    k_small = extract_kernel_center(k_full, expected_size=expected)
    if k_small.sum() <= 1e-12:
    # fallback: keep previous kernel if available
        if prev_k is not None and prev_k.sum() > 1e-12:
            k_small = prev_k.copy()
        else:
            # fallback 3x3 delta
            k_small = np.zeros((3,3), dtype=np.float32)
            k_small[1,1] = 1.0
    return k_small


import numpy as np

def threshold_dark_channel(l, D, w_k, xi):
    """
    Constructs the auxiliary variable p for the solver.
    l: Current latent image (H, W, C) or (H, W)
    D: Current dark channel (H, W)
    """
    threshold = w_k / xi
    
    # We want to penalize non-zero dark channel values.
    # Standard L0 logic: if error is small, force to 0. If large, keep it.
    # However, for blind deconv, we often use a mask approach.
    
    # 1. Create the mask where the Dark Channel is significant (violates sparsity)
    # Note: You may need to tune this comparison (>) vs (<) depending on your specific prior logic.
    # For standard sparsity: keep 'large' values (edges) and zero out 'small' (noise).
    mask = D > threshold 

    # 2. Initialize p as the current latent image
    if l.ndim == 3:
        # Broadcast mask to 3 channels
        mask_3d = mask[:, :, np.newaxis]
        p = l.copy()
        # For pixels that SHOULD be zero (small noise), we set them to 0.
        # For pixels that are edges/bright (large D), we keep them as l.
        p[~mask_3d] = 0 
    else:
        p = l.copy()
        p[~mask] = 0
        
    return p

def threshold_gradient(g, theta, lam):
    gh, gv = g
    mag = np.sqrt(gh*gh + gv*gv)
    T = theta / (lam + 1e-8)

    # Standard L0 Gradient thresholding:
    # Keep gradients that are large (edges), zero out small ones (texture/noise)
    mask = mag > T

    gh_t = gh * mask
    gv_t = gv * mask

    return gh_t, gv_t